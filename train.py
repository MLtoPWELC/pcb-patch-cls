"""训练 / 评估入口。

用法示例(单卡 GPU):
  python train.py --task multi9 --model lightpcb --aux none --epochs 60
  python train.py --task binary --model lightpcb --aux fft --epochs 60
  python train.py --task multi9 --model mobilenetv3_small --epochs 60   # 基线对照
CPU 快速自检:
  python train.py --task binary --model lightpcb --epochs 1 --max-steps 30 --bs 32 --smoke
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import paths
from torch.utils.data import DataLoader, Dataset

import models as M

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_device(spec):
    """auto 时按 CUDA > MPS > CPU 的顺序取第一个可用设备。"""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp(spec, device):
    """返回 None / 'fp16' / 'bf16'。CPU 上默认关闭: autocast 在无 AVX512-BF16 的机器上更慢。"""
    if spec == "off":
        return None
    if spec == "auto":
        if device.type != "cuda":
            return None
        try:
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        except Exception:
            return "fp16"
    return spec


def worker_init(_worker_id):
    """DataLoader 子进程默认继承主进程的线程数, N 个 worker × M 线程会在有限核数上严重超订。
    每个 worker 内的张量增强是单样本小算子, 单线程即可。"""
    torch.set_num_threads(1)


class PatchDS(Dataset):
    def __init__(self, csv, img_root, split, size, ncls, train=True, aug="light", degrade=0.25,
                 cache=None, gray=False, img_split=None, shift=0):
        root = Path(img_root)
        # COCO 布局下图片在 train2017 / val2017 子目录; 若已直接放平则回退到 root
        # 交叉验证时折内验证集是从 train2017 切出来的, 图片目录要单独指定,
        # 仍按 split 去找会跑到 val2017 下, 直接 FileNotFoundError。
        img_split = img_split or split
        sub = root / ("train2017" if img_split == "train" else "val2017")
        self.root = sub if sub.is_dir() else root
        self.size = size
        self.ncls = ncls
        self.train = train
        self.rows = []
        with open(csv, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                p = line.strip().split(",")
                if not p or p[0] != split:
                    continue
                self.rows.append((p[1], int(p[2]), int(p[3]), int(p[4]), p[5], p[6]))
        self.aug = aug
        self.degrade = degrade
        self.gray = gray
        self.shift = shift
        # 预解码缓存(可选): 命中则直接从 memmap 取 patch, 省掉每个 epoch 的 JPEG 解码。
        # 行数必须与 csv 一致, 否则视为过期缓存并自动回退到在线解码。
        #
        # 这里只记路径与 shape, 不在主进程打开 memmap: DataLoader 用 spawn 起 worker 时
        # 会把整个 dataset pickle 过去, 若把 memmap 挂在实例上, 每个 worker 都会
        # 复制一份完整的缓存数组(二分类 873 MB × workers), 既慢又会随机崩在
        # "OSError: [Errno 22] Invalid argument / pickle data was truncated"。
        # __getstate__ 里也会丢掉已打开的句柄, worker 首次取值时各自只读映射, 不复制。
        self.cache = None          # 非 None 表示"有缓存", 供外部判断
        self._u8 = None
        self._shape = None
        self._mmap = None
        self._idx = None           # 抽样用的行索引(smoke 模式), None 表示全部
        if cache:
            cands = sorted(Path(cache).glob(f"{Path(csv).stem}_{split}_*.json"))
            if cands:
                meta = json.load(open(cands[0], encoding="utf-8"))
                u8 = cands[0].with_suffix(".u8")
                if u8.exists() and meta.get("n") == len(self.rows):
                    c = meta["size"]
                    self._u8 = u8
                    self._shape = (meta["n"], c, c, 3)
                    self.cache = True
                    print(f"[cache] {split}: {u8.name} {meta['n']} 条 @ {c}x{c}")
                else:
                    print(f"[cache] {split}: 缓存与 csv 行数不符({meta.get('n')} vs {len(self.rows)}), "
                          f"已忽略, 建议重跑 prepare_cache.py")
        if ncls == 1:
            # 二分类: 正样本=1, 任何负样本=0(含困难/简单两类负样本)
            self.labels = np.array([[1.0 if r[5] == "pos" else 0.0] for r in self.rows], dtype=np.float32)
        else:
            self.labels = np.array([[int(c) for c in r[4]] for r in self.rows], dtype=np.float32)

    def _cache_arr(self):
        """惰性打开 memmap。在 worker 进程里才会真正执行, 主进程不占句柄。"""
        if self._mmap is None:
            self._mmap = np.memmap(self._u8, dtype=np.uint8, mode="r", shape=self._shape)
        return self._mmap if self._idx is None else self._mmap[self._idx]

    def subsample(self, step):
        """按步长抽样行。缓存索引必须同步抽样, 否则图文错位(不报错, 但标签全乱)。"""
        self.rows = self.rows[::step]
        self.labels = self.labels[::step]
        if self._shape is not None:
            self._idx = np.arange(self._shape[0])[::step]

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mmap"] = None          # 不要把句柄/数据带进 worker
        return d

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, x, y, s, lab, kind = self.rows[i]
        if self.cache is not None:
            im = Image.fromarray(np.ascontiguousarray(self._cache_arr()[i]))
        else:
            im = Image.open(self.root / fn).convert("RGB")
            im = im.crop((x, y, x + s, y + s))
        # 灰度消融: AOI 用单色光源成像, RGB 三通道高度相关, 颜色本身可能不携带判别信息。
        # 转 L 放在 resize 之前: 双线性插值对加权求和(灰度化)可交换, 结果等价但少做两倍计算。
        if self.gray:
            im = im.convert("L")
        # 分辨率退化增强: 正样本有 14.5% 来自 96~226 的窗口下采样, 负样本多为原尺寸裁切。
        # 对两类以同一概率做"先降后升"重采样, 让清晰度统计与类别无关, 消除捷径特征。
        if self.train and self.degrade > 0 and random.random() < self.degrade:
            r = int(round(self.size * random.uniform(0.55, 1.15)))
            im = im.resize((max(24, r), max(24, r)), Image.BICUBIC)
        im = im.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        if self.gray:
            t = torch.from_numpy(arr.copy()).unsqueeze(0) / 255.0
        else:
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1) / 255.0
        if self.train:
            if random.random() < 0.5:
                t = torch.flip(t, dims=[2])
            if random.random() < 0.5:
                t = torch.flip(t, dims=[1])
            if self.aug == "strong":
                # 小角度旋转 + 亮度对比度扰动(用可微外的简单实现)
                ang = random.uniform(-12, 12) * math.pi / 180
                c, s_ = math.cos(ang), math.sin(ang)
                theta = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0]], dtype=torch.float32)
                grid = F.affine_grid(theta.unsqueeze(0), (1, 3, self.size, self.size), align_corners=False)
                t = F.grid_sample(t.unsqueeze(0), grid, align_corners=False).squeeze(0)
                t = (t * random.uniform(0.85, 1.15)).clamp(0, 1)
                t = ((t - 0.5) * random.uniform(0.85, 1.15) + 0.5).clamp(0, 1)
        # 随机平移增强: 让缺陷不必总在窗口中心。用 reflect 填充而不是零填充,
        # 零填充会在边缘造出"黑框", 模型可能把它当成线索。
        if self.train and self.shift > 0:
            d = self.shift
            dy, dx = random.randint(-d, d), random.randint(-d, d)
            t = F.pad(t.unsqueeze(0), (d, d, d, d), mode="reflect").squeeze(0)
            t = t[:, d + dy:d + dy + self.size, d + dx:d + dx + self.size]
        # MEAN/STD 是三通道形状, 直接用在单通道张量上会广播成 (3,H,W),
        # 表现为"模型要 1 通道、输入却是 3 通道"的运行时报错。灰度走标量归一化。
        if self.gray:
            t = (t - 0.5) / 0.25
        else:
            t = (t - MEAN) / STD
        target = torch.tensor(self.labels[i], dtype=torch.float32)
        return t, target


def binary_auc(y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for gi in range(len(cnt)):
        idx = np.where(inv == gi)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(y, s):
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    rec = tp / max(1, y.sum())
    # 标准 AP: 每个正样本处的 precision 平均
    if y.sum() == 0:
        return float("nan")
    return float((prec * y).sum() / y.sum())


def f1_at(y, p, thr=0.5):
    pred = (np.asarray(p) >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec)), float(prec), float(rec)


def fpr_at_tpr(y, s, target=0.95):
    """AOI 关注的指标: 召回锁到 target 时的误报率。"""
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order]
    s = np.asarray(s)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / max(1, y.sum())
    idx = np.where(rec >= target)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    return float(fp[i] / max(1, (1 - y).sum()))


def predict(model, loader, device):
    """前向一次, 返回 (probs, targets)。后续可用不同阈值反复算指标而不必重跑前向。"""
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device, non_blocking=True))
            logits.append(out.float().cpu())
            ys.append(y)
    z = torch.cat(logits).numpy()
    ys = torch.cat(ys).numpy()
    return 1 / (1 + np.exp(-z)), ys


def metrics_from(probs, ys, ncls, thresh=0.5):
    """由概率算指标。多标签口径下 thresh 对 acc/subsetAcc/mF1 生效, mAP 与阈值无关。"""
    if ncls == 1:
        y = ys[:, 0].astype(int)
        p = probs[:, 0]
        f1, prec, rec = f1_at(y, p, thresh)
        return {
            "acc": float(((p >= thresh).astype(int) == y).mean()),
            "f1": f1, "precision": prec, "recall": rec,
            "auc": binary_auc(y, p),
            "fpr@r95": fpr_at_tpr(y, p, 0.95),
            "n": int(len(y)), "pos_rate": float(y.mean()),
        }
    aps, f1s = [], []
    per_class = {}
    for c in range(ncls):
        ap = average_precision(ys[:, c], probs[:, c])
        f1, _, _ = f1_at(ys[:, c], probs[:, c], thresh)
        aps.append(ap)
        f1s.append(f1)
        per_class[c] = {"ap": ap, "f1": f1, "pos": int(ys[:, c].sum())}
    pred = (probs >= thresh).astype(int)
    subset_acc = float((pred == ys).all(axis=1).mean())
    # 逐标签准确率: 所有样本 × 9 个标签位置的平均正确率, 多标签下比 subset_acc 更可读
    label_acc = float((pred == ys).mean())
    per_class_acc = {c: float((pred[:, c] == ys[:, c]).mean()) for c in range(ncls)}
    return {"mAP": float(np.nanmean(aps)), "macro_f1": float(np.nanmean(f1s)),
            "label_acc": label_acc, "subset_acc": subset_acc,
            "per_class": per_class, "per_class_acc": per_class_acc,
            "mean_pos": float(pred.sum(axis=1).mean()), "n": int(len(ys)),
            "thresh": float(thresh)}


def evaluate(model, loader, ncls, device, thresh=0.5):
    probs, ys = predict(model, loader, device)
    return metrics_from(probs, ys, ncls, thresh)


def sweep_threshold(probs, ys, ncls, lo=0.05, hi=0.96, step=0.05):
    """在验证集上扫一个全局阈值, 取 macro-F1 最高者。

    多标签 BCE 下各类先验差异大, 0.5 往往使模型过度预测(mean_pos 远大于真值均标签数)。
    只在验证集上调一个标量并如实报告, 不逐类调, 避免过拟合验证集。
    """
    best_t, best_f1, best_m = 0.5, -1.0, None
    t = lo
    while t <= hi + 1e-9:
        m = metrics_from(probs, ys, ncls, t)
        if m["macro_f1"] > best_f1:
            best_t, best_f1, best_m = t, m["macro_f1"], m
        t += step
    return best_t, best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multi9", "binary"], default="multi9")
    ap.add_argument("--model", default="lightpcb")
    ap.add_argument("--aux", default="none", choices=["none", "fft", "hf"])
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--se", dest="se", action="store_true", default=False,
                    help="启用 SE 通道注意力。默认关闭: 消融实测 SE 用 +17.5k 参数换来 "
                         "0 收益(0.7964 -> 0.7930), 故不作为主模型的一部分")
    ap.add_argument("--no-se", dest="se", action="store_false",
                    help="旧参数, 与默认行为相同, 保留兼容")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--img-root", default=paths.DATA_ROOT,
                    help="Data_COCO 目录；默认取自环境变量 PCB_DATA 或 dataset.path")
    ap.add_argument("--input-size", type=int, default=96)
    ap.add_argument("--rep", action="store_true",
                    help="重参数化: 训练时 dw 位并行 3x3/1x1/恒等 三分支, 训练结束自动合并成单分支")
    ap.add_argument("--depth", type=int, default=2, choices=[1, 2],
                    help="每级堆几个块。1=只留下采样块, 参数减半, 但必须靠空洞补回感受野")
    ap.add_argument("--dilation", default="1,2,4",
                    help="三个阶段空洞率, 逗号分隔; 传 1,1,1 即关闭空洞卷积(消融用)")
    ap.add_argument("--img-split", default=None, choices=["train", "val"],
                    help="图片所在子目录; 默认跟随 split。交叉验证时折内验证集来自 train2017, 需显式指定为 train")
    ap.add_argument("--gray", action="store_true",
                    help="把输入转成单通道灰度。AOI 单色光源成像, RGB 三通道高度相关; "
                         "实测参数量只降 0.16%%, 主要影响是首层计算量与颜色冗余")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="每隔几个 epoch 在验证集上评估一次, 1=每个 epoch 都评估, 0=只在最后一个 epoch")
    ap.add_argument("--tune-thresh", dest="tune_thresh", action="store_true", default=True,
                    help="训练结束后在验证集上校准一个全局阈值(默认开)")
    ap.add_argument("--no-tune-thresh", dest="tune_thresh", action="store_false",
                    help="关闭阈值校准")
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--aug", default="strong", choices=["light", "strong"])
    ap.add_argument("--shift", type=int, default=0,
                    help="训练时随机平移的最大像素; 0=关闭。用于削弱缺陷-中心的位置先验")
    ap.add_argument("--degrade", type=float, default=0.25, help="分辨率退化增强概率(两类同概率)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4, help="每个 worker 预取的 batch 数")
    ap.add_argument("--cache", default="data/cache", help="预解码缓存目录, 用 --no-cache 关闭")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:1 / mps")
    ap.add_argument("--amp", default="auto", choices=["auto", "off", "fp16", "bf16"],
                    help="auto: CUDA 上自动选 bf16 或 fp16, CPU 上关闭")
    ap.add_argument("--latency-device", default="cpu",
                    help="测时延用的设备, 默认固定为 cpu: 论文里的时延必须跨模型同设备可比, "
                         "在 GPU 上训却报 GPU 时延会误导部署评估")
    ap.add_argument("--threads", type=int, default=0, help="CPU 计算线程数, 0 表示沿用 torch 默认")
    ap.add_argument("--channels-last", action="store_true", help="NHWC 布局, 一般只在 CUDA 上有收益")
    ap.add_argument("--compile", action="store_true", help="torch.compile, 环境不支持时自动回退")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="只跑少量样本, 用于环境自检")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = resolve_amp(args.amp, device)
    if args.threads:
        torch.set_num_threads(args.threads)
    ncls = 1 if args.task == "binary" else 9
    csv = args.csv or f"data/patches_{args.task}.csv"
    if not Path(csv).exists():
        raise SystemExit(f"找不到 {csv}, 先跑 prepare_data.py --task {args.task}")
    # 数据目录换机器后最容易失效, 这里提前拦下并给出修正路径, 避免在 worker 里抛 FileNotFoundError
    if not (Path(args.img_root) / "annotations").is_dir():
        paths.check(root=args.img_root)
        raise SystemExit(f"\n图片目录不可用: {args.img_root}")
    if args.workers and args.threads and args.workers * 1 + args.threads > (os.cpu_count() or 8):
        print(f"[warn] workers({args.workers}) + threads({args.threads}) 超过核数 {os.cpu_count()}, 建议 --workers {max(1, (os.cpu_count() or 8) - args.threads)}")

    cache_dir = None if args.no_cache else args.cache
    tr = PatchDS(csv, args.img_root, "train", args.input_size, ncls, train=True,
                 aug=args.aug, degrade=args.degrade, cache=cache_dir, gray=args.gray,
                 img_split=args.img_split, shift=args.shift)
    va = PatchDS(csv, args.img_root, "val", args.input_size, ncls, train=False,
                 cache=cache_dir, gray=args.gray, img_split=args.img_split, shift=args.shift)
    if args.smoke:
        # 按步长抽样而不是取前 N 条: CSV 里正样本排在前面, 取前 N 条会得到单一类别。
        # subsample 会把缓存索引一起抽样, 否则索引与标签错位(不报错, 但图文对不上)。
        tr.subsample(5)
        va.subsample(5)
    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=(device.type == "cuda"), drop_last=False,
                       persistent_workers=args.workers > 0, worker_init_fn=worker_init,
                       prefetch_factor=(args.prefetch if args.workers > 0 else None))
    # 验证集不需要梯度, batch 放大以减少批次与进程调度开销(逐 epoch 评估时差别明显)
    bs_va = max(args.bs, 256)
    dl_va = DataLoader(va, batch_size=bs_va, shuffle=False, num_workers=args.workers,
                       pin_memory=(device.type == "cuda"), worker_init_fn=worker_init,
                       persistent_workers=args.workers > 0,
                       prefetch_factor=(args.prefetch if args.workers > 0 else None))

    in_ch = 1 if args.gray else 3
    kw = {}
    if args.model == "lightpcb":
        kw = {"width": args.width, "aux": args.aux, "se": args.se, "in_ch": in_ch,
              "dil": tuple(int(x) for x in args.dilation.split(",")),
              "depth": args.depth, "rep": args.rep}
    model = M.build_model(args.model, ncls, **kw).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[info] torch.compile 已启用")
        except Exception as e:
            print(f"[warn] torch.compile 不可用, 回退 eager 模式: {type(e).__name__}")
    nparam = M.count_params(model)
    flops = M.count_flops(model, (1, in_ch, args.input_size, args.input_size))
    # 时延固定在与训练设备解耦的 lat_dev 上测: 用 deepcopy 避免把训练模型搬离 GPU。
    lat_dev = torch.device(args.latency_device)
    if lat_dev.type != device.type:
        import copy
        _m_lat = copy.deepcopy(model).to(lat_dev)
        lat = M.measure_latency(_m_lat, (1, in_ch, args.input_size, args.input_size), device=lat_dev)
        del _m_lat
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        lat = M.measure_latency(model, (1, in_ch, args.input_size, args.input_size), device=device)
        model.to(device)
    print(f"[cfg] {device} amp={amp or 'off'} bs={args.bs} workers={args.workers} "
          f"threads={torch.get_num_threads()} chlast={args.channels_last} compile={args.compile} | "
          f"params={nparam:,} macs={flops:.2f}M lat(bs1,{lat_dev.type})={lat:.2f}ms "
          f"train={len(tr)} val={len(va)}")

    # 类别不均衡: 用正样本频率的倒数做 pos_weight
    if ncls > 1:
        freq = tr.labels.mean(axis=0).clip(1e-3, 1 - 1e-3)
        pos_weight = torch.tensor((1 - freq) / freq, dtype=torch.float32, device=device)
    else:
        p = tr.labels[:, 0].mean()
        pos_weight = torch.tensor([(1 - p) / max(p, 1e-3)], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(dl_tr))
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.25)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp == "fp16"))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = torch.float16 if amp == "fp16" else torch.bfloat16

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.model}_aux-{args.aux}_w{args.width}_s{args.seed}"
    # SE 是消融项, 开启时加后缀区分, 否则与主模型写同一个 json/pt 互相覆盖
    if args.se:
        tag += "_se"
    # 灰度和非 96 输入也是消融项, 必须进 tag, 否则会覆盖主实验的结果文件
    if args.gray:
        tag += "_gray"
    if args.input_size != 96:
        tag += f"_in{args.input_size}"
    if args.dilation != "1,2,4":
        tag += "_dil" + args.dilation.replace(",", "")
    if args.depth != 2:
        tag += f"_d{args.depth}"
    if args.rep:
        tag += "_rep"
    if args.shift > 0:
        tag += f"_sh{args.shift}"
    if args.csv and ("random" in Path(args.csv).stem or "randneg" in Path(args.csv).stem):
        tag += "_randneg"
    best = -1
    best_metrics = None
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        run, nb = 0.0, 0
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp is not None):
                out = model(x)
                loss = lossf(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            nb += 1
            if args.max_steps and i + 1 >= args.max_steps:
                break
        if args.eval_every > 0 and (ep % args.eval_every == 0 or ep == args.epochs):
            m = evaluate(model, dl_va, ncls, device)
            key = m.get("mAP", m.get("auc", 0.0))
            if key is not None and not math.isnan(key) and key > best:
                best, best_metrics = key, m
                torch.save(model.state_dict(), out_dir / f"{tag}.pt")
                mark = " *"
            else:
                mark = ""
            print(f"ep{ep:03d} loss {run/max(nb,1):.4f} | " +
                  (f"acc {m['label_acc']:.4f} subsetAcc {m['subset_acc']:.4f} "
                   f"mAP {m['mAP']:.4f} mF1 {m['macro_f1']:.4f} "
                   f"npos {m['mean_pos']:.2f}"
                   if ncls > 1 else
                   f"acc {m['acc']:.4f} F1 {m['f1']:.4f} AUC {m['auc']:.4f} FPR@R95 {m['fpr@r95']:.4f}") +
                  f" | {time.time() - t_ep:.1f}s{mark}")
        else:
            print(f"ep{ep:03d} loss {run/max(nb,1):.4f} | {time.time() - t_ep:.1f}s")

    # ---- 训练结束后, 用最优权重在验证集上校准一个全局阈值 ----
    # 多标签 BCE 下 0.5 常导致过度预测(mean_pos 远大于真值均标签数), subsetAcc 被严重压低。
    # 只调一个标量阈值, 且仅用验证集, 并把 t 一并写入结果, 保证可复核。
    tuned = None
    ckpt = out_dir / f"{tag}.pt"
    if args.tune_thresh and ncls > 1 and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        probs, ys_v = predict(model, dl_va, device)
        t_best, m_best = sweep_threshold(probs, ys_v, ncls)
        tuned = {"thresh": t_best, **{k: v for k, v in m_best.items() if k != "per_class"}}
        print(f"阈值校准: t={t_best:.2f} 时 macro-F1 最高 "
              f"-> acc {m_best['label_acc']:.4f} subsetAcc {m_best['subset_acc']:.4f} "
              f"mF1 {m_best['macro_f1']:.4f} npos {m_best['mean_pos']:.2f} "
              f"(0.5 阈值下 subsetAcc {best_metrics.get('subset_acc', float('nan')):.4f})"
              if best_metrics else "")

    # 重参数化: 训练结束后把多分支合并成单分支, 并核对等价性。
    # 合并是精确等价的线性变换, 若误差超阈值说明实现有错, 直接中止以免保存坏模型。
    if args.rep:
        model.eval()
        _xb = torch.randn(min(8, args.bs), in_ch, args.input_size, args.input_size,
                          device=device)
        with torch.no_grad():
            _y0 = model(_xb)
        model.switch_to_deploy()
        with torch.no_grad():
            _y1 = model(_xb)
        _err = float((_y0 - _y1).abs().max())
        print(f"[rep] 合并前后最大输出差 {_err:.2e}")
        if _err > 1e-3:
            raise SystemExit(f"重参数化合并不等价 (差 {_err:.2e}), 拒绝保存")
        nparam = M.count_params(model)
        flops = M.count_flops(model, (1, in_ch, args.input_size, args.input_size))
        import copy as _copy
        _m = _copy.deepcopy(model).to(lat_dev)
        lat = M.measure_latency(_m, (1, in_ch, args.input_size, args.input_size),
                                device=lat_dev)
        del _m
        print(f"[rep] 合并后 params={nparam:,} macs={flops:.2f}M "
              f"lat(bs1,{lat_dev.type})={lat:.2f}ms")

    res = {
        "tag": tag, "task": args.task, "model": args.model, "aux": args.aux,
        "width": args.width, "se": args.se, "input_size": args.input_size,
        "gray": bool(args.gray), "csv": str(csv), "dilation": args.dilation,
        "depth": args.depth, "rep": bool(args.rep),
        "params": int(nparam), "macs_M": float(flops),
        "latency_ms": float(lat), "latency_device": lat_dev.type,
        "device": str(device), "amp": amp or "off",
        "channels_last": bool(args.channels_last), "compile": bool(args.compile),
        "threads": torch.get_num_threads(), "workers": args.workers, "bs": args.bs,
        "train_minutes": round((time.time() - t_start) / 60, 2),
        "sec_per_epoch": round((time.time() - t_start) / args.epochs, 1),
        "seed": args.seed, "best": best_metrics, "tuned": tuned,
    }
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "best"}, ensure_ascii=False, indent=2))
    print("best:", json.dumps(best_metrics, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
