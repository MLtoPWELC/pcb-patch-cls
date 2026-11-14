"""交叉验证: 解决"阈值在验证集上调、指标又在同一个验证集上报"的乐观偏差。

做法
    官方 train/val 划分保持不动(原数据集是按板分的, 打乱会泄漏)。
    在 train 内部切 5 折: 4 折训练 + 1 折校准阈值, 然后把得到的阈值
    拿到官方 val 上评估一次。val 全程只用于最终报告, 不参与任何选择。

    这样每个折都给出一个"没在评估集上见过"的指标, 5 折汇总成 mean±std,
    可以直接写进论文, 而不是单点数字。

用法
    python cv_eval.py --task multi9 --folds 5
    python cv_eval.py --task binary --folds 5
    python cv_eval.py --task multi9 --folds 5 --reuse    # 复用已训好的折, 只重新评估
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import models as M
import paths
from train import PatchDS, metrics_from, predict


CV_DIR = Path("data/cv")
LOCK = Path("data/cv/.cv_eval.lock")


def acquire():
    """原子锁: 先写临时文件再 os.link, 同时启动两个实例时只有一个能成功。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < 6 * 3600:
            print(f"[abort] 已有实例在跑 (锁 {LOCK}, {age/60:.0f} 分钟前), 本次退出")
            return False
        print("[warn] 旧锁已过期, 接管")
    tmp = LOCK.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(str(os.getpid()), encoding="utf-8")
    try:
        os.link(tmp, LOCK)
        return True
    except FileExistsError:
        print(f"[abort] 已有实例在跑 (锁 {LOCK}), 本次退出")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def split_folds(src_csv, task, folds, seed):
    """按原 csv 切成 folds 份, 生成 {task}_fold{k}.csv。

    只重排 split 标签, 不动行内容, 因此每个折的正负样本比例与原始一致。
    """
    CV_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(src_csv, encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                rows.append(p)

    tr_idx = [i for i, r in enumerate(rows) if r[0] == "train"]
    va_idx = [i for i, r in enumerate(rows) if r[0] == "val"]
    rng = random.Random(seed)
    rng.shuffle(tr_idx)
    chunks = np.array_split(np.array(tr_idx), folds)

    paths_out = []
    for k in range(folds):
        holdout = set(chunks[k].tolist())
        out = CV_DIR / f"{task}_fold{k}.csv"
        with open(out, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for i, r in enumerate(rows):
                if i in holdout:
                    f.write(",".join(["val"] + r[1:]) + "\n")
                elif r[0] == "val":
                    # 官方验证集标成 test: PatchDS 只按 train/val 取数, 会忽略它,
                    # 这样训练折既看不到官方 val, 也保证最终评估是干净的
                    f.write(",".join(["test"] + r[1:]) + "\n")
                else:
                    f.write(",".join(r) + "\n")
        n_tr = len(tr_idx) - len(holdout)
        print(f"[fold {k}] train={n_tr} 内层val={len(holdout)} 官方val={len(va_idx)}(标记为 test, 训练时不读) -> {out}")
        paths_out.append(out)
    return paths_out, Path(src_csv)


def ensure_cache(csv_path, workers=2):
    """给新 csv 建预解码缓存; 已有就跳过。没有缓存也能跑, 只是每轮慢一倍。"""
    stem = Path(csv_path).stem
    hit = list(Path("data/cache").glob(f"{stem}_*_128.json"))
    if len(hit) >= 1:
        print(f"[cache] {stem} 已存在, 跳过")
        return
    print(f"[cache] 为 {stem} 建缓存 ...")
    subprocess.run([sys.executable, "prepare_cache.py", "--csv", str(csv_path),
                    "--size", "128", "--workers", str(workers),
                    "--img-split", "train"], check=False)


def train_fold(csv_path, task, out_dir, epochs, seed, extra, workers=6):
    cmd = [sys.executable, "-u", "train.py", "--task", task, "--model", "lightpcb",
           "--csv", str(csv_path), "--epochs", str(epochs), "--bs", "128",
           "--threads", "1", "--workers", str(workers), "--device", "cuda", "--amp", "bf16",
           "--seed", str(seed), "--out", str(out_dir),
           "--img-split", "train"] + extra
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    return r.returncode == 0


@torch.no_grad()
def eval_on(csv_path, split, cfg, model, device, bs=256):
    ncls = 1 if cfg["task"] == "binary" else 9
    ds = PatchDS(csv_path, paths.DATA_ROOT, split, cfg["input_size"], ncls,
                 train=False, cache="data/cache", gray=bool(cfg.get("gray")))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    probs, ys = predict(model, loader, device)
    return probs, ys, ncls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multi9", "binary"], default="multi9")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/cv")
    ap.add_argument("--reuse", action="store_true", help="跳过训练, 直接用已有的折结果重新评估")
    ap.add_argument("--extra", default="", help="透传给 train.py 的额外参数, 如 '--input-size 128'")
    ap.add_argument("--workers", type=int, default=6, help="训练时的 DataLoader worker 数")
    args = ap.parse_args()

    if not acquire():
        return 1
    try:
        return run(args)
    finally:
        LOCK.unlink(missing_ok=True)


def run(args):
    csv_path = args.csv or f"data/patches_{args.task}.csv"
    if not Path(csv_path).exists():
        raise SystemExit(f"找不到 {csv_path}")
    out_dir = Path(args.out) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = args.extra.split() if args.extra else []

    folds_csv, official = split_folds(csv_path, args.task, args.folds, args.seed)

    if not args.reuse:
        for csvp in folds_csv:
            ensure_cache(csvp)

    results = []
    for k, csvp in enumerate(folds_csv):
        cfg_json = out_dir / f"fold{k}.json"
        pt = out_dir / f"fold{k}.pt"
        if args.reuse and cfg_json.exists():
            print(f"[reuse] fold {k}")
        else:
            # 每折单独一个输出目录, 避免 train.py 的 tag 命名互相覆盖
            ok = train_fold(csvp, args.task, out_dir / f"fold{k}_run", args.epochs,
                            args.seed + k, extra, workers=args.workers)
            if not ok:
                print(f"[fail] fold {k}")
                continue
            # train.py 写成 {tag}.json, 挪成统一名字
            cand = sorted((out_dir / f"fold{k}_run").glob("*.json"),
                          key=lambda p: p.stat().st_mtime)
            if not cand:
                print(f"[fail] fold {k} 没有产出 json")
                continue
            shutil.copy(cand[-1], cfg_json)
            src_pt = cand[-1].with_suffix(".pt")
            if src_pt.exists():
                shutil.copy(src_pt, pt)

        if not cfg_json.exists():
            continue
        cfg_dict = json.loads(cfg_json.read_text(encoding="utf-8"))
        thr = float((cfg_dict.get("tuned") or {}).get("thresh", 0.5))

        # 用主模型的结构参数重建 (train.py 的 json 里有 width/aux/se/input_size/gray)
        ncls = 1 if args.task == "binary" else 9
        kw = {"width": cfg_dict["width"], "aux": cfg_dict["aux"],
              "se": cfg_dict["se"], "in_ch": 1 if cfg_dict.get("gray") else 3}
        model = M.build_model("lightpcb", ncls, **kw).to(args.device)
        if pt.exists():
            model.load_state_dict(torch.load(pt, map_location=args.device, weights_only=True))
        model.eval()

        probs, ys, ncls = eval_on(Path(csv_path), "val", cfg_dict, model, args.device)
        m = metrics_from(probs, ys, ncls, thr)
        key = "mAP" if ncls > 1 else "acc"
        print(f"[fold {k}] 阈值 {thr:.2f} | {key} {m[key]:.4f} "
              f"macroF1 {m.get('macro_f1', float('nan')):.4f}", flush=True)
        results.append({"fold": k, "thresh": thr, **{k2: v for k2, v in m.items()
                                                    if isinstance(v, (int, float))}})

    if not results:
        raise SystemExit("没有成功的折")

    # 汇总
    keys = ["mAP", "macro_f1", "label_acc", "subset_acc"] if args.task == "multi9" \
        else ["acc", "f1", "auc", "fpr@r95"]
    print("\n=== 5 折汇总 (每折阈值取自训练折, 指标在官方 val 上算) ===")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in results if k in r and not np.isnan(r[k])])
        if not len(v):
            continue
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "n": int(len(v))}
        print(f"{k:<12} {v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}   (n={len(v)})")

    sf = out_dir / "summary.json"
    sf.write_text(json.dumps({"task": args.task, "folds": len(results),
                              "metrics": summary, "per_fold": results},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] {sf}")


if __name__ == "__main__":
    main()
