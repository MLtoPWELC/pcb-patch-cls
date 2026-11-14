"""检验"编号邻近"是否构成事实上的同板泄漏。

思路: 不改划分、不重训, 直接拿训练好的模型在官方 val 上推理, 把 val 样本按
"训练集里是否存在编号极近(<=N)的同前缀图"分成两组, 分别算指标。
若"有近邻"组的指标显著高于"无近邻"组, 说明同板/同批信息在起作用。
"""
import argparse
import bisect
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import models as M
import paths
from train import PatchDS, predict


def file_num(f):
    m = re.search(r"(\d+)", f)
    return int(m.group(1)) if m else -1


def file_prefix(f):
    return f.split("_")[0] if "_" in f else f[:1]


def avg_precision(y, s):
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/e250/multi9_lightpcb_aux-none_w1.0_s42_dil111.json")
    ap.add_argument("--csv", default="data/patches_multi9.csv")
    ap.add_argument("--near", type=int, default=5, help="编号差多少以内算'近邻'")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.loads(Path(args.json).read_text(encoding="utf-8"))
    ncls = 9
    if cfg["model"] == "lightpcb":
        kw = {"width": cfg["width"], "aux": cfg["aux"], "se": cfg["se"],
              "in_ch": 1 if cfg.get("gray") else 3,
              "dil": tuple(int(x) for x in cfg.get("dilation", "1,2,4").split(",")),
              "depth": cfg.get("depth", 2)}
    else:
        kw = {}
    model = M.build_model(cfg["model"], ncls, **kw)
    model.load_state_dict(torch.load(Path(args.json).with_suffix(".pt"),
                                     map_location="cuda", weights_only=True))
    model.eval().to(args.device)

    ds = PatchDS(args.csv, paths.DATA_ROOT, "val", cfg["input_size"], ncls,
                 train=False, cache="data/cache", gray=bool(cfg.get("gray")))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probs, ys = predict(model, loader, args.device)

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7 and p[0] == "val":
                rows.append((p[1], int(p[2]), int(p[3])))

    tr_nums = defaultdict(list)
    with open(args.csv, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7 and p[0] == "train":
                tr_nums[file_prefix(p[1])].append(file_num(p[1]))
    for k in tr_nums:
        tr_nums[k] = sorted(set(tr_nums[k]))

    def has_near(fn):
        pre, v = file_prefix(fn), file_num(fn)
        arr = tr_nums.get(pre, [])
        if not arr:
            return False
        i = bisect.bisect_left(arr, v)
        cand = []
        if i < len(arr):
            cand.append(abs(arr[i] - v))
        if i > 0:
            cand.append(abs(arr[i - 1] - v))
        return bool(cand) and min(cand) <= args.near

    near = np.array([has_near(fn) for fn, _, _ in rows])
    print(f"val 样本 {len(rows)}, 其中训练集有 <= {args.near} 邻近编号的: "
          f"{near.sum()} ({near.mean()*100:.1f}%)")

    for lab, mask in [("有近邻", near), ("无近邻", ~near)]:
        if mask.sum() == 0:
            continue
        aps = [avg_precision(ys[mask][:, c], probs[mask][:, c]) for c in range(ncls)]
        pred = (probs[mask] >= 0.5).astype(np.float32)
        print(f"  {lab:<6} n={mask.sum():>4}  mAP {np.nanmean(aps):.4f}  "
              f"subsetAcc {(pred==ys[mask]).all(1).mean():.4f}  "
              f"labelAcc {(pred==ys[mask]).mean():.4f}")

    # 按前缀细看
    print()
    print("按前缀拆分:")
    for pre in sorted({file_prefix(fn) for fn, _, _ in rows}):
        m = np.array([file_prefix(fn) == pre for fn, _, _ in rows])
        mn = m & near
        if mn.sum() == 0 or (~m).sum() == 0:
            continue
        aps_n = np.nanmean([avg_precision(ys[mn][:, c], probs[mn][:, c]) for c in range(ncls)])
        mf = m & ~near
        aps_f = np.nanmean([avg_precision(ys[mf][:, c], probs[mf][:, c]) for c in range(ncls)]) \
            if mf.sum() else float("nan")
        print(f"  {pre:<3} 有近邻 n={mn.sum():>4} mAP {aps_n:.4f} | "
              f"无近邻 n={mf.sum():>4} mAP {aps_f:.4f}")


if __name__ == "__main__":
    main()
