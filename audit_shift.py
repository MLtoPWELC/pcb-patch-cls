"""平移鲁棒性: 把验证集的取窗在原始图上整体平移, 看性能掉多少。

为什么在原图上平移而不是对 patch 做 shift: patch 平移会引入黑边/循环填充,
那些伪影本身就会让精度掉, 测出来分不清是"位置变了"还是"图像坏了"。
在原图上挪窗口, 得到的仍是一张真实的 PCB 图, 只是缺陷在窗口里的位置变了。

跑法:
    python audit_shift.py
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import models as M
import paths


def avg_precision(y, s):
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",
                    default="results/e250/multi9_lightpcb_aux-none_w1.0_s42_dil111.json")
    ap.add_argument("--csv", default="data/patches_multi9.csv")
    ap.add_argument("--shifts", type=int, nargs="+", default=[0, 10, 20, 30, 40])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.loads(Path(args.json).read_text(encoding="utf-8"))
    ncls = 9
    kw = {"width": cfg["width"], "aux": cfg["aux"], "se": cfg["se"],
          "in_ch": 1 if cfg.get("gray") else 3,
          "dil": tuple(int(x) for x in cfg.get("dilation", "1,2,4").split(",")),
          "depth": cfg.get("depth", 2)}
    model = M.build_model("lightpcb", ncls, **kw)
    model.load_state_dict(torch.load(Path(args.json).with_suffix(".pt"),
                                     map_location=args.device, weights_only=True))
    model.eval().to(args.device)
    size = cfg["input_size"]
    gray = bool(cfg.get("gray"))

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7 and p[0] == "val":
                rows.append({"file": p[1], "x": int(p[2]), "y": int(p[3]),
                             "s": int(p[4]), "labels": p[5]})
    print(f"验证集 {len(rows)} 条, 输入 {size}x{size}, 模型 {Path(args.json).stem}")

    root = Path(paths.DATA_ROOT) / "val2017"
    if not root.is_dir():
        root = Path(paths.DATA_ROOT)
    # 预读原图, 避免每档重复解压
    imgs = {}
    for r in rows:
        if r["file"] not in imgs:
            imgs[r["file"]] = Image.open(root / r["file"]).convert("RGB")
    print(f"原图 {len(imgs)} 张已载入")

    ys = np.array([[int(c) for c in r["labels"]] for r in rows], dtype=np.float32)
    rng = random.Random(args.seed)
    # 每档用同一组方向, 保证可比
    dirs = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in rows]

    print()
    print(f"{'平移幅度':>8}{'mAP':>9}{'Δ':>8}{'subsetAcc':>11}{'labelAcc':>10}")
    base = None
    for sh in args.shifts:
        probs = []
        with torch.no_grad():
            for i, r in enumerate(rows):
                im = imgs[r["file"]]
                W, H = im.size
                x, y, s = r["x"], r["y"], r["s"]
                if sh > 0:
                    ux, uy = dirs[i]
                    # 归一化方向 × 幅度, 再夹到图像边界内保证窗口完整
                    nx = int(round(x + ux * sh))
                    ny = int(round(y + uy * sh))
                    nx = max(0, min(W - s, nx))
                    ny = max(0, min(H - s, ny))
                else:
                    nx, ny = x, y
                patch = im.crop((nx, ny, nx + s, ny + s)).resize((size, size), Image.BILINEAR)
                if gray:
                    arr = np.asarray(patch.convert("L"), dtype=np.float32)[None] / 255.0
                else:
                    arr = np.asarray(patch, dtype=np.float32).transpose(2, 0, 1) / 255.0
                t = torch.from_numpy(arr.copy())
                t = (t - 0.5) / 0.25 if gray else (t - torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)) / 0.25
                probs.append(t)
            probs = torch.stack(probs).to(args.device)
        out = []
        bs = 128
        with torch.no_grad():
            for i in range(0, len(probs), bs):
                out.append(torch.sigmoid(model(probs[i:i + bs]).float()).cpu().numpy())
        pr = np.concatenate(out)

        aps = [avg_precision(ys[:, c], pr[:, c]) for c in range(ncls)]
        m = float(np.nanmean(aps))
        pred = (pr >= 0.5).astype(np.float32)
        if base is None:
            base = m
        print(f"{sh:>8}{m:>9.4f}{m-base:>+8.4f}"
              f"{(pred==ys).all(1).mean():>11.4f}{(pred==ys).mean():>10.4f}")


if __name__ == "__main__":
    main()
