"""把 patch 预解码成 memmap 缓存, 消除每个 epoch 重复的 JPEG 解码。

为什么要做: model 的 fwd+bwd 在这台 CPU 上约 544 img/s(15 s/epoch), 但实测一个 epoch 要
51 s —— 差值全在 DataLoader 里反复解 226x226 的 JPEG。缓存后 worker 只剩增强, 不再解图。

用法:
  python prepare_cache.py --task binary --size 128
  python prepare_cache.py --task multi9 --size 128
  python prepare_cache.py --all                 # 两个任务都建

产物: data/cache/<task>_<split>_<size>.u8 + 同名 .json(记录行数, 供一致性校验)
缓存分辨率取 128 而不是 96: 训练时的"先降后升"模糊增强需要比 96 更高的源分辨率。
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import paths

CACHE = Path("data/cache")


def _load_one(job):
    """单条样本: 开图 -> 裁窗口 -> 双线性缩到 csize -> uint8 HWC。"""
    path, x, y, s, csize = job
    im = Image.open(path).convert("RGB")
    im = im.crop((x, y, x + s, y + s)).resize((csize, csize), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def read_rows(csv, split):
    rows = []
    with open(csv, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if not p or p[0] != split:
                continue
            rows.append((p[1], int(p[2]), int(p[3]), int(p[4])))
    return rows


def build(csv, img_root, split, csize, workers, chunk=1024, img_split=None):
    rows = read_rows(csv, split)
    n = len(rows)
    if n == 0:
        print(f"{csv} 无 {split} 行, 跳过")
        return
    root = Path(img_root)
    img_split = img_split or split
    sub = root / ("train2017" if img_split == "train" else "val2017")
    root = sub if sub.is_dir() else root

    CACHE.mkdir(parents=True, exist_ok=True)
    stem = Path(csv).stem
    out = CACHE / f"{stem}_{split}_{csize}.u8"
    arr = np.memmap(out, dtype=np.uint8, mode="w+", shape=(n, csize, csize, 3))

    jobs = [(str(root / fn), x, y, s, csize) for fn, x, y, s in rows]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i in range(0, n, chunk):
            part = list(ex.map(_load_one, jobs[i:i + chunk], chunksize=16))
            arr[i:i + len(part)] = np.stack(part)
            done = i + len(part)
            print(f"\r  {split} {done}/{n}  {done / (time.time() - t0):.0f} img/s", end="", flush=True)
    arr.flush()
    del arr
    mb = out.stat().st_size / 2**20
    print(f"\r  {split} {n} 条 -> {out.name}  {mb:.0f} MB  {time.time() - t0:.0f}s")
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"n": n, "size": csize, "split": split, "csv": Path(csv).name,
                   "img_root": str(root)}, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "multi9"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--img-root", default=paths.DATA_ROOT,
                    help="Data_COCO 目录；默认取自环境变量 PCB_DATA 或 dataset.path")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--img-split", default=None, choices=["train", "val"])

    args = ap.parse_args()

    if not (Path(args.img_root) / "train2017").is_dir():
        paths.check(root=args.img_root)
        raise SystemExit(f"\n图片目录不可用: {args.img_root}")

    tasks = ["binary", "multi9"] if args.all else [args.task or "binary"]
    for t in tasks:
        csv = args.csv or f"data/patches_{t}.csv"
        if not Path(csv).exists():
            print(f"找不到 {csv}, 跳过")
            continue
        print(f"=== {t} ===")
        for split in ["train", "val"]:
            build(csv, args.img_root, split, args.size, args.workers, img_split=args.img_split)


if __name__ == "__main__":
    main()
