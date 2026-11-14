"""抽样可视化: 原图+标注框+取窗位置 | 正样本 patch | 同图困难负样本 | 同图简单负样本。

  python inspect_samples.py --n 6 --out data/preview.png
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

import paths

CAT = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
CELL = 190
PAD = 6


def load_boxes(coco_json):
    d = json.load(open(coco_json, encoding="utf-8"))
    boxes = defaultdict(list)
    for a in d["annotations"]:
        boxes[a["file_name"]] if False else None
    fn_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    for a in d["annotations"]:
        boxes[fn_by_id[a["image_id"]]].append((a["bbox"], a["category_id"]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/patches_binary.csv")
    ap.add_argument("--img-root", default=paths.DATA_ROOT)
    ap.add_argument("--anno", default=str(Path(paths.ANNO_DIR) / "instances_train2017.json"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/preview.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.img_root) / "train2017"
    boxes = load_boxes(args.anno)

    rows = [l.strip().split(",") for l in open(args.csv, encoding="utf-8").readlines()[1:]]
    pos = [r for r in rows if r[0] == "train" and r[6] == "pos"]
    neg_by_file = defaultdict(list)
    for r in rows:
        if r[0] == "train" and r[6] != "pos":
            neg_by_file[r[1]].append(r)
    cand = [r for r in pos if len(neg_by_file[r[1]]) >= 1]
    picks = rng.sample(cand, min(args.n, len(cand)))

    W = 4 * (CELL + PAD) + PAD
    H = len(picks) * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)

    for i, r in enumerate(picks):
        fn, x, y, s = r[1], int(r[2]), int(r[3]), int(r[4])
        im = Image.open(root / fn).convert("RGB")
        # 列 1: 原图 + 全部 GT 框(红) + 取窗位置(蓝)
        base = im.copy()
        d1 = ImageDraw.Draw(base)
        for (bx, by, bw, bh), cid in boxes.get(fn, []):
            d1.rectangle([bx, by, bx + bw, by + bh], outline=(255, 0, 0), width=2)
        d1.rectangle([x, y, x + s, y + s], outline=(0, 120, 255), width=2)
        base = base.resize((CELL, CELL))
        # 列 2: 正样本 patch
        posc = im.crop((x, y, x + s, y + s)).resize((CELL, CELL))
        # 列 3/4: 同图负样本(困难优先)
        negs = sorted(neg_by_file[fn], key=lambda t: 0 if t[6] == "neg_hard" else 1)[:2]
        cells = [base, posc]
        for nr in negs:
            nx, ny, ns = int(nr[2]), int(nr[3]), int(nr[4])
            nc = im.crop((nx, ny, nx + ns, ny + ns)).resize((CELL, CELL))
            nd = ImageDraw.Draw(nc)
            nd.rectangle([0, 0, CELL - 1, CELL - 1], outline=(0, 160, 0), width=3)
            cells.append(nc)
        while len(cells) < 4:
            cells.append(Image.new("RGB", (CELL, CELL), (235, 235, 235)))
        for j, c in enumerate(cells):
            sheet.paste(c, (PAD + j * (CELL + PAD), PAD + i * (CELL + PAD)))
        lab = [CAT[k] for k, v in enumerate(r[5]) if v == "1"]
        dr.text((PAD + 1 * (CELL + PAD) + 4, PAD + i * (CELL + PAD) + 4),
                "+" + "/".join(lab), fill=(255, 255, 255))
        dr.text((PAD + 2 * (CELL + PAD) + 4, PAD + i * (CELL + PAD) + 4),
                "neg", fill=(255, 255, 255))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("saved", args.out, sheet.size)
    print("列说明: 原图(红=GT框, 蓝=正样本取窗) | 正样本 patch | 同图负样本 | 同图负样本")


if __name__ == "__main__":
    main()
