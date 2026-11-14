"""生成论文用的数据处理示意图: 原图标注 + 取窗 -> 送进网络的 patch。

与训练管线完全一致的取窗逻辑直接读 patches_binary.csv, 不重新实现, 避免图和代码对不上。
产物: paper/figs/data_pipeline.pdf / .png
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

import paths

OUT = Path("paper/figs")
ABBR = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]


def load_rows():
    rows = defaultdict(list)
    with open("data/patches_binary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["file"]].append(r)
    return rows


def load_boxes(split, fname):
    """读该图的 GT 框。返回 [(x,y,w,h,cls_idx)]。"""
    jf = Path(paths.ANNO_DIR) / f"instances_{split}2017.json"
    if not hasattr(load_boxes, "_cache"):
        load_boxes._cache = {}
    if jf not in load_boxes._cache:
        load_boxes._cache[jf] = json.loads(jf.read_text(encoding="utf-8"))
    d = load_boxes._cache[jf]
    fid = next((im["id"] for im in d["images"] if im["file_name"] == fname), None)
    if fid is None:
        return []
    # COCO 类别 id -> 0..8, 按 json 里的顺序
    cats = [c["id"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    out = []
    for a in d["annotations"]:
        if a["image_id"] != fid:
            continue
        out.append((*a["bbox"], cats.index(a["category_id"])))
    return out


def pick_example(rows):
    """挑一个既有多标签正样本、又有困难/简单负样本, 且画面本身有足够纹理的图。

    只看标注信息会挑到几乎全黑的样本(负样本面板一片黑, 图没有信息量),
    所以这里再按"图像标准差 + 缺陷框尺寸"打分, 并对负样本窗口的亮度设下限。
    """
    cand = []
    for f, rs in rows.items():
        pos = [r for r in rs if r["kind"] == "pos"]
        hard = [r for r in rs if r["kind"] == "neg_hard"]
        easy = [r for r in rs if r["kind"] == "neg_easy"]
        if not (pos and hard and easy):
            continue
        if pos[0]["labels"].count("1") != 2:
            continue
        cand.append((f, pos[0], hard, easy))
    # 候选太多就抽样, 避免加载上千张图
    cand = cand[:: max(1, len(cand) // 160)][:160]

    best = None
    for f, pos, hard, easy in cand:
        split = pos["split"]
        try:
            im = Image.open(Path(paths.DATA_ROOT) / f"{split}2017" / f).convert("RGB")
        except OSError:
            continue
        g = np.asarray(im.convert("L"), dtype=np.float32)
        texture = float(g.std())
        boxes = load_boxes(split, f)
        if not boxes:
            continue
        area = max(w * h for _, _, w, h, _ in boxes)
        # 负样本窗口不能是纯黑: 要求平均灰度够高
        def bright(r):
            x, y, s = int(r["x"]), int(r["y"]), int(r["size"])
            c = g[y:y + s, x:x + s]
            return float(c.mean()) if c.size else 0.0
        bh, be = bright(hard[0]), bright(easy[0])
        if min(bh, be) < 18:
            continue
        score = texture * (area ** 0.22)
        if best is None or score > best[0]:
            best = (score, f, pos, hard[0], easy[0], texture, bh, be)
    if best is None:
        raise SystemExit("没有找到合适的示例图")
    print(f"[pick] {best[1]} texture={best[5]:.1f} neg_bright={best[6]:.1f}/{best[7]:.1f}")
    return best[1], best[2], best[3], best[4]


def crop(img, x, y, s, size=128):
    return img.crop((x, y, x + s, y + s)).resize((size, size), Image.BICUBIC)


def main():
    rows = load_rows()
    fname, pos, hard, easy = pick_example(rows)
    split = pos["split"]
    img = Image.open(Path(paths.DATA_ROOT) / f"{split}2017" / fname).convert("RGB")
    boxes = load_boxes(split, fname)

    fig = plt.figure(figsize=(3.5, 2.44))
    gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.15)

    # (a) 原图 + 标注框 + 两个候选窗
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    for (bx, by, bw, bh, ci) in boxes:
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="#e02020", lw=1.0))
    px, py, ps = int(pos["x"]), int(pos["y"]), int(pos["size"])
    ax.add_patch(Rectangle((px, py), ps, ps, fill=False, ec="#1f6fb4", lw=1.4))
    nx, ny, ns = int(hard["x"]), int(hard["y"]), int(hard["size"])
    ax.add_patch(Rectangle((nx, ny), ns, ns, fill=False, ec="#2e8b57", lw=1.4, ls=(0, (4, 2))))
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) source + boxes", fontsize=6.0, pad=2.5)
    for sp in ax.spines.values():
        sp.set_color("#999999"); sp.set_linewidth(0.6)

    # (b)-(d) 三个 patch, 2x2 的第二格起
    panels = [
        (pos, "(b) positive", "#1f6fb4"),
        (hard, "(c) hard negative", "#2e8b57"),
        (easy, "(d) easy negative", "#2e8b57"),
    ]
    slots = [(0, 1), (1, 0), (1, 1)]
    for (r, c), (row, title, col) in zip(slots, panels):
        ax = fig.add_subplot(gs[r, c])
        x, y, s = int(row["x"]), int(row["y"]), int(row["size"])
        ax.imshow(crop(img, x, y, s, size=384), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=6.0, pad=2.0, color="#222222")
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.0)
        ax.set_xlabel("")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "data_pipeline.pdf")
    fig.savefig(OUT / "data_pipeline.png", dpi=420)
    print("saved", OUT / "data_pipeline.pdf")
    print("example:", fname, "| pos", pos["labels"], f"win={pos['size']}", "| boxes", len(boxes))


if __name__ == "__main__":
    main()
