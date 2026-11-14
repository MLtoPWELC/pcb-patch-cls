"""生成论文引言用的 "六类典型缺陷形态" 图。

样本全部选自 DsPCBSD+ 的验证集, 优先挑"单标签 + 缺陷框占窗口比例大"的样本,
保证每一格里的缺陷形态肉眼可辨。原图只有 226x226, 用 LANCZOS 上采样到 320 再画。

产物: paper/figs/defect_types.png
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import paths

OUT = Path("paper/figs")
ABBR = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
FULL = {"SH": "short", "OP": "open", "SP": "spur", "MB": "mouse bite",
        "CS": "scratch", "CFO": "foreign object"}
# 图上用的标签取 DsPCBSD+ 原始数据论文的类别全名 (Lv et al., Sci. Data 11:811, 2024);
# 数据集标注里只有缩写, 读者无法从图注还原, 因此这里直接写全名, 长名换行。
LABEL = {"SH": "short", "OP": "open", "SP": "spur", "MB": "mouse bite",
         "CS": "conductor\nscratch", "CFO": "conductor\nforeign object"}
SHOW = ["SH", "OP", "SP", "MB", "CS", "CFO"]          # 与引言正文列出的六类一致
# 人工从候选拼版中挑的样本(每类形态最清楚的一张), 固定文件名保证可复现
FINAL = {"SH": "S_11833725.jpg", "OP": "S_11462696.jpg", "SP": "S_10098502.jpg",
         "MB": "S_11463550.jpg", "CS": "S_11105550.jpg", "CFO": "0067994.jpg"}


def load_boxes(split, fname):
    jf = Path(paths.ANNO_DIR) / f"instances_{split}2017.json"
    if not hasattr(load_boxes, "_c"):
        load_boxes._c = {}
    if jf not in load_boxes._c:
        load_boxes._c[jf] = json.loads(jf.read_text(encoding="utf-8"))
    d = load_boxes._c[jf]
    fid = next((im["id"] for im in d["images"] if im["file_name"] == fname), None)
    if fid is None:
        return []
    cats = [c["id"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    out = []
    for a in d["annotations"]:
        if a["image_id"] == fid:
            out.append((a["bbox"], cats.index(a["category_id"])))
    return out


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def pick(rows):
    """按 FINAL 取指定样本; 找不到时回退到"缺陷框占比最大"的单标签样本。"""
    want = {v: k for k, v in FINAL.items()}
    best = {c: None for c in SHOW}
    for r in rows:
        if r["split"] != "val" or r["kind"] != "pos":
            continue
        lab = r["labels"]
        if lab.count("1") != 1:
            continue
        ci = lab.index("1")
        name = ABBR[ci]
        if name not in best:
            continue
        if r["file"] not in want or want[r["file"]] != name:
            continue
        x, y, s = int(r["x"]), int(r["y"]), int(r["size"])
        best[name] = (1.0, r, (x, y, s, s))
    for name in SHOW:
        if best[name] is None:
            raise SystemExit(f"{name}: 未找到指定样本 {FINAL[name]}")
    return best


def crop(fname, split, win, out_size):
    root = Path(paths.DATA_ROOT) / ("train2017" if split == "train" else "val2017")
    im = Image.open(root / fname).convert("RGB")
    x, y, w, h = win
    box = (max(0, x), max(0, y), min(im.width, x + w), min(im.height, y + h))
    return im.crop(box).resize((out_size, out_size), Image.LANCZOS)


def main():
    with open("data/patches_multi9.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = pick(rows)

    n = len(SHOW)
    fig, axes = plt.subplots(1, n, figsize=(3.5, 0.80))
    for ax, name in zip(axes.ravel(), SHOW):
        item = best[name]
        if item is None:
            ax.axis("off")
            continue
        _, r, win = item
        ax.imshow(crop(r["file"], r["split"], win, 320), interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#777777")
            sp.set_linewidth(0.5)
        ax.set_title(LABEL.get(name, name), fontsize=5.6, pad=1.6, color="#111111",
                     linespacing=0.95)
        print(f"{name:<4} {r['file']:<20} win={win}")
    fig.subplots_adjust(left=0.010, right=0.988, top=0.74, bottom=0.02, wspace=0.07)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "defect_types.png", dpi=400)
    plt.close(fig)

    im = Image.open(OUT / "defect_types.png")
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        rgba = im.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        im = bg
        im.save(OUT / "defect_types.png")
    print("saved", OUT / "defect_types.png", im.size, im.mode)


if __name__ == "__main__":
    main()
