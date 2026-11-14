"""每类 F1 图(单栏)。数据直接读 analyze_errors.py 产出的 per_class.csv, 不手抄数字。

产物: paper/figs/per_class_bar.png
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC = Path("paper/figs/per_class.csv")
OUT = Path("paper/figs/per_class_bar.png")

# 类别全名取自 DsPCBSD+ 原始数据论文 (Lv et al., Sci. Data 11:811, 2024, 第 9 类列表)。
# 数据集标注里只有缩写, 论文正文也没有给出全部名称, 所以图上直接写全名。
FULL = {
    "SH": "short",
    "SP": "spur",
    "SC": "spurious copper",
    "OP": "open",
    "MB": "mouse bite",
    "HB": "hole breakout",
    "CS": "conductor scratch",
    "CFO": "conductor foreign object",
    "BMFO": "base material foreign object",
}


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    items = [(r["class"], float(r["f1"]), int(r["support"])) for r in rows if r["class"] != "macro"]
    macro = next(float(r["f1"]) for r in rows if r["class"] == "macro")
    items.sort(key=lambda t: t[1])

    names = [f"{FULL[n]} ({sp})" for n, _, sp in items]
    vals = np.array([v for _, v, _ in items])
    y = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(3.5, 1.30))
    # 低于宏平均的用暖色, 高于的用冷色, 一眼分出难类
    colors = ["#c0504d" if v < macro else "#4a8f5e" for v in vals]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="none")
    for yy, v in zip(y, vals):
        # 白底把宏平均虚线压在文字后面, 否则 0.80--0.86 三档的数值会和虚线重叠
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", ha="left", fontsize=5.6,
                color="#222222", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.axvline(macro, color="#333333", lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(macro + 0.012, len(items) - 0.02, f"macro {macro:.2f}", fontsize=5.8,
            color="#333333", ha="left", va="bottom", zorder=7)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.6)
    ax.set_xlim(0, 1.16)
    ax.set_ylim(-0.6, len(items) + 0.26)
    ax.set_xlabel("F1 at the operating threshold", fontsize=6.4, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.0, length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(False)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, dpi=420)

    im = Image.open(OUT)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        im2 = im.convert("RGBA")
        bg.paste(im2, mask=im2.split()[-1])
        im = bg
        im.save(OUT)
    print("saved", OUT, "| macro", macro, "| order", [n for n, _, _ in items])


if __name__ == "__main__":
    main()
