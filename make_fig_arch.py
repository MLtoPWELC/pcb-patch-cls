"""生成论文用的网络结构图(单栏, 3.5in 宽), 与 models.py 一一对应。

三段式布局:
  上  主干 6 级: input -> stem -> s1 -> s2 -> s3 -> s4, 每框写 名称 + 通道×高×宽;
      标 "×2" 的级含两个 DS 块(下采样块 + 保持块), 主干共 1+4×2 = 9 个块
  中  两个任务头: 同一个 96 维池化向量, 9 分类头出 9 个 sigmoid, 筛查头出 1 个
  下  一个 DS 块的内部: 3×3 DW+BN -> ReLU6 -> 1×1 PW+BN -> (+残差) -> ReLU6

与上一版的差别(旧版最小字号 3.9pt, 印在 3.5in 单栏里不可读):
  * 删掉图内底部那行 "one DS block, nine in the trunk; the residual applies ..." 注记;
    残差条件和块数正文 §III-D 已有, 图注也不再复述;
  * 删掉 "/2" 小字——各框下的分辨率 96→48→24→12→6 已经把 stride 2 说明白了;
  * 主干由 7 框减到 6 框(GAP 由方框改为总线旁 3 字符标注), 每框因此宽 0.07in,
    形如 16×48×48 的形状串才能在 6.0pt 下放进框内;
  * 全部文字 5.8~7.0pt(旧版 3.9~5.6pt); 图高 1.36in -> 1.58in。

注意 SE: train.py 的 --se 默认 False, 最终 checkpoint 里没有 se.* 权重, 所以块图不画 SE。

产物: paper/figs/arch.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

OUT = Path("paper/figs")

FILL = "#dce8f4"      # 主干
EDGE = "#2f5d8a"
FILL2 = "#eef2ef"     # 输入/块内部
EDGE2 = "#4e7057"
FILL3 = "#faeedb"     # 两个任务头
EDGE3 = "#96702f"
TXT = "#111111"
GREY = "#4a4a4a"

W, H = 3.5, 1.58

NAME, SHAPE, TAG = 7.0, 6.0, 5.8
HEAD, HEADSUB = 7.0, 6.4
BLK, BLKSUB, LABEL = 6.6, 6.0, 6.0

X0 = 0.03              # 左边界
GAP = 0.075            # 主干相邻框的间隙


def box(ax, x, y, w, h, lines, fill=FILL, edge=EDGE, pad=0.003):
    """圆角框 + 多行居中文字; lines 为 [(文本, 字号, 竖直位置比例), ...]。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size=0.010",
                                fc=fill, ec=edge, lw=0.65))
    for s, fs, fy in lines:
        ax.text(x + w / 2, y + h * fy, s, ha="center", va="center", fontsize=fs, color=TXT)


def arrow(ax, p, q, color=GREY, lw=0.6, ms=3.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=EDGE2, lw=0.6):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")


def fit(fig, s, fs, wmax):
    """把字号压到文本宽度不超过 wmax(英寸), 防止长字符串贴住框线。"""
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=fs)
    cur = t.get_window_extent(renderer=r).width / fig.dpi
    t.remove()
    return fs if cur <= wmax else fs * wmax / cur


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---------------- 上排: 主干 ----------------
    yA, hA = 1.16, 0.26
    stages = [("input", "3×96×96", FILL2, EDGE2, ""),
              ("stem", "16×48×48", FILL, EDGE, ""),
              ("s1", "32×24×24", FILL, EDGE, "×2"),
              ("s2", "48×12×12", FILL, EDGE, "×2"),
              ("s3", "64×6×6", FILL, EDGE, "×2"),
              ("s4", "96×6×6", FILL, EDGE, "×2")]
    n = len(stages)
    w = (W - 2 * X0 - (n - 1) * GAP) / n
    x, xs = X0, []
    for name, shape, fc, ec, tag in stages:
        fss = fit(fig, shape, SHAPE, w - 0.055)
        box(ax, x, yA, w, hA, [(name, NAME, 0.68), (shape, fss, 0.26)], fc, ec)
        if tag:  # 该级由两个块组成(下采样块 + 保持块)
            ax.text(x + w - 0.028, yA + hA - 0.042, tag, ha="right", va="center",
                    fontsize=TAG, color=GREY)
        xs.append((x, w))
        x += w + GAP

    for i in range(n - 1):
        x1 = xs[i][0] + xs[i][1]
        arrow(ax, (x1 + 0.010, yA + hA / 2), (xs[i + 1][0] - 0.010, yA + hA / 2))

    # ---------------- 中排: 两个任务头 ----------------
    yB, hB = 0.66, 0.24
    hw = 1.60
    hx = [X0, W - X0 - hw]
    heads = [("9-class head", "FC 96→9, nine sigmoids"),
             ("2-class head", "FC 96→1, one sigmoid")]
    for x0, (t, d) in zip(hx, heads):
        ax.add_patch(FancyBboxPatch((x0, yB), hw, hB,
                                    boxstyle="round,pad=0.003,rounding_size=0.010",
                                    fc=FILL3, ec=EDGE3, lw=0.65))
        ax.text(x0 + hw / 2, yB + hB * 0.68, t, ha="center", va="center",
                fontsize=HEAD, color=TXT)
        ax.text(x0 + hw / 2, yB + hB * 0.29, d, ha="center", va="center",
                fontsize=HEADSUB, color="#333333")

    # 池化向量送到两个头: 从 s4 下方引出, 一条横向总线
    gx = xs[-1][0] + xs[-1][1] / 2
    ybus = 1.02
    line(ax, [gx, gx], [yA, ybus])
    ax.text(gx + 0.042, (yA + ybus) / 2, "GAP", ha="left", va="center",
            fontsize=LABEL, color=GREY)
    line(ax, [hx[0] + hw / 2, gx], [ybus, ybus])
    for x0 in hx:
        arrow(ax, (x0 + hw / 2, ybus), (x0 + hw / 2, yB + hB), color=EDGE2)
    ax.text((hx[0] + hw / 2 + hx[1] + hw / 2) / 2, ybus,
            "96-d vector, one head per run", ha="center", va="center",
            fontsize=LABEL, color=GREY,
            bbox=dict(fc="white", ec="none", pad=0.8))

    # ---------------- 下排: 一个 DS 块 ----------------
    yC, hC = 0.20, 0.28
    seq = [("3×3 DW", "+ BN", 0.60), ("ReLU6", None, 0.46), ("1×1 PW", "+ BN", 0.60),
           ("+", None, 0.28), ("ReLU6", None, 0.46)]
    prem = 0.16
    total = sum(s[2] for s in seq) + prem * (len(seq) - 1)
    x, oxs = (W - total) / 2, []
    for a, b, bw in seq:
        if a == "+":  # 加和点画成圆, 与卷积框区分
            ax.add_patch(Circle((x + bw / 2, yC + hC / 2), bw / 2 - 0.004, fc="white", ec=EDGE2, lw=0.65))
            ax.text(x + bw / 2, yC + hC / 2 - 0.006, "+", ha="center", va="center",
                    fontsize=7.6, color=TXT)
        else:
            lines = [(a, BLK, 0.68), (b, BLKSUB, 0.27)] if b else [(a, BLK, 0.5)]
            box(ax, x, yC, bw, hC, lines, FILL2, EDGE2, pad=0.003)
        oxs.append((x, bw))
        x += bw + prem

    mid = yC + hC / 2
    for i in range(len(oxs) - 1):
        x1 = oxs[i][0] + oxs[i][1]
        arrow(ax, (x1 + 0.006, mid), (oxs[i + 1][0] - 0.006, mid), color=EDGE2, ms=3.2)
    arrow(ax, (0.06, mid), (oxs[0][0] - 0.006, mid), color=EDGE2, ms=3.2)
    arrow(ax, (oxs[-1][0] + oxs[-1][1] + 0.006, mid), (W - 0.06, mid), color=EDGE2, ms=3.2)

    # 残差: 从块入口绕过 DW/ReLU6/PW, 汇入加和点
    plus = [i for i, (a, _b, _w) in enumerate(seq) if a == "+"][0]
    sum_x = oxs[plus][0] + oxs[plus][1] / 2
    yres = 0.585
    ax.add_patch(Circle((0.13, mid), 0.007, fc=EDGE2, ec=EDGE2, lw=0))
    line(ax, [0.13, 0.13], [mid, yres])
    line(ax, [0.13, sum_x], [yres, yres])
    arrow(ax, (sum_x, yres), (sum_x, yC + hC - 0.004 + 0.008), color=EDGE2, ms=3.2)
    ax.text(sum_x + 0.055, yres - 0.034, "residual", ha="left", va="center",
            fontsize=LABEL, color=GREY)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "arch.png", dpi=600, facecolor="white")
    plt.close(fig)

    im = Image.open(OUT / "arch.png")
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        rgba = im.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        im = bg
        im.save(OUT / "arch.png")
    print("saved", OUT / "arch.png", im.size, im.mode)


if __name__ == "__main__":
    main()
