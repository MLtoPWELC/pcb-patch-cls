"""9 类模型的错误分析: 每类漏检/误报、以及类别之间怎么互相混淆。

跑法:
    python analyze_errors.py --model results/multi9_lightpcb_aux-none_w1.0_s42.json
    python analyze_errors.py --model ... --split train      # 看训练集(通常是过拟合参照)

产物:
    paper/figs/per_class.png     每类 TP/FP/FN 与 P/R/F1
    paper/figs/confusion.png     真实类 -> 被误判成哪一类 的热图
    paper/figs/per_class.csv     上表的数据, 方便直接写进论文
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # 只出图不开窗, 不需要 tkinter
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

import paths
from train import PatchDS
from view_samples import load_classes, load_model

C_TP, C_FP, C_FN = "#4a8f5e", "#c0504d", "#e0a33e"


def setup_font():
    """有中文字体就用中文标签, 没有就退回英文, 避免出现一堆方块。"""
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


@torch.no_grad()
def infer(model, ds, device, bs=256):
    """返回 (probs, targets), 都是 (N, ncls)。逐 batch 拼, 避免一次性占满显存。"""
    ps, ys = [], []
    n = len(ds)
    for s in range(0, n, bs):
        idx = range(s, min(s + bs, n))
        xb = torch.stack([ds[i][0] for i in idx]).to(device)
        yb = torch.stack([ds[i][1] for i in idx])
        ps.append(torch.sigmoid(model(xb).float()).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def per_class_stats(ys, pred, names):
    rows = []
    for c, nm in enumerate(names):
        y, p = ys[:, c] > 0.5, pred[:, c] > 0.5
        tp = int((y & p).sum())
        fp = int((~y & p).sum())
        fn = int((y & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec and not np.isnan(prec) else float("nan")
        rows.append({"class": nm, "support": int(y.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})
    return rows


def confusion_pairs(ys, pred, probs, names):
    """真实有的类没被判出来时, 模型改判成了什么。

    一个漏检样本通常还会多判出好几个类, 若把这些都记一遍, 行和会超过实际漏检数,
    所以只记概率最高的那个额外类。没有额外类(判成了无缺陷)时记在对角线上。
    这样行和严格等于该类别的 FN, 图好解释。
    """
    n = len(names)
    conf = np.zeros((n, n), dtype=int)
    for i in range(len(ys)):
        truth = {c for c in range(n) if ys[i, c] > 0.5}
        got = {c for c in range(n) if pred[i, c] > 0.5}
        miss = truth - got
        if not miss:
            continue
        extra = got - truth
        target = max(extra, key=lambda k: probs[i, k]) if extra else None
        for t in miss:
            conf[t, t if target is None else target] += 1
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="results/multi9_xxx.json")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/figs")
    args = ap.parse_args()

    cn = setup_font()
    L = (lambda a, b: a if cn else b)      # 中英切换
    names = load_classes()
    model, cfg, ncls = load_model(args.model, args.device)
    if ncls == 1:
        raise SystemExit("这个脚本针对 9 类多标签; 二分类请看 view_samples.py")

    csv = cfg.get("csv") or f"data/patches_{cfg['task']}.csv"
    ds = PatchDS(csv, paths.DATA_ROOT, args.split, cfg["input_size"], ncls,
                 train=False, cache="data/cache", gray=bool(cfg.get("gray")))
    thr = float((cfg.get("tuned") or {}).get("thresh", 0.5))
    print(f"[cfg] {Path(args.model).stem}  {args.split} 集 {len(ds)} 条  "
          f"阈值 {thr:.2f}  params={sum(p.numel() for p in model.parameters()):,}")

    probs, ys = infer(model, ds, args.device)
    pred = (probs >= thr).astype(np.float32)
    rows = per_class_stats(ys, pred, names)

    print(f"\n{'类别':<7}{'样本':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'精确率':>9}{'召回率':>9}{'F1':>8}")
    for r in rows:
        print(f"{r['class']:<7}{r['support']:>6}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
              f"{r['precision']:>9.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}")
    macro = {k: float(np.nanmean([r[k] for r in rows])) for k in ("precision", "recall", "f1")}
    print(f"{'宏平均':<7}{'':>6}{'':>6}{'':>6}{'':>6}"
          f"{macro['precision']:>9.3f}{macro['recall']:>9.3f}{macro['f1']:>8.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 图 1: 每类 TP/FP/FN 堆叠条 + P/R/F1
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(rows))
    tp = np.array([r["tp"] for r in rows])
    fp = np.array([r["fp"] for r in rows])
    fn = np.array([r["fn"] for r in rows])
    ax1.bar(x, tp, color=C_TP, label=L("正确检出 (TP)", "TP"))
    ax1.bar(x, fp, bottom=tp, color=C_FP, label=L("误报 (FP)", "FP"))
    ax1.bar(x, fn, bottom=tp + fp, color=C_FN, label=L("漏检 (FN)", "FN"))
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["class"] for r in rows])
    ax1.set_ylabel(L("样本数", "count"))
    ax1.set_title(L(f"各类别的检出与错误（{args.split} 集，阈值 {thr:.2f}）",
                    f"per-class TP/FP/FN ({args.split}, thr={thr:.2f})"), fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    for k, (key, col) in enumerate(zip(("precision", "recall", "f1"),
                                       ("#2f6fd0", "#8a5cb8", "#c0504d"))):
        vals = [r[key] for r in rows]
        ax2.plot(x, vals, "o-", color=col, ms=4, lw=1.6,
                 label={"precision": L("精确率", "precision"),
                        "recall": L("召回率", "recall"),
                        "f1": "F1"}[key])
    ax2.axhline(macro["f1"], color="#999", ls="--", lw=1,
                label=L(f"宏平均 F1 = {macro['f1']:.3f}", f"macro F1={macro['f1']:.3f}"))
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["class"] for r in rows])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("score")
    ax2.set_title(L("各类别的精确率 / 召回率 / F1", "per-class P / R / F1"), fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(color="#e8e8e8", lw=0.8)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    p1 = out / "per_class.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- 图 2: 真实类 -> 被误判成哪一类
    conf = confusion_pairs(ys, pred, probs, names)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    m = conf.astype(float)
    im = ax.imshow(m, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(L("模型改判成的类别", "predicted instead"))
    ax.set_ylabel(L("真实存在但未检出的类别", "missed true class"))
    ax.set_title(L("漏检时模型改判成了什么（格子内为样本数）",
                   "what the model predicts instead when it misses"), fontsize=11)
    vmax = m.max() if m.max() > 0 else 1
    for i in range(len(names)):
        for j in range(len(names)):
            if m[i, j] == 0:
                continue
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > vmax * 0.55 else "#333333")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.text(0.5, 0.012, L("对角线 = 该类完全没被任何类接住（即判成了无缺陷）",
                           "diagonal = missed with no other class predicted"),
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    p2 = out / "confusion.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---- 表格
    csvp = out / "per_class.csv"
    with open(csvp, "w", encoding="utf-8-sig", newline="") as f:
        f.write("class,support,tp,fp,fn,precision,recall,f1\n")
        for r in rows:
            f.write(f"{r['class']},{r['support']},{r['tp']},{r['fp']},{r['fn']},"
                    f"{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
        f.write(f"macro,,,,,{macro['precision']:.4f},{macro['recall']:.4f},{macro['f1']:.4f}\n")

    # ---- 文字结论: 哪些类互相混淆, 哪些主要是漏检
    pairs = [(int(conf[i, j]), names[i], names[j])
             for i in range(len(names)) for j in range(len(names)) if i != j and conf[i, j] > 0]
    pairs.sort(reverse=True)
    pure_miss = {names[i]: int(conf[i, i]) for i in range(len(names))}
    mdp = out / "error_analysis.md"
    with open(mdp, "w", encoding="utf-8") as f:
        f.write(f"# 9 类错误分析（{args.split} 集，阈值 {thr:.2f}）\n\n")
        f.write(f"模型：`{Path(args.model).stem}`　参数量 {sum(p.numel() for p in model.parameters()):,}\n\n")
        f.write("| 类别 | 样本 | TP | FP | FN | 精确率 | 召回率 | F1 |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['class']} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                    f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |\n")
        f.write(f"| **宏平均** | | | | | **{macro['precision']:.3f}** | "
                f"**{macro['recall']:.3f}** | **{macro['f1']:.3f}** |\n\n")
        f.write("## 最容易被判成别的类（非对角线）\n\n")
        for n_, a, b in pairs[:6]:
            f.write(f"- 真实有 **{a}** 但没检出，改判成 **{b}**：{n_} 次\n")
        f.write("\n## 完全漏掉（判成无缺陷）\n\n")
        for k, v in sorted(pure_miss.items(), key=lambda t: -t[1]):
            sup = next(r["support"] for r in rows if r["class"] == k)
            rate = v / sup if sup else 0
            f.write(f"- **{k}**：{v} / {sup} 个样本（{rate*100:.0f}%）完全没判出\n")
        f.write("\n## 读法\n\n"
                "- 对角线占比高 = 模型不是把它认成别的类，而是干脆没反应，"
                "通常是该类的视觉线索太弱（小、浅、与铜箔纹理接近）。\n"
                "- 非对角线出现大数 = 两类形态相近，靠单 patch 难以区分。\n")

    print(f"\n[ok] {p1}\n[ok] {p2}\n[ok] {csvp}")
    print(f"\n阈值来自训练产出的 tuned.thresh = {thr:.2f}; 换 --split train 可对比训练集")


if __name__ == "__main__":
    main()
