"""训练可靠性审计: 量化回答"patch 怎么取、有没有泄漏、标签对不对"。

跑法:
    python audit_data.py
    python audit_data.py --csv data/patches_binary.csv
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import paths


def load_coco(root, split):
    p = Path(root) / "annotations" / f"instances_{split}2017.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    fn_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    by_file = defaultdict(list)
    for a in d["annotations"]:
        by_file[fn_by_id[a["image_id"]]].append((a["bbox"], a["category_id"]))
    dims = {im["file_name"]: (im["width"], im["height"]) for im in d["images"]}
    return by_file, dims


def read_rows(csv):
    rows = []
    with open(csv, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                rows.append({"split": p[0], "file": p[1], "x": int(p[2]), "y": int(p[3]),
                             "size": int(p[4]), "labels": p[5], "kind": p[6]})
    return rows


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/patches_binary.csv")
    ap.add_argument("--root", default=paths.DATA_ROOT)
    args = ap.parse_args()

    rows = read_rows(args.csv)
    boxes = {}
    dims = {}
    for sp in ("train", "val"):
        try:
            b, d = load_coco(args.root, sp)
            boxes[sp] = b
            dims[sp] = d
        except FileNotFoundError:
            print(f"[warn] 缺 {sp} 标注")
    print(f"[data] {args.csv}: {len(rows)} 行, 类型 {dict(Counter(r['kind'] for r in rows))}\n")

    # ---------------- 1. 正样本的中心偏移 ----------------
    print("=" * 72)
    print("1. 正样本里缺陷是否被摆在图像中心")
    print("=" * 72)
    offs = []
    for r in rows:
        if r["kind"] != "pos":
            continue
        anns = boxes.get(r["split"], {}).get(r["file"], [])
        if not anns:
            continue
        # 锚点 = 面积最大的框(与 prepare_data 一致)
        ax, ay, aw, ah = max((b for b, _ in anns), key=lambda b: b[2] * b[3])
        cx, cy = ax + aw / 2, ay + ah / 2
        wcx, wcy = r["x"] + r["size"] / 2, r["y"] + r["size"] / 2
        offs.append((abs(cx - wcx) / r["size"], abs(cy - wcy) / r["size"]))
    if offs:
        o = np.array(offs)
        d = np.sqrt((o ** 2).sum(1))
        print(f"  样本数 {len(d)}")
        for q in (50, 75, 90, 95, 99):
            print(f"  {q:>2d}% 分位: 归一化距离 {np.percentile(d, q):.4f}"
                  f"  ({np.percentile(d, q)*96:.1f} px @96)")
        print(f"  平均 {d.mean():.4f}, 最大 {d.max():.4f}")
        print(f"  完全居中(<1% )的比例: {(d < 0.01).mean()*100:.1f}%")
        print(f"  距离 <10% 的比例:     {(d < 0.10).mean()*100:.1f}%")
        print(f"  若为均匀随机, 期望平均距离约 0.38")
        print("  -> 平均值越小, 说明缺陷越被钉在窗口中心")

    # ---------------- 2. 训练/验证是否共享图片 ----------------
    print()
    print("=" * 72)
    print("2. 训练集与验证集是否共享同一张图")
    print("=" * 72)
    tr_files = {r["file"] for r in rows if r["split"] == "train"}
    va_files = {r["file"] for r in rows if r["split"] == "val"}
    inter_files = tr_files & va_files
    print(f"  train 涉及图片 {len(tr_files)}, val 涉及图片 {len(va_files)}")
    print(f"  交集: {len(inter_files)} 张" + (f"  {list(inter_files)[:5]}" if inter_files else "  (无重叠, 正常)"))
    # 官方标注本身的图片交集
    off_tr = set(boxes.get("train", {}).keys())
    off_va = set(boxes.get("val", {}).keys())
    print(f"  官方 COCO 划分本身交集: {len(off_tr & off_va)} 张")
    # 同一张图是否贡献了多个 patch
    per_file = Counter((r["split"], r["file"]) for r in rows)
    multi = [k for k, v in per_file.items() if v > 1]
    print(f"  同图产生多个 patch 的图片数: {len(multi)}"
          f"  (每个 patch 都来自同 split, 因此不构成跨集泄漏)")
    if multi:
        v = Counter(per_file[k] for k in multi)
        print(f"    每图 patch 数分布: {dict(sorted(v.items()))}")

    # ---------------- 3. 负样本是否真的无缺陷 ----------------
    print()
    print("=" * 72)
    print("3. 负样本窗口内是否真的没有缺陷")
    print("=" * 72)
    margin = 6
    bad = 0
    n_neg = 0
    min_area_ratio = 1.0
    for r in rows:
        if r["kind"] == "pos":
            continue
        n_neg += 1
        win = (r["x"], r["y"], r["size"], r["size"])
        for (bx, by, bw, bh), _ in boxes.get(r["split"], {}).get(r["file"], []):
            # 外扩 margin 后仍不应相交
            e = (bx - margin, by - margin, bw + 2 * margin, bh + 2 * margin)
            if inter(win, e) > 0:
                bad += 1
                min_area_ratio = min(min_area_ratio, inter(win, (bx, by, bw, bh)) /
                                     max(1, bw * bh))
                break
    print(f"  负样本 {n_neg} 个, 与'外扩 {margin}px 的标注框'相交的: {bad}"
          f" ({bad/max(1,n_neg)*100:.2f}%)")
    print("  -> 这项应为 0; 非 0 说明 CSV 与标注版本不一致")

    # ---------------- 4. 标签是否与窗口内容一致 ----------------
    print()
    print("=" * 72)
    print("4. 正样本标签是否与窗口内的框一致")
    print("=" * 72)
    mismatch = 0
    n_pos = 0
    nlab = Counter()
    for r in rows:
        if r["kind"] != "pos":
            continue
        n_pos += 1
        anns = boxes.get(r["split"], {}).get(r["file"], [])
        win = (r["x"], r["y"], r["size"], r["size"])
        expect = set()
        for (bx, by, bw, bh), cid in anns:
            a = bw * bh
            if a > 0 and inter(win, (bx, by, bw, bh)) / a >= 0.6:
                expect.add(cid - 1)
        got = {i for i, c in enumerate(r["labels"]) if c == "1"}
        nlab[len(got)] += 1
        if not expect:
            # 锚点兜底: prepare_data 在窗口没盖住 60% 时强制给锚点类
            continue
        if not got & expect:
            mismatch += 1
    print(f"  正样本 {n_pos} 个, 标签与'覆盖≥60%的框'完全无交集的: {mismatch}"
          f" ({mismatch/max(1,n_pos)*100:.2f}%)")
    print(f"  每个 patch 的标签数分布: {dict(sorted(nlab.items()))}")
    tot = sum(nlab.values())
    print(f"  平均标签数 {sum(k*v for k,v in nlab.items())/max(1,tot):.3f}")

    # ---------------- 5. 泄漏的另一种形式 ----------------
    print()
    print("=" * 72)
    print("5. 其他潜在泄漏点")
    print("=" * 72)
    tr_box_files = {r["file"] for r in rows if r["split"] == "train"}
    va_box_files = {r["file"] for r in rows if r["split"] == "val"}
    # 文件名前缀(可能是板号/批次)是否跨集
    def prefix(f):
        return f.split("_")[0] if "_" in f else f[:1]
    pre_tr = Counter(prefix(f) for f in tr_box_files)
    pre_va = Counter(prefix(f) for f in va_box_files)
    both = set(pre_tr) & set(pre_va)
    print(f"  文件名前缀跨集的: {sorted(both)}")
    for p in sorted(both):
        print(f"    {p}: train {pre_tr[p]} 张 / val {pre_va[p]} 张")
    print("  注: 前缀若非板号, 则不能用来判断同板泄漏; "
          "DsPCBSD+ 官方划分是按板做的, 我们沿用未改动")


if __name__ == "__main__":
    main()
