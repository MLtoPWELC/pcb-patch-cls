"""从 DsPCBSD+ 的 COCO 标注构建 patch 级分类数据集。

产物:
  data/patches_{task}.csv   每行一个样本: split,file,x,y,size,labels,kind,nbox
  data/meta.json            类别名、统计、构造参数

两个任务口径:
  --task multi9   9 类多标签(只取含缺陷的正样本 patch)
  --task binary   有无缺陷二分类(正样本 patch + 同图困难负样本 patch)

负样本构造原则(论文里要写清楚):
  1. 只从同一张图内采样 --> 光照、板面纹理与正样本一致, 避免"背景太干净"的虚假高分
  2. 与任一 bbox 交集为 0, 且 bbox 外扩 --margin 像素后仍无交集 --> 排除残缺缺陷
  3. 按到最近 bbox 的距离升序, 最近的作"困难负样本", 随机补"简单负样本"
  4. 负样本窗口边长从正样本的边长经验分布中抽取 --> 让正负两类经历相同的缩放统计。
     若负样本一律用原尺寸 96 裁切, 而 14.5% 的正样本是更大窗口下采样而来,
     模型会靠"清晰度"而不是缺陷形态区分两类, 这是致命的捷径。
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import paths

SPLITS = [("train", "instances_train2017.json"),
          ("val", "instances_val2017.json")]


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def intersect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def expand(box, m):
    x, y, w, h = box
    return (x - m, y - m, w + 2 * m, h + 2 * m)


def positive_window(box, crop, W, H, rng):
    """以最大缺陷框为锚点取窗口, 允许抖动但必须完整包含锚点框。"""
    x, y, w, h = box
    size = max(crop, int(max(w, h) * 1.15) + 2)
    size = min(size, min(W, H))
    cx, cy = x + w / 2.0, y + h / 2.0
    slack_x = max(0.0, (size - w) / 2.0 - 2)
    slack_y = max(0.0, (size - h) / 2.0 - 2)
    jx = rng.uniform(-1, 1) * slack_x * 0.5
    jy = rng.uniform(-1, 1) * slack_y * 0.5
    x0 = cx + jx - size / 2.0
    y0 = cy + jy - size / 2.0
    x0 = min(max(0.0, x0), W - size)
    y0 = min(max(0.0, y0), H - size)
    if x < x0:
        x0 = max(0.0, x)
    if y < y0:
        y0 = max(0.0, y)
    if x + w > x0 + size:
        x0 = max(0.0, x + w - size)
    if y + h > y0 + size:
        y0 = max(0.0, y + h - size)
    return int(round(x0)), int(round(y0)), int(size)


def negative_windows(boxes, size, W, H, n_hard, n_easy, margin, rng, mode="hard"):
    """在同图内枚举指定边长的候选窗口, 取困难/简单负样本。

    mode="hard": 按到最近 bbox 的距离升序, 最近的作困难负样本(默认, 主张的构造方式)
    mode="random": 不做距离排序, 从无缺陷候选里均匀随机抽取。
        这是"困难负样本"的对照组: 只去掉"优先选靠近缺陷的窗口"这一个变量,
        "窗口内无缺陷 + 外扩 margin 后仍无交集"的约束保留, 否则标签会出错。
    """
    stride = max(8, size // 8)
    cand = []
    for y0 in range(0, max(1, H - size + 1), stride):
        for x0 in range(0, max(1, W - size + 1), stride):
            win = (x0, y0, size, size)
            ok = True
            dmin = 1e9
            for b in boxes:
                if intersect(win, expand(b, margin)) > 0:
                    ok = False
                    break
                bx, by, bw, bh = b
                dx = max(x0 - (bx + bw), bx - (x0 + size), 0)
                dy = max(y0 - (by + bh), by - (y0 + size), 0)
                dmin = min(dmin, (dx * dx + dy * dy) ** 0.5)
            if ok:
                cand.append((dmin, x0, y0))
    if not cand:
        return []
    if mode == "random":
        rng.shuffle(cand)
        picked = cand[:n_hard + n_easy]
        return [(x, y, "neg_rand") for _, x, y in picked]
    cand.sort(key=lambda t: t[0])
    hard = cand[:n_hard]
    rest = cand[n_hard:]
    rng.shuffle(rest)
    easy = rest[:n_easy]
    return [(x, y, "neg_hard") for _, x, y in hard] + [(x, y, "neg_easy") for _, x, y in easy]


def window_labels(boxes, x0, y0, s, ncls):
    lab = [0] * ncls
    for b in boxes:
        inter = intersect((x0, y0, s, s), (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]))
        area = b["bbox"][2] * b["bbox"][3]
        if area > 0 and inter / area >= 0.6:
            lab[b["category_id"] - 1] = 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default=paths.DATA_ROOT,
                    help="Data_COCO 目录；默认取自环境变量 PCB_DATA 或 dataset.path")
    ap.add_argument("--out", default="data")
    ap.add_argument("--task", choices=["multi9", "binary"], default="multi9")
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--neg-hard", type=int, default=1)
    ap.add_argument("--neg-easy", type=int, default=1)
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--neg-mode", choices=["hard", "random"], default="hard",
                    help="hard=按离缺陷距离优先取困难负样本(默认); random=从无缺陷候选里随机取, 作对照")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (Path(args.coco_dir) / "annotations").is_dir():
        paths.check(root=args.coco_dir)
        raise SystemExit(f"\n标注目录不可用: {args.coco_dir}")

    rng = random.Random(args.seed)
    root = Path(args.coco_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 第一遍: 正样本, 同时收集窗口边长的经验分布 ----
    positives = []
    per_split = {}
    cat_names = None
    dims = {}
    for split, js in SPLITS:
        coco = load_coco(root / "annotations" / js)
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        cat_names = [c["name"] for c in cats]
        by_img = defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)
        boxes_of = {}
        for im in coco["images"]:
            anns = by_img.get(im["id"], [])
            if not anns:
                continue
            boxes_of[im["file_name"]] = anns
            dims[im["file_name"]] = (im["width"], im["height"])
            anchor = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
            x0, y0, s = positive_window(anchor["bbox"], args.crop, im["width"], im["height"], rng)
            lab = window_labels(anns, x0, y0, s, len(cat_names))
            if sum(lab) == 0:
                lab = [0] * len(cat_names)
                lab[anchor["category_id"] - 1] = 1
            positives.append((split, im["file_name"], x0, y0, s, "".join(map(str, lab)), "pos", len(anns)))
        per_split[split] = boxes_of

    size_pool = [p[4] for p in positives]

    rows = list(positives)
    stats = Counter()
    for r in rows:
        stats[r[0] + "_pos"] += 1
        stats["multilabel" if r[5].count("1") > 1 else "singlelabel"] += 1

    # ---- 第二遍: 负样本, 边长与正样本同分布 ----
    if args.task == "binary":
        for split, boxes_of in per_split.items():
            for fn, anns in boxes_of.items():
                W, H = dims[fn]
                negs, used_s = [], None
                for _ in range(6):
                    s = min(rng.choice(size_pool), min(W, H))
                    negs = negative_windows([a["bbox"] for a in anns], s, W, H,
                                            args.neg_hard, args.neg_easy, args.margin, rng,
                                            mode=args.neg_mode)
                    if negs:
                        used_s = s
                        break
                if not negs:
                    stats["img_without_neg"] += 1
                    continue
                for nx, ny, kind in negs:
                    rows.append((split, fn, nx, ny, used_s, "0" * len(cat_names), kind, 0))
                    stats[split + "_" + kind] += 1

    # 非默认负样本模式单独落一个 csv, 绝不覆盖 patches_binary.csv —— 那是主实验的数据源
    sfx = "" if (args.task != "binary" or args.neg_mode == "hard") else f"_{args.neg_mode}"
    csv_path = out_dir / f"patches_{args.task}{sfx}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("split,file,x,y,size,labels,kind,nbox\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    meta = {
        "task": args.task, "classes": cat_names, "crop": args.crop,
        "neg_hard": args.neg_hard, "neg_easy": args.neg_easy, "margin": args.margin,
        "neg_mode": args.neg_mode,
        "seed": args.seed, "num_rows": len(rows),
        "pos_size_dist": {str(k): v for k, v in sorted(Counter(size_pool).items())},
        "stats": dict(stats),
    }
    # 同理 meta 也不能覆盖: 主实验的 meta.json 记录了正式构造参数
    meta_path = out_dir / (f"meta{sfx}.json" if sfx else "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] {csv_path}  样本数 {len(rows)}")
    for k, v in sorted(stats.items()):
        print(f"     {k}: {v}")
    print("     类别:", cat_names)


if __name__ == "__main__":
    main()
