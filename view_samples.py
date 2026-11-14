"""交互式样本查看器: 原始数据 / 处理后的 patch / 模型的分类结果。

生成一个自包含的 HTML 文件(图片以 base64 内嵌), 用浏览器打开后逐个翻看。
不用 matplotlib 的窗口, 是因为这台机器的 Python 没带 tkinter, 装 PySide6 又太大;
HTML 方案零依赖、能缩放、也不受 PyCharm 后端影响。

用法:
    python view_samples.py                              # 原始数据 + patch, 默认验证集
    python view_samples.py --split train --only pos     # 只看训练集正样本
    python view_samples.py --task multi9 --only neg_hard
    python view_samples.py --mode predict \\
        --model results/binary_lightpcb_aux-none_w1.0_s42.json
    python view_samples.py --mode data --rows 12 --save data/preview.png   # 导出拼图, 不开浏览器

键盘: ← → 翻页, 也可以点页面下方的按钮。
"""
import argparse
import base64
import io
import json
import random
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import models as M
import paths
from train import PatchDS

POS_RGB = (40, 115, 240)
NEG_RGB = (25, 165, 100)
GT_RGB = (230, 50, 50)
PANEL = 300          # 原图与 patch 在页面里的显示边长
JPEG_Q = 82


# --------------------------------------------------------------------------- 数据

def load_classes():
    p = Path("data/meta.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["classes"]
        except Exception:
            pass
    return ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]


def load_coco_boxes(anno_path):
    d = json.loads(Path(anno_path).read_text(encoding="utf-8"))
    fn_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    out = defaultdict(list)
    for a in d["annotations"]:
        out[fn_by_id[a["image_id"]]].append((a["bbox"], a["category_id"]))
    return out


def read_rows(csv):
    rows = []
    with open(csv, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                rows.append({"split": p[0], "file": p[1], "x": int(p[2]), "y": int(p[3]),
                             "size": int(p[4]), "labels": p[5], "kind": p[6],
                             "nbox": int(p[7]) if len(p) > 7 else 0})
    return rows


def label_names(labels, classes):
    if labels.strip("0") == "":
        return ["无缺陷"]
    return [classes[i] for i, v in enumerate(labels) if v == "1" and i < len(classes)]


def b64_jpeg(im, size=PANEL, q=JPEG_Q):
    im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def deprocess(t):
    """把 (x-0.5)/0.25 归一化还原成 [0,1], 返回 PIL 图。"""
    x = (t * 0.25 + 0.5).clamp(0, 1)
    if x.dim() == 3 and x.shape[0] == 1:
        a = x[0].numpy()
        a = np.repeat(a[:, :, None], 3, axis=2)
    else:
        a = x.permute(1, 2, 0).numpy()
    return Image.fromarray((a * 255).astype(np.uint8))


# --------------------------------------------------------------------------- 模型

def load_model(model_json, device="cpu"):
    """按训练时保存的 json 重建模型并载入同名 .pt, 保证可视化用的结构与训练完全一致。

    注意: dilation / depth / rep 必须一起还原。这几个参数只改卷积的采样方式而不改权重形状,
    漏传时 load_state_dict 不会报错, 但输出会全错——表现是"每个类别都判不出来"。
    """
    cfg = json.loads(Path(model_json).read_text(encoding="utf-8"))
    ncls = 1 if cfg["task"] == "binary" else len(load_classes())
    kw = {}
    if cfg["model"] == "lightpcb":
        kw = {"width": cfg["width"], "aux": cfg["aux"], "se": cfg["se"],
              "in_ch": 1 if cfg.get("gray") else 3,
              "depth": cfg.get("depth", 2),
              "rep": bool(cfg.get("rep", False)),
              "dil": tuple(int(x) for x in cfg.get("dilation", "1,2,4").split(","))}
    model = M.build_model(cfg["model"], ncls, **kw)
    pt = Path(model_json).with_suffix(".pt")
    if not pt.exists():
        raise SystemExit(f"找不到权重 {pt}")
    model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
    model.eval().to(device)
    return model, cfg, ncls


# --------------------------------------------------------------------------- 生成条目

def make_data_items(args, rows, coco, classes):
    sel = [r for r in rows if r["split"] == args.split
           and (args.only is None or r["kind"] == args.only)]
    if not sel:
        raise SystemExit(f"没有符合条件的样本 (split={args.split}, only={args.only})")
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(sel):
        sel = rng.sample(sel, args.sample)
    sel.sort(key=lambda r: (r["kind"] != "pos", r["file"], r["y"], r["x"]))

    root = Path(args.img_root)
    sub = root / ("train2017" if args.split == "train" else "val2017")
    base = sub if sub.is_dir() else root

    items = []
    for r in sel:
        im = Image.open(base / r["file"]).convert("RGB")
        x, y, s = r["x"], r["y"], r["size"]
        color = POS_RGB if r["kind"] == "pos" else NEG_RGB
        left = im.copy()
        d = ImageDraw.Draw(left)
        for (bx, by, bw, bh), _ in coco.get(r["file"], []):
            d.rectangle([bx, by, bx + bw, by + bh], outline=GT_RGB, width=2)
        d.rectangle([x, y, x + s, y + s], outline=color, width=3)
        patch = im.crop((x, y, x + s, y + s))
        items.append({
            "title": f"[{r['split']}] {r['file']}",
            "left": b64_jpeg(left), "right": b64_jpeg(patch),
            "left_cap": "原图 · 红框=标注  彩色框=取窗",
            "right_cap": "裁剪并 resize 后的输入 patch",
            "lines": [
                ["取窗坐标", f"x={x}, y={y}, size={s}"],
                ["样本类型", "正样本" if r["kind"] == "pos" else f"负样本 ({r['kind']})"],
                ["标签", " + ".join(label_names(r["labels"], classes))],
                ["同图标注框", str(r["nbox"])],
            ],
            "probs": None,
        })
    return items


def make_predict_items(args, model, cfg, ncls, classes):
    # 老的结果 json 里没有 csv 字段, 逐级回退到默认命名, 避免 Path(None)
    train_csv = cfg.get("csv") or args.csv or f"data/patches_{cfg['task']}.csv"
    if not Path(train_csv).exists():
        raise SystemExit(f"模型训练用的数据索引不存在: {train_csv}")
    ds = PatchDS(train_csv, args.img_root, args.split, cfg["input_size"], ncls,
                 train=False, cache=None, gray=bool(cfg.get("gray")))
    rows = read_rows(train_csv)
    sel = [r for r in rows if r["split"] == args.split
           and (args.only is None or r["kind"] == args.only)]
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(sel):
        sel = rng.sample(sel, args.sample)
    sel.sort(key=lambda r: (r["kind"] != "pos", r["file"], r["y"], r["x"]))

    # PatchDS 按 csv 顺序读入, 这里用 (file,x,y) 定位行号, 避免索引错位
    key2idx = {(r[0], r[1], r[2]): i for i, r in enumerate(ds.rows)}
    thr = float((cfg.get("tuned") or {}).get("thresh", 0.5))
    names = ["缺陷"] if ncls == 1 else classes

    root = Path(args.img_root)
    sub = root / ("train2017" if args.split == "train" else "val2017")
    base = sub if sub.is_dir() else root

    items = []
    for r in sel:
        idx = key2idx.get((r["file"], r["x"], r["y"]))
        if idx is None:
            continue
        img_t, _ = ds[idx]
        with torch.no_grad():
            probs = torch.sigmoid(
                model(img_t.unsqueeze(0).to(args.device)).float()).cpu().numpy()[0]
        patch = deprocess(img_t)

        # 同时给出原图和窗口位置, 否则只看一个 patch 无法判断它在板面上的上下文
        try:
            full = Image.open(base / r["file"]).convert("RGB")
            dfull = ImageDraw.Draw(full)
            col = POS_RGB if r["kind"] == "pos" else NEG_RGB
            dfull.rectangle([r["x"], r["y"], r["x"] + r["size"], r["y"] + r["size"]],
                            outline=col, width=3)
        except Exception:
            full = patch

        if ncls == 1:
            p = float(probs[0])
            truth = "缺陷" if r["labels"].strip("0") else "正常"
            pred = "缺陷" if p >= thr else "正常"
            ok = (pred == "缺陷") == (truth == "缺陷")
            lines = [["真实", truth], ["预测", f"{pred} (p={p:.3f})"],
                     ["判定", "正确" if ok else "错误"], ["阈值", f"{thr:.2f}"]]
            mark = "✓" if ok else "✗"
        else:
            truth = " + ".join(label_names(r["labels"], classes))
            top = int(np.argmax(probs))
            picked = [classes[k] for k in range(len(classes)) if probs[k] >= thr]
            lines = [["真实", truth], ["最高分", f"{classes[top]} ({probs[top]:.3f})"],
                     ["预测为", ", ".join(picked) or "无缺陷"]]
            mark = ""
        items.append({
            "title": f"{mark} [{r['split']}] {r['file']}  ({r['kind']})",
            "left": b64_jpeg(full), "right": b64_jpeg(patch),
            "left_cap": "原图 · 彩色框=送进模型的窗口",
            "right_cap": f"模型实际输入 ({cfg['input_size']}×{cfg['input_size']}"
                         f"{', 灰度' if cfg.get('gray') else ''})",
            "lines": lines,
            "probs": {"names": names, "vals": [float(v) for v in probs], "thr": thr},
        })
    return items, thr


# --------------------------------------------------------------------------- HTML

HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root{--fg:#1a1a1a;--mut:#777;--line:#e3e3e3;--acc:#2f6fd0;--bg:#f4f5f7}
*{box-sizing:border-box}
body{margin:0;padding:20px;background:var(--bg);color:var(--fg);
     font:14px/1.6 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;background:#fff;border-radius:12px;
      padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.hd{display:flex;justify-content:space-between;align-items:baseline;
    border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:16px}
h1{font-size:15px;font-weight:600;margin:0}
.count{color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}
.pics{display:flex;gap:16px;flex-wrap:wrap}
.fig{flex:0 0 auto}
.fig img{display:block;border-radius:8px;border:1px solid var(--line);
         width:__PANEL__px;height:__PANEL__px;object-fit:cover}
.cap{color:var(--mut);font-size:12px;margin-top:6px;max-width:__PANEL__px}
.info{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}
.kv{display:grid;grid-template-columns:96px 1fr;gap:4px 12px;font-size:13px}
.kv dt{color:var(--mut)}
.kv dd{margin:0;font-weight:500}
.bars{margin-top:14px}
.bar{display:grid;grid-template-columns:74px 1fr 52px;gap:10px;align-items:center;
     margin:5px 0;font-size:12px}
.track{position:relative;height:10px;background:#eef0f3;border-radius:5px;overflow:hidden}
.fill{height:100%;background:#a9c6ea;border-radius:5px}
.fill.on{background:#2f6fd0}
.num{text-align:right;color:var(--mut);font-variant-numeric:tabular-nums}
.thr{position:absolute;top:-2px;bottom:-2px;width:1px;background:#d04a4a}
.nav{display:flex;gap:8px;align-items:center;justify-content:center;margin-top:18px}
button{border:1px solid var(--line);background:#fff;padding:7px 18px;border-radius:7px;
       cursor:pointer;font-size:13px}
button:hover{border-color:var(--acc);color:var(--acc)}
.hint{text-align:center;color:var(--mut);font-size:12px;margin-top:8px}
</style></head>
<body><div class="wrap">
<noscript><p style="color:#a00">这个页面需要启用 JavaScript 才能翻页。</p></noscript>
<div class="hd"><h1>__TITLE__</h1><div class="count" id="cnt"></div></div>
<div class="pics" id="pics"></div>
<div class="info"><dl class="kv" id="kv"></dl><div class="bars" id="bars"></div></div>
<div class="nav">
  <button id="prev">← 上一张</button><button id="next">下一张 →</button>
</div>
<div class="hint">方向键 ← → 翻页 · 数字键 1-9 快速跳转 · 共 __N__ 张</div>
</div>
<script>
const D = __DATA__;
let i = 0;
function el(t, c, txt){const e=document.createElement(t); if(c)e.className=c;
  if(txt!==undefined)e.textContent=txt; return e;}
function render(){
  const d = D[i];
  document.getElementById('cnt').textContent = (i+1) + ' / ' + D.length;
  const pics = document.getElementById('pics');
  pics.innerHTML = '';
  ['left','right'].forEach(k=>{
    if(!d[k]) return;
    const f = el('div','fig');
    const im = el('img'); im.src = d[k];
    f.appendChild(im);
    const cap = d[k+'_cap']; if(cap) f.appendChild(el('div','cap',cap));
    pics.appendChild(f);
  });
  const kv = document.getElementById('kv'); kv.innerHTML='';
  d.lines.forEach(([k,v])=>{ kv.appendChild(el('dt',null,k)); kv.appendChild(el('dd',null,v)); });
  const bars = document.getElementById('bars'); bars.innerHTML='';
  if(d.probs){
    d.probs.names.forEach((nm,idx)=>{
      const v = d.probs.vals[idx];
      const row = el('div','bar');
      row.appendChild(el('span',null,nm));
      const tr = el('div','track');
      const fl = el('div', (v>=d.probs.thr?'fill on':'fill'));
      fl.style.width = (Math.max(0.5, v*100)) + '%';
      tr.appendChild(fl);
      const th = el('div','thr'); th.style.left = (d.probs.thr*100)+'%';
      tr.appendChild(th);
      row.appendChild(tr);
      row.appendChild(el('span','num', v.toFixed(3)));
      bars.appendChild(row);
    });
    bars.appendChild(el('div','cap','竖红线 = 判决阈值 ' + d.probs.thr.toFixed(2)));
  }
  document.getElementById('prev').disabled = (i===0);
  document.getElementById('next').disabled = (i===D.length-1);
}
document.getElementById('prev').onclick = ()=>{ if(i>0){i--;render();} };
document.getElementById('next').onclick = ()=>{ if(i<D.length-1){i++;render();} };
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowRight'||e.key===' '){ if(i<D.length-1){i++;render();} e.preventDefault(); }
  else if(e.key==='ArrowLeft'){ if(i>0){i--;render();} e.preventDefault(); }
  else if(/^[1-9]$/.test(e.key)){ const t=(+e.key-1)*10; if(t<D.length){i=t;render();} }
});
render();
</script></body></html>"""


def write_html(items, out_path, title):
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    html = (HTML.replace("__TITLE__", title).replace("__DATA__", data)
                .replace("__N__", str(len(items))).replace("__PANEL__", str(PANEL)))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _font(size):
    """优先用系统里的中文字体, 找不到就退回 PIL 内置位图字体。"""
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
              "/System/Library/Fonts/PingFang.ttc"):
        try:
            from PIL import ImageFont
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    from PIL import ImageFont
    return ImageFont.load_default()


def save_grid(items, out_path, cols, legend_rows=1):
    """把若干条拼成一张总览图。左上角标签加半透明底, 否则压在铜箔纹理上看不清。"""
    n = len(items)
    rows = (n + cols - 1) // cols
    cell, gap, strip = 220, 8, 26
    W = cols * (cell + gap) + gap
    H = rows * (cell + strip + gap) + gap + 26
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = _font(13)
    for k, it in enumerate(items):
        im = Image.open(io.BytesIO(base64.b64decode(it["left"].split(",")[1])))
        r, c = divmod(k, cols)
        x0 = gap + c * (cell + gap)
        y0 = gap + r * (cell + strip + gap)
        sheet.paste(im.resize((cell, cell)), (x0, y0))
        tag = it["lines"][2][1] if len(it["lines"]) > 2 else ""
        kind = it["lines"][1][1] if len(it["lines"]) > 1 else ""
        name = f"{tag}  ·  {'正' if '正样本' in kind else '负'}"
        tw = d.textlength(name, font=f)
        d.rectangle([x0, y0, x0 + min(cell, tw + 10), y0 + 19], fill=(0, 0, 0))
        d.text((x0 + 5, y0 + 3), name, fill=(255, 255, 255), font=f)
    d.text((gap + 2, H - 20),
           "红框 = 人工标注的缺陷框    蓝框 = 取出的正样本窗口    绿框 = 取出的负样本窗口",
           fill=(70, 70, 70), font=_font(12))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return Path(out_path)


# --------------------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["data", "predict"], default="data")
    ap.add_argument("--task", choices=["multi9", "binary"], default="binary")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--model", default=None, help="predict 模式必填: results/xxx.json")
    ap.add_argument("--img-root", default=paths.DATA_ROOT)
    ap.add_argument("--anno", default=None)
    ap.add_argument("--only", default=None, help="pos / neg_hard / neg_easy / neg_rand")
    ap.add_argument("--sample", type=int, default=60, help="最多取多少条, 0=全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None,
                    help="HTML 输出路径, 默认按模式命名为 data/view_data.html 或 data/view_pred.html")
    ap.add_argument("--rows", type=int, default=0, help=">0 时改为一律导出网格 png")
    ap.add_argument("--save", default=None, help="导出网格 png 的路径(配合 --rows)")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    classes = load_classes()
    csv = args.csv or f"data/patches_{args.task}.csv"
    if not Path(csv).exists():
        raise SystemExit(f"找不到 {csv}, 先跑: python prepare_data.py --task {args.task}")

    rows = read_rows(csv)
    print(f"[data] {csv}: {len(rows)} 条 | 类型分布 {dict(Counter(r['kind'] for r in rows))}")

    if args.mode == "predict":
        if not args.model:
            raise SystemExit("predict 模式需要 --model results/xxx.json")
        model, cfg, ncls = load_model(args.model, args.device)
        items, thr = make_predict_items(args, model, cfg, ncls, classes)
        title = (f"分类结果 · {Path(args.model).stem} · "
                 f"params={M.count_params(model):,} · 阈值 {thr:.2f}")
        print(f"[model] {Path(args.model).stem}  params={M.count_params(model):,} "
              f"input={cfg['input_size']} gray={bool(cfg.get('gray'))} thr={thr:.2f}")
    else:
        anno = args.anno or str(Path(paths.ANNO_DIR) /
                                f"instances_{'train' if args.split == 'train' else 'val'}2017.json")
        coco = load_coco_boxes(anno) if Path(anno).exists() else {}
        if not coco:
            print(f"[warn] 未读到标注 {anno}, 原图上不画 GT 框")
        items = make_data_items(args, rows, coco, classes)
        title = f"数据处理前后对照 · {csv} · {args.split} 集"

    print(f"[view] 生成 {len(items)} 条")
    if args.rows > 0:
        p = save_grid(items, args.save or "data/preview.png", args.rows)
        print(f"[ok] 网格图已保存 {p}")
        return
    # 预测视图按 task 分文件, 否则 9 类跑一次就把二分类的覆盖了
    default_out = ("data/view_predict_%s.html" % args.task if args.mode == "predict"
                   else "data/view_data_%s.html" % args.split)
    out = write_html(items, args.out or default_out, title)
    print(f"[ok] 已生成 {out.resolve()}")
    print("     浏览器里用 ← → 翻页, 数字键快速跳转")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
