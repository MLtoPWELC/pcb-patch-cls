"""从 results/summary.csv 生成论文三张表的 LaTeX 表体，直接粘进 paper/main.tex。

  python make_tables.py              # 打印三张表的表体
  python make_tables.py --write      # 同时写入 paper/tables_generated.tex
  python make_tables.py --round 3    # 小数位

设计意图：表格数字必须由脚本从 summary.csv 生成，禁止手抄。
手抄是 CHECKLIST 里点名的头号错误来源（摘要数字与表格不一致）。
"""
import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "summary.csv"

DISPLAY = {
    "lightpcb": "Ours",
    "mobilenetv3_small": "MobileNetV3-Small",
    "shufflenetv2_x0_5": r"ShuffleNetV2$\times$0.5",
    "resnet18": "ResNet18",
}


def load():
    if not SUMMARY.exists():
        raise SystemExit(f"找不到 {SUMMARY}，先跑 run_all.py")
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key, nd=4):
    v = (row.get(key) or "").strip()
    if not v or v.lower() == "nan":
        return "--"
    try:
        x = float(v)
    except ValueError:
        return v
    if math.isnan(x):
        return "--"
    return f"{x:.{nd}f}"


def params_m(row):
    """参数量: 小于 1M 用 k 表示, 否则用 M。51,770 -> 51.8k, 更容易看出量级差。"""
    try:
        p = int(float(row["params"]))
    except (KeyError, ValueError):
        return "--"
    if p < 1_000_000:
        return f"{p/1e3:.1f}k"
    return f"{p/1e6:.2f}M"


def model_label(row):
    name = row.get("model", "")
    lab = DISPLAY.get(name, name)
    if name == "lightpcb":
        aux = row.get("aux", "none")
        if aux == "fft":
            lab = "Ours + spectral gate"
        elif aux == "hf":
            lab = "Ours + HF gate"
        else:
            lab = "Ours (spatial)"
    return lab


def order_key(row):
    """让主模型在我们可控的顺序里出现：Ours 三行在基线上方，由调用方分段。"""
    return (0 if row.get("model") == "lightpcb" else 1)


def table_multi9(rows, nd):
    rs = [r for r in rows if r["task"] == "multi9" and r.get("width") == "1.00"
          and not r.get("tag", "").endswith("_nose")]
    base = sorted([r for r in rs if r["model"] != "lightpcb"],
                  key=lambda r: float(r.get("params") or 0))
    ours = [r for r in rs if r["model"] == "lightpcb"]
    aux_order = {"none": 0, "fft": 1, "hf": 2}
    ours.sort(key=lambda r: aux_order.get(r.get("aux", "none"), 9))
    out = []
    for r in base:
        out.append(f"{model_label(r)} & {fnum(r,'mAP',nd)} & {fnum(r,'macro_f1',nd)} & "
                   f"{params_m(r)} & {fnum(r,'latency_ms',2)} \\\\")
    if base:
        out.append("\\midrule")
    for r in ours:
        out.append(f"{model_label(r)} & {fnum(r,'mAP',nd)} & {fnum(r,'macro_f1',nd)} & "
                   f"{params_m(r)} & {fnum(r,'latency_ms',2)} \\\\")
    return out


def table_binary(rows, nd):
    rs = [r for r in rows if r["task"] == "binary" and r.get("width") == "1.00"]
    base = sorted([r for r in rs if r["model"] != "lightpcb"],
                  key=lambda r: float(r.get("params") or 0))
    ours = [r for r in rs if r["model"] == "lightpcb"]
    aux_order = {"none": 0, "fft": 1, "hf": 2}
    ours.sort(key=lambda r: aux_order.get(r.get("aux", "none"), 9))
    out = []
    for r in base:
        out.append(f"{model_label(r)} & {fnum(r,'acc',nd)} & {fnum(r,'f1',nd)} & "
                   f"{fnum(r,'auc',nd)} & {fnum(r,'fpr_r95',nd)} & {params_m(r)} \\\\")
    if base:
        out.append("\\midrule")
    for r in ours:
        out.append(f"{model_label(r)} & {fnum(r,'acc',nd)} & {fnum(r,'f1',nd)} & "
                   f"{fnum(r,'auc',nd)} & {fnum(r,'fpr_r95',nd)} & {params_m(r)} \\\\")
    return out


def table_ablation(rows, nd):
    rs = [r for r in rows if r["task"] == "multi9" and r["model"] == "lightpcb"]
    # 同 (width, aux, se) 只保留一行
    seen, keep = set(), []
    for r in rs:
        k = (r.get("width"), r.get("aux"), r.get("se"))
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)

    def rank(r):
        """按论文表 III 的排布: 宽度递增 -> 去 SE -> 频域旁路。"""
        w = float(r.get("width") or 1)
        se_off = str(r.get("se", "")).lower() == "false"
        aux = r.get("aux", "none")
        if se_off:
            return (w + 0.01, 1)
        return (w, 0 if aux == "none" else (2 if aux == "fft" else 3))

    keep.sort(key=rank)
    out = []
    for r in keep:
        se = r"\checkmark" if str(r.get("se", "")).lower() != "false" else "--"
        aux = {"none": "--", "fft": "FFT", "hf": "HF"}.get(r.get("aux", "none"), r.get("aux"))
        out.append(f"Ours & {float(r.get('width') or 1):.2f} & {se} & {aux} & "
                   f"{fnum(r,'mAP',nd)} & {params_m(r)} \\\\")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = load()
    # 同一 tag 可能因重跑在 summary.csv 里出现多行, 后写的覆盖先写的
    dedup = {}
    for r in rows:
        dedup[r.get("tag", "")] = r
    rows = list(dedup.values())
    # summary.csv 里 width 可能是 "1.0" 或 "1"，统一成两位
    for r in rows:
        try:
            r["width"] = f"{float(r.get('width') or 1):.2f}"
        except ValueError:
            pass
    if not rows:
        raise SystemExit("summary.csv 是空的")

    blocks = {
        "table1": (r"Nine-class (Tab.~\ref{tab:multi9})", table_multi9(rows, args.round)),
        "table2": (r"Screening (Tab.~\ref{tab:binary})", table_binary(rows, args.round)),
        "table3": (r"Ablation (Tab.~\ref{tab:ablation})", table_ablation(rows, args.round)),
    }
    lines = []
    for key, (title, body) in blocks.items():
        seg = [f"% ==== {title} ===="] + body + [""]
        print("\n".join(seg))
        lines += seg

    if args.write:
        out = ROOT / "paper" / "tables_generated.tex"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[ok] 写入 {out}")


if __name__ == "__main__":
    main()
