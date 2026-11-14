"""按论文表格需要, 顺序跑完所有实验并汇总成 results/summary.csv。

  python run_all.py --stage table1        # 9 类主表
  python run_all.py --stage table2        # 二分类表
  python run_all.py --stage ablation      # 宽度/频域消融
  python run_all.py --stage all
  python run_all.py --stage table1 --dry  # 只打印命令
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

BASE = ["--epochs", "60", "--bs", "128", "--threads", "1", "--workers", "6",
        "--device", "cuda", "--amp", "bf16"]

TABLE1 = [  # 9 类缺陷分类: 主模型 3 个频域配置 + 3 个基线
    ["--task", "multi9", "--model", "lightpcb", "--aux", "none", "--width", "1.0"],
    ["--task", "multi9", "--model", "lightpcb", "--aux", "fft", "--width", "1.0"],
    ["--task", "multi9", "--model", "lightpcb", "--aux", "hf", "--width", "1.0"],
    ["--task", "multi9", "--model", "mobilenetv3_small"],
    ["--task", "multi9", "--model", "shufflenetv2_x0_5"],
    ["--task", "multi9", "--model", "resnet18"],
]

TABLE2 = [  # 有无缺陷二分类
    ["--task", "binary", "--model", "lightpcb", "--aux", "none", "--width", "1.0"],
    ["--task", "binary", "--model", "lightpcb", "--aux", "fft", "--width", "1.0"],
    ["--task", "binary", "--model", "mobilenetv3_small"],
    ["--task", "binary", "--model", "shufflenetv2_x0_5"],
    ["--task", "binary", "--model", "resnet18"],
]

ABLATION = [  # 宽度 → 参数量-准确率曲线; 以及 SE 开关(默认关, 此处单独打开对照)、频域旁路
    ["--task", "multi9", "--model", "lightpcb", "--width", "0.5"],
    ["--task", "multi9", "--model", "lightpcb", "--width", "0.75"],
    ["--task", "multi9", "--model", "lightpcb", "--width", "1.5"],
    ["--task", "multi9", "--model", "lightpcb", "--se"],
]

STAGES = {"table1": TABLE1, "table2": TABLE2, "ablation": ABLATION}


def expected_tag(cfg, seed):
    """与 train.py 的 tag 生成规则保持一致, 用于 --from-json 直接定位结果文件。

    必须按"带值参数 / 裸开关参数"分别处理: --se 后面没有取值, 用 zip(cfg[::2], cfg[1::2])
    会把后一个参数当成它的值, 定位到错误的文件。
    """
    task = model = None
    aux, width, se = "none", 1.0, False
    i = 0
    while i < len(cfg):
        k = cfg[i]
        if k == "--se":
            se = True
            i += 1
            continue
        v = cfg[i + 1] if i + 1 < len(cfg) and not cfg[i + 1].startswith("--") else None
        if k == "--task":
            task = v
        elif k == "--model":
            model = v
        elif k == "--aux":
            aux = v or "none"
        elif k == "--width":
            width = float(v) if v else 1.0
        i += 2 if v is not None else 1
    tag = f"{task}_{model}_aux-{aux}_w{width}_s{seed}"
    if se:
        tag += "_se"
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="table1", choices=list(STAGES) + ["all"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖每组的 epoch 数(CPU 上可先跑 20 试水)")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu, 缺省用 BASE 里的 cuda")
    ap.add_argument("--amp", default=None, choices=["auto", "off", "fp16", "bf16"])
    ap.add_argument("--fresh", action="store_true", help="清空 summary.csv 再写")
    ap.add_argument("--from-json", dest="from_json", action="store_true",
                    help="不训练, 只把已存在的 results/*.json 重建成 summary.csv")
    args = ap.parse_args()

    base = list(BASE)
    for flag, val in [("--epochs", args.epochs), ("--bs", args.bs),
                      ("--threads", args.threads), ("--workers", args.workers),
                      ("--device", args.device), ("--amp", args.amp)]:
        if val is not None:
            base[base.index(flag) + 1] = str(val)

    runs = []
    if args.stage == "all":
        for k in ["table1", "table2", "ablation"]:
            runs += STAGES[k]
    else:
        runs += STAGES[args.stage]

    rows = []
    for cfg in runs:
        cmd = [sys.executable, "train.py"] + cfg + base + ["--seed", args.seed]
        tag = expected_tag(cfg, args.seed)
        if args.from_json:
            src = Path("results") / f"{tag}.json"
            if not src.exists():
                print("[miss]", src.name, flush=True)
                continue
            print(f"[read] {src.name}", flush=True)
        else:
            print("\n$", " ".join(cmd), flush=True)
            if args.dry:
                continue
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print("[fail]", cfg, flush=True)
                continue
            # 按 tag 精确定位, 不再用 mtime 取最新: 并发或失败时会取到别的配置的结果
            src = Path("results") / f"{tag}.json"
            if not src.exists():
                print("[miss]", src.name, flush=True)
                continue
        r = json.loads(src.read_text(encoding="utf-8"))
        best = r.get("best") or {}
        rows.append({
            "tag": r["tag"], "task": r["task"], "model": r["model"], "aux": r["aux"],
            "width": r["width"], "se": r.get("se"),
            "params": r["params"], "macs_M": round(r["macs_M"], 2),
            "latency_ms": round(r["latency_ms"], 3),
            "latency_device": r.get("latency_device", ""),
            "mAP": round(best.get("mAP", float("nan")), 4) if best.get("mAP") is not None else "",
            "macro_f1": round(best.get("macro_f1", float("nan")), 4) if best.get("macro_f1") is not None else "",
            "f1": round(best.get("f1", float("nan")), 4) if best.get("f1") is not None else "",
            "subset_acc": round(best.get("subset_acc", float("nan")), 4) if best.get("subset_acc") is not None else "",
            "acc": round(best.get("acc", float("nan")), 4) if best.get("acc") is not None else "",
            "auc": round(best.get("auc", float("nan")), 4) if best.get("auc") is not None else "",
            "fpr_r95": round(best.get("fpr@r95", float("nan")), 4) if best.get("fpr@r95") is not None else "",
        })

    if rows:
        out = Path("results/summary.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        # 按 tag 去重后整体重写: 追加模式在重跑/并发时会写出重复行, 且无法保证列顺序
        merged = {}
        if out.exists() and not args.fresh:
            with open(out, newline="", encoding="utf-8") as f:
                for old in csv.DictReader(f):
                    if old.get("tag"):
                        merged[old["tag"]] = old
        for r in rows:
            merged[r["tag"]] = r
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(merged.values())
        print(f"\n[ok] 汇总写入 {out} ({len(merged)} 行)")


if __name__ == "__main__":
    main()
