"""补充消融实验: 灰度输入 / 输入分辨率 / 随机负样本对照。

三组实验各自只改变一个变量, 其余与主实验完全一致(同 csv、同增强、同超参、同 seed):
  1. gray     : 输入转单通道 vs RGB          -> 回答"AOI 单色成像下颜色是否必要"
  2. in64     : 输入 64 vs 96                -> 补齐"参数-计算量-精度"效率分析
  3. randneg  : 负样本随机采样 vs 困难优先   -> 证明同图困难负样本这一构造不是白做的

结果写 results/*.json, 汇总另存 results/extra_summary.csv, 不碰 run_seeds 的 summary.csv。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("results/.run_extra.lock")
BASE = ["--epochs", "60", "--bs", "128", "--threads", "1", "--workers", "3",
        "--device", "cuda", "--amp", "bf16"]

RUNS = [
    ("灰度-二分类", ["--task", "binary", "--model", "lightpcb", "--gray"], "binary_lightpcb_aux-none_w1.0_s42_gray"),
    ("灰度-9类",    ["--task", "multi9", "--model", "lightpcb", "--gray"], "multi9_lightpcb_aux-none_w1.0_s42_gray"),
    ("64输入-9类",  ["--task", "multi9", "--model", "lightpcb", "--input-size", "64"], "multi9_lightpcb_aux-none_w1.0_s42_in64"),
    ("64输入-二分类", ["--task", "binary", "--model", "lightpcb", "--input-size", "64"], "binary_lightpcb_aux-none_w1.0_s42_in64"),
    ("随机负样本",  ["--task", "binary", "--model", "lightpcb", "--csv", "data/patches_binary_random.csv"], "binary_lightpcb_aux-none_w1.0_s42_randneg"),
]


def acquire():
    """原子创建锁文件: 先写临时文件再 os.link, 两个实例同时启动时只有一个能成功。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < 6 * 3600:
            print(f"[abort] 已有实例在跑 (锁 {LOCK}, {age/60:.0f} 分钟前), 本次退出")
            return False
        print("[warn] 锁已过期, 覆盖")
    tmp = LOCK.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(str(os.getpid()), encoding="utf-8")
    try:
        os.link(tmp, LOCK)
        ok = True
    except FileExistsError:
        ok = False
    finally:
        tmp.unlink(missing_ok=True)
    if not ok:
        print(f"[abort] 已有实例在跑 (锁 {LOCK}), 本次退出")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if not acquire():
        return 1
    try:
        rows = []
        for name, cfg, _ in RUNS:
            for seed in args.seeds:
                cmd = [sys.executable, "-u", "train.py"] + cfg + BASE + ["--seed", str(seed)]
                print(f"\n===== {name} (seed {seed}) =====", flush=True)
                print("$", " ".join(cmd), flush=True)
                t0 = time.time()
                ret = subprocess.run(cmd)
                if ret.returncode != 0:
                    print(f"[fail] {name} seed={seed}", flush=True)
                    continue
                print(f"[ok] {name} 用时 {time.time()-t0:.0f}s", flush=True)

        # 直接从磁盘读所有 json, 不依赖 subprocess 的返回值, 避免漏记
        out = Path("results/extra_summary.csv")
        keys = None
        for p in sorted(Path("results").glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if not any(p.stem.endswith(k) for _, _, k in RUNS):
                continue
            b = r.get("best") or {}
            t = r.get("tuned") or {}
            row = {
                "tag": p.stem, "task": r["task"], "model": r["model"],
                "aux": r.get("aux"), "width": r.get("width"), "se": r.get("se"),
                "gray": r.get("gray"), "input_size": r.get("input_size"),
                "csv": Path(r.get("csv", "")).name,
                "params": r["params"], "macs_M": round(r["macs_M"], 3),
                "latency_ms": round(r["latency_ms"], 3),
            }
            for k in ("mAP", "macro_f1", "acc", "f1", "auc", "fpr@r95", "subset_acc"):
                if b.get(k) is not None:
                    row[k] = round(b[k], 4)
            for k in ("thresh", "macro_f1", "acc", "fpr@r95"):
                if t.get(k) is not None:
                    row["tuned_" + k] = round(t[k], 4) if isinstance(t[k], float) else t[k]
            rows.append(row)
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            if args.fresh or not out.exists():
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
            print(f"\n[ok] 汇总写入 {out} ({len(rows)} 行)", flush=True)
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
