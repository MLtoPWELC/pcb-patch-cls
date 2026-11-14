"""补跑单个随机种子的全部 15 组配置。

这个环境会把后台命令重复执行一次, 两套 pipeline 各自带 8 个 DataLoader worker
时, 16 个进程同时 import torch 会崩在
  OSError: [WinError 998] 内存位置访问无效
所以这里用 os.link 做原子锁(先检查再创建的做法在同时启动时会双双通过), 并且
直接在进程内串行调用 train.py, 不再经过 run_all.py 那一层 subprocess。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import run_all as RA

LOCK = Path("results/.run_one_seed.lock")


def acquire():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < 8 * 3600:
            print(f"[abort] 已有实例在跑 (锁 {LOCK}, {age/60:.0f} 分钟前), 本次退出")
            return False
        print("[warn] 旧锁已过期, 接管")
    tmp = LOCK.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(str(os.getpid()), encoding="utf-8")
    try:
        os.link(tmp, LOCK)
        return True
    except FileExistsError:
        print(f"[abort] 已有实例在跑 (锁 {LOCK}), 本次退出")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not acquire():
        return 1
    try:
        cfgs = RA.TABLE1 + RA.TABLE2 + RA.ABLATION
        base = [x.replace("--workers", "--workers") for x in RA.BASE]
        base[base.index("--workers") + 1] = str(args.workers)
        print(f"[run] seed={args.seed} 共 {len(cfgs)} 组, workers={args.workers}", flush=True)
        t0 = time.time()
        fail = []
        for i, cfg in enumerate(cfgs, 1):
            cmd = [sys.executable, "-u", "train.py"] + cfg + base + ["--seed", str(args.seed)]
            print(f"\n[{i}/{len(cfgs)}] $ {' '.join(cmd)}", flush=True)
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                fail.append((i, cfg))
                print(f"[fail] 第 {i} 组返回 {ret.returncode}", flush=True)
        dt = (time.time() - t0) / 60
        print(f"\n[ok] seed={args.seed} 完成 {len(cfgs)-len(fail)}/{len(cfgs)} 组, 用时 {dt:.1f} 分钟", flush=True)
        for i, cfg in fail:
            print("  失败:", i, " ".join(cfg), flush=True)
        return 0 if not fail else 2
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
