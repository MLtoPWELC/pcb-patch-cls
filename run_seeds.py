"""多随机种子批量实验: 每个种子跑一遍 run_all 的三张表, 结果按 tag 累加进 summary.csv。

  python run_seeds.py                 # 默认 seed 42,43,44, 全表
  python run_seeds.py --seeds 42 43   # 指定种子
  python run_seeds.py --stage table1  # 只跑表 I

单次 15 组在本机(RTX 3060)约 35 分钟, 三个种子约 1.5~2 小时。

带一把文件锁: 误启第二个实例会直接退出, 而不是两套训练同时写同一批 json
(并发写会让 DataLoader 抢核崩在 worker 里, 结果文件也会互相覆盖)。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("results/.run_seeds.lock")


def acquire():
    """原子地抢锁。

    不能用"先 exists() 再 write_text()": 两个实例几乎同时启动时会双双通过检查,
    然后各带 8 个 DataLoader worker 同时导入 torch, 表现为 worker 里随机抛
    ModuleNotFoundError: No module named 'torch.multiprocessing.reductions'
    (16 个进程并发导入同一批 .pyc), 结果两套训练互相覆盖同一批 json。
    O_CREAT|O_EXCL 由文件系统保证只有一个能建成。
    """
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - LOCK.stat().st_mtime
                pid = LOCK.read_text(encoding="utf-8").strip()
            except OSError:
                age, pid = 0, "?"
            if attempt == 0 and age > 6 * 3600:
                print(f"[warn] 锁已过期({age/3600:.1f} 小时, pid={pid}), 覆盖", flush=True)
                LOCK.unlink(missing_ok=True)
                continue
            print(f"[abort] 已有实例在跑 (锁 pid={pid}, {age/60:.0f} 分钟前), 本次退出", flush=True)
            return False
    return False


def release():
    LOCK.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--stage", default="all", choices=["all", "table1", "table2", "ablation"])
    ap.add_argument("--fresh", action="store_true", help="第一个种子前清空 summary.csv")
    args = ap.parse_args()

    if not acquire():
        sys.exit(1)
    try:
        for i, s in enumerate(args.seeds):
            cmd = [sys.executable, "run_all.py", "--stage", args.stage, "--seed", str(s)]
            if args.fresh and i == 0:
                cmd.append("--fresh")
            print(f"\n{'='*60}\n[seed {s}] {' '.join(cmd)}\n{'='*60}", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd)
            print(f"[seed {s}] 结束 rc={r.returncode} 用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
        print("\n[all done] 汇总: results/summary.csv", flush=True)
    finally:
        release()


if __name__ == "__main__":
    main()
