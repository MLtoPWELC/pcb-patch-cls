"""环境自检 + 设备基准。

用法:
  python env_check.py                  # 打印环境、给出推荐训练命令
  python env_check.py --bench          # 加测当前设备的 fwd / fwd+bwd 耗时
  python env_check.py --sweep          # 扫精度 / channels_last / 线程数
  python env_check.py --dataloader     # 测数据管线吞吐, 判断是否会发生数据饥饿
  python env_check.py --all            # 以上全部

设计意图: 同一份代码要能在「只有 CPU 的笔记本」和「带卡的服务器」上跑,
差异只体现在本脚本检测出的这三件事上 —— 设备、可用精度、数据管线是否跟得上。
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import torch

import models as M
import paths


def hr(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # 无 CUDA 构建下部分后端访问会抛错
        return f"不可用({type(e).__name__})" if default is None else default


def env_report():
    hr("基础环境")
    print(f"Python        {sys.version.split()[0]}  ({platform.machine()})")
    print(f"操作系统      {platform.system()} {platform.release()}")
    print(f"PyTorch       {torch.__version__}")
    print(f"torchvision   {safe(lambda: __import__('torchvision').__version__)}")
    print(f"CUDA runtime  {torch.version.cuda or '无 (CPU-only 构建)'}")
    print(f"cuDNN         {safe(lambda: torch.backends.cudnn.version()) if torch.cuda.is_available() else '无'}")

    hr("设备")
    n = torch.cuda.device_count()
    if n:
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"cuda:{i}  {p.name}  {p.total_memory / 2**30:.1f} GiB  "
                  f"sm_{p.major}{p.minor}  {p.multi_processor_count} SM")
        print(f"bf16 支持     {safe(lambda: torch.cuda.is_bf16_supported())}")
    else:
        print("GPU           无可用 CUDA 设备")
    print(f"MPS           {safe(lambda: torch.backends.mps.is_available(), False)}")

    hr("CPU 与后端")
    print(f"逻辑核数      {torch.get_num_threads()} (torch) / {safe(lambda: __import__('os').cpu_count())} (os)")
    print(f"MKLDNN        {safe(lambda: torch.backends.mkldnn.is_available())}")
    print(f"AVX2          {safe(lambda: torch.backends.cpu.get_cpu_capability())}")
    print(f"TF32 (matmul) {safe(lambda: torch.backends.cuda.matmul.allow_tf32)}")
    print(f"TF32 (cudnn)  {safe(lambda: torch.backends.cudnn.allow_tf32)}")

    return recommend(n)


def cache_status(cache_dir="data/cache"):
    """统计预解码缓存的覆盖情况。

    缓存命中后 worker 只做增强、不再解 JPEG, 数据管线变轻, 应把核让给模型计算,
    因此最优 workers/threads 与无缓存时不同。推荐值必须据此调整。
    """
    d = Path(cache_dir)
    if not d.is_dir():
        return {"exists": False, "tasks": set(), "gb": 0.0, "n": 0}
    js = sorted(d.glob("patches_*_*.json"))
    tasks = set()
    for p in js:
        # patches_multi9_train_128.json -> multi9
        stem = p.stem
        for t in ("multi9", "binary"):
            if stem.startswith(f"patches_{t}_"):
                tasks.add(t)
    u8 = sorted(d.glob("patches_*_*.u8"))
    total = sum(p.stat().st_size for p in u8)
    return {"exists": bool(u8), "tasks": tasks, "gb": total / 1024**3, "n": len(u8)}


def recommend(n_gpu):
    """按环境给出可执行的配置建议。"""
    hr("推荐配置")
    on_gpu = n_gpu > 0
    bf16 = bool(on_gpu and safe(lambda: torch.cuda.is_bf16_supported(), False) is True)
    cores = safe(lambda: __import__("os").cpu_count(), 8) or 8
    cs = cache_status()

    if on_gpu:
        bs, amp, workers, chlast = 256, ("bf16" if bf16 else "fp16"), 8, True
        threads, dev = 1, "cuda"
    else:
        bs, amp, chlast = 96, "off", False
        dev = "cpu"
        if cs["exists"]:
            # 有缓存: 少 worker、多线程。实测 8 核下 6/2 为 30 s/epoch, 4/4 为 44 s。
            workers = max(1, min(2, cores // 4))
            threads = max(1, cores - workers)
        else:
            # 无缓存: worker 要解 JPEG, 需要更多进程
            workers = max(1, cores // 2)
            threads = max(1, cores - workers)

    print(f"设备        --device {dev}")
    print(f"批大小      --bs {bs}" + ("   (显存足够可上 512)" if on_gpu else "   (CPU 上再大只增加延迟, 不提升吞吐)"))
    print(f"混合精度    --amp {amp}" + ("   (bf16 不需要 GradScaler)" if amp == "bf16"
                                  else "   (fp16 由 GradScaler 自动接管)" if amp == "fp16"
                                  else "   (CPU 上 bf16 autocast 通常更慢, 除非有 AVX512-BF16)"))
    print(f"数据线程    --workers {workers}")
    print(f"内存布局    --channels-last {str(chlast).lower()}")
    print(f"编译加速    --compile {str(on_gpu).lower()}" + ("   (Windows 需 triton-windows, 不可用会自动回退)" if on_gpu else ""))
    print(f"CPU 线程    --threads {threads}" + ("   (GPU 训练时把 CPU 线程压低, 避免与 DataLoader 抢核)"
                                                if on_gpu else f"   (threads+workers={threads + workers}, 不超过 {cores} 核)"))

    if not on_gpu:
        if cs["exists"]:
            print(f"预解码缓存  命中  {cs['n']} 个文件 / {cs['gb']:.2f} GB  "
                  f"({', '.join(sorted(cs['tasks'])) or '未识别任务'})"
                  f"   -> worker 只做增强, 少 worker 多线程更快")
        else:
            print("预解码缓存  未生成   -> 每 epoch 会反复解 JPEG, 建议先跑 "
                  "python prepare_cache.py --all")

    cmd = (f"python train.py --task binary --model lightpcb --device {dev} --bs {bs} "
           f"--amp {amp} --workers {workers} --channels-last {str(chlast).lower()} "
           f"--threads {threads} --epochs 60")
    print("\n推荐命令:\n  " + cmd)
    return {"device": dev, "bs": bs, "amp": amp, "workers": workers,
            "threads": threads, "cache": cs}


def make(device, bs, size, channels_last):
    model = M.build_model("lightpcb", 9).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    x = torch.randn(bs, 3, size, size, device=device)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return model, x


def bench_one(device, bs=64, size=96, amp="off", channels_last=False, compile_=False,
              warmup=6, iters=15):
    model, x = make(device, bs, size, channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    y = torch.rand(bs, 9, device=device)
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16

    def step():
        with torch.autocast(device_type=device, dtype=dtype, enabled=amp != "off"):
            out = model(x)
            loss = lossf(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    note = ""
    if compile_:
        t0 = time.time()
        try:
            model = torch.compile(model)
            step()
            note = f" 编译 {time.time() - t0:.1f}s"
        except Exception as e:
            return None, f"compile 失败: {type(e).__name__}"

    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device, dtype=dtype, enabled=amp != "off"):
                model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    return {"ms": ms, "img_s": bs / (ms / 1000), "note": note}, None


def run_bench(rec):
    hr("设备基准 (lightpcb, 9 类, 96x96, fwd+bwd)")
    dev = rec["device"]
    cfgs = [("基准", dict(amp="off"))]
    if dev == "cuda":
        cfgs += [("fp16", dict(amp="fp16")), ("channels_last", dict(amp=rec["amp"], channels_last=True)),
                 ("amp+chlast", dict(amp=rec["amp"], channels_last=True, compile_=True))]
    else:
        cfgs += [("8 线程", dict(amp="off", bs=64))]

    for i, (name, kw) in enumerate(cfgs):
        bs = kw.pop("bs", 64)
        r, err = bench_one(dev, bs=bs, amp=kw.pop("amp", "off"),
                           channels_last=kw.pop("channels_last", False),
                           compile_=kw.pop("compile_", False))
        if err:
            print(f"{name:<14} {err}")
        else:
            print(f"{name:<14} {r['ms']:>8.2f} ms/step   {r['img_s']:>8.1f} img/s{r['note']}")
        if i == 0 and dev == "cpu":
            print("               (CPU 上 bs=64 已接近最优, 再增大只增加单步延迟)")


def run_sweep(rec):
    hr("线程数扫描 (CPU)")
    if rec["device"] != "cpu":
        print("当前不是 CPU 设备, 跳过")
        return
    orig = torch.get_num_threads()
    for t in [1, 2, 4, 8]:
        torch.set_num_threads(t)
        r, err = bench_one("cpu", bs=64, amp="off", warmup=3, iters=8)
        print(f"threads={t}  " + (err if err else f"{r['ms']:.2f} ms/step  {r['img_s']:.1f} img/s"))
    torch.set_num_threads(orig)
    print("建议: 取吞吐最高的那个; DataLoader worker 数 × 各自线程数 不要超过物理核数。")


def _bench_loader(ds, bs, w, on_gpu, n_warm=3, n_iter=8):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=w,
                    pin_memory=on_gpu, persistent_workers=w > 0,
                    prefetch_factor=(4 if w > 0 else None))
    it = iter(dl)
    for _ in range(n_warm):
        next(it)
    t0 = time.time()
    k = 0
    for _ in range(n_iter):
        next(it)
        k += 1
    ms = (time.time() - t0) / k * 1000
    return ms, bs / (ms / 1000)


def run_dataloader(rec, csv="data/patches_binary.csv", img_root=None, cache="data/cache"):
    """分别测「无缓存」与「有缓存」两条路径, 并对比不同 worker 数。

    训练默认走缓存, 只测无缓存路径会高估数据瓶颈、给出错误的 worker 建议。
    """
    hr("数据管线吞吐")
    import train as T

    if not Path(csv).exists():
        print(f"找不到 {csv}, 先跑 prepare_data.py")
        return
    root = img_root or paths.DATA_ROOT
    bs = rec["bs"]
    on_gpu = rec["device"] == "cuda"
    n_cfg = rec["cache"]
    cores = safe(lambda: __import__("os").cpu_count(), 8) or 8
    workers_set = sorted({0, rec["workers"], min(4, cores // 2)})

    def build(with_cache):
        ds = T.PatchDS(csv, root, "train", 96, 1, train=True, aug="strong",
                       cache=(cache if with_cache else None))
        # 截到前 2000 条; 缓存必须同步切片, 否则图文错位
        ds.rows, ds.labels = ds.rows[:2000], ds.labels[:2000]
        if ds.cache is not None:
            ds.cache = ds.cache[:2000]
        return ds

    for label, with_cache in (("无缓存", False), ("有缓存", n_cfg["exists"])):
        if label == "有缓存" and not with_cache:
            continue
        try:
            ds = build(with_cache)
        except Exception as e:
            print(f"{label}: 构建失败 {e}")
            continue
        print(f"\n[{label}]")
        best = (0.0, None)
        for w in workers_set:
            try:
                ms, ips = _bench_loader(ds, bs, w, on_gpu)
                print(f"  workers={w}  {ms:7.2f} ms/batch  {ips:7.1f} img/s")
                if ips > best[0]:
                    best = (ips, w)
            except Exception as e:
                print(f"  workers={w}  失败: {e}")
        if best[1] is not None:
            print(f"  -> 该路径下最优 workers={best[1]} ({best[0]:.1f} img/s)")

    print("\n判据: 数据管线 img/s 应不低于模型 img/s 的 1.5 倍, 否则会空等。")
    print("注意: 本基准只测数据管线, 不含模型计算, 因此**测不出** worker 与模型抢核的开销。")
    print("      只要 img/s 达到模型吞吐的 1.5 倍即已够用, 此时应选更少的 worker, 把核让给模型。")
    print("      最终以端到端每 epoch 时间为准（看训练日志里的 epNNN ... 秒数）。")
    print("不达标时的处理顺序: 加 --workers / 加 --prefetch / 用 --cache 预解码缓存。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dataloader", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    rec = env_report()
    if args.bench or args.all:
        run_bench(rec)
    if args.sweep or args.all:
        run_sweep(rec)
    if args.dataloader or args.all:
        run_dataloader(rec)


if __name__ == "__main__":
    main()
