"""数据根目录解析。

换机器时不必改任何脚本。解析顺序：

1. 环境变量 ``PCB_DATA``：指向 ``Data_COCO`` 目录（含 ``train2017`` /
   ``val2017`` / ``annotations``）。
2. 项目根目录下的 ``dataset.path`` 文件：首行写路径，``#`` 开头为注释。
两项都没有时退回空串，由调用方报出可操作的路径错误。

用法::

    import paths
    img_root = args.img_root or paths.DATA_ROOT
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_CONF_FILE = PROJECT_ROOT / "dataset.path"


def _read_conf() -> str:
    if not _CONF_FILE.is_file():
        return ""
    for line in _CONF_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def resolve() -> str:
    """返回 Data_COCO 目录的绝对路径。"""
    for cand in (os.environ.get("PCB_DATA", "").strip(), _read_conf()):
        if cand and Path(cand).is_dir():
            return cand
    # 都不可用时返回配置值（可能为空），让调用方报出有意义的路径错误
    return os.environ.get("PCB_DATA", "").strip() or _read_conf()


DATA_ROOT = resolve()
ANNO_DIR = str(Path(DATA_ROOT) / "annotations")


def check(verbose: bool = True, root: str | None = None) -> bool:
    """校验目录结构，返回是否可用。``root`` 用于校验显式传入的路径。"""
    data_root = root or DATA_ROOT
    need = [Path(data_root) / "annotations",
            Path(data_root) / "train2017",
            Path(data_root) / "val2017"]
    ok = all(p.is_dir() for p in need)
    if verbose:
        if root:
            src = "命令行参数"
        elif os.environ.get("PCB_DATA", "").strip() == DATA_ROOT:
            src = "环境变量 PCB_DATA"
        elif _read_conf() == DATA_ROOT:
            src = _CONF_FILE.name
        else:
            src = "内置默认值"
        print(f"数据根目录: {data_root}  (来源: {src})")
        for p in need:
            print(f"  {'OK  ' if p.is_dir() else 'MISS'} {p}")
        if not ok:
            print("\n修正方式（任选其一）：")
            print("  1) 在本文件同目录新建 dataset.path，首行写入 Data_COCO 的绝对路径；")
            print("  2) 设置环境变量 PCB_DATA 指向同一目录；")
            print("  3) 命令行显式传 --img-root / --coco-dir。")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
