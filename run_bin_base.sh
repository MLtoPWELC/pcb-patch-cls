#!/bin/bash
# 二分类基线在 250 epoch 下重测, 用于和我们的最终配置公平对比。
# 之前的表 II 用的是 60 epoch 的基线数字, 若不改会造成"我们训得更久"的不公平比较。
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY="${PY:-python}"   # Windows 虚拟环境可先 export PY=.venv/Scripts/python.exe
for m in mobilenetv3_small shufflenetv2_x0_5 resnet18; do
  for s in 42 43 44; do
    echo "### binary $m s$s ###"
    $PY -u train.py --task binary --model $m --epochs 250 --bs 128 --threads 1 \
        --workers 6 --device cuda --amp bf16 --seed $s --out results/ebin 2>&1 \
        | grep -E "^ep250" | cut -c1-95
  done
done
echo "===== 二分类基线完成 ====="
