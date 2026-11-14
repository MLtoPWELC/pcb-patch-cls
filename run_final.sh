#!/bin/bash
# 补齐论文剩余三张表所需的数据。全部用最终配置: 无空洞 + shift=12。
# 表 IV(训练轮数): 60 epoch 三种子
# 表 V(分辨率):    64/128/160
# 表 II(二分类):   三种子
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY="${PY:-python}"   # Windows 虚拟环境可先 export PY=.venv/Scripts/python.exe
BASE="--model lightpcb --dilation 1,1,1 --shift 12 --bs 128 --threads 1 --workers 6 --device cuda --amp bf16"

echo "===== 表IV: 60 epoch 三种子 (无空洞) ====="
for s in 42 43 44; do
  echo "--- epoch60 seed=$s ---"
  $PY -u train.py --task multi9 $BASE --epochs 60 --seed $s --out results/e60 2>&1 | grep -E "^ep060" | cut -c1-90
done

echo "===== 表V: 分辨率 64/128/160 ====="
for sz in 64 128 160; do
  echo "--- input=$sz ---"
  if [ "$sz" = "160" ]; then EXTRA="--no-cache"; else EXTRA=""; fi
  $PY -u train.py --task multi9 $BASE --input-size $sz $EXTRA --epochs 250 --seed 42 --out results/esize 2>&1 | grep -E "^ep250" | cut -c1-90
done

echo "===== 表II: 二分类 三种子 ====="
for s in 42 43 44; do
  echo "--- binary seed=$s ---"
  $PY -u train.py --task binary $BASE --epochs 250 --seed $s --out results/ebin 2>&1 | grep -E "^ep250" | cut -c1-90
done

echo "===== 深度消融 @250 (depth=1) ====="
echo "--- depth1 无空洞 ---"
$PY -u train.py --task multi9 --model lightpcb --depth 1 --dilation 1,1,1 --bs 128 --threads 1 --workers 6 --device cuda --amp bf16 --epochs 250 --seed 42 --out results/edepth 2>&1 | grep -E "^ep250" | cut -c1-90
echo "--- depth1 空洞2/4/8 ---"
$PY -u train.py --task multi9 --model lightpcb --depth 1 --dilation 2,4,8 --bs 128 --threads 1 --workers 6 --device cuda --amp bf16 --epochs 250 --seed 42 --out results/edepth 2>&1 | grep -E "^ep250" | cut -c1-90
echo "--- depth1 w1.5 无空洞 ---"
$PY -u train.py --task multi9 --model lightpcb --depth 1 --width 1.5 --dilation 1,1,1 --bs 128 --threads 1 --workers 6 --device cuda --amp bf16 --epochs 250 --seed 42 --out results/edepth 2>&1 | grep -E "^ep250" | cut -c1-90

echo "===== 全部完成 ====="
