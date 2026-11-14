# PCB-Patch-Cls: a 34k-parameter patch classifier for PCB defect re-inspection

Reference implementation for patch-level defect classification and false-call screening on
[DsPCBSD+](https://doi.org/10.1038/s41597-024-03656-8) (Scientific Data 11, 811, 2024).

The classifier is a depthwise-separable CNN with dilated blocks and a 16× downsampling factor.
It has **34,250 parameters** (multi-label, 9 classes) and **33,474** (binary screening), runs at
**~1.3 ms per image on a single CPU thread** (batch 1, no quantization), and reaches **0.909 mAP**
on the 9-class task averaged over three seeds.

Two task definitions are supported:

| task | samples | label | what it measures |
|---|---|---|---|
| `multi9` | one patch per annotated defect box | 9-bit multi-hot | defect classification |
| `binary` | the same patches plus in-image negatives | 0/1 | screening / false-call rate at the AOI stage |

## Repository layout

```
models.py            LightPCB and the baseline wrappers (torchvision, re-headed)
train.py             training / evaluation loop, metrics, latency & MACs probes
prepare_data.py      builds the patch index (data/patches_*.csv + meta.json)
prepare_cache.py     optional memmap cache of decoded patches
paths.py             resolves the dataset root (PCB_DATA env var or dataset.path)
run_all.py           the paper's table configurations
run_seeds.py         multi-seed driver; run_one_seed.py for a single seed
cv_eval.py           5-fold CV with thresholds calibrated inside the training folds
analyze_errors.py    per-class TP/FP/FN and confusion matrices
audit_data.py        dataset statistics and split sanity checks
audit_leak.py        near-duplicate / index-adjacency leak checks
audit_shift.py       translation-robustness probe used in the ablation table
view_samples.py      self-contained HTML viewer for patches and predictions
make_tables.py       LaTeX table bodies generated from results/ (no hand-copied numbers)
make_fig_*.py        paper figures (written to paper/figs/, created on demand)
env_check.py         device / precision / throughput report
data/patches_*.csv   committed patch index (rebuildable, kept for reference)
results/             one JSON per run, plus summary.csv
```

## Installation

Python 3.10+ (tested on 3.13) with PyTorch.

```bash
pip install -r requirements.txt
# CUDA build: follow https://pytorch.org for your driver, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python env_check.py --bench --dataloader     # prints device, precision, throughput
```

Nothing else is required for training or evaluation. The figure scripts additionally need
`matplotlib`.

## Dataset

Download DsPCBSD+ from the Scientific Data article (DOI
[10.1038/s41597-024-03656-8](https://doi.org/10.1038/s41597-024-03656-8)) and point the code at
the `Data_COCO` directory, which must contain `annotations/`, `train2017/` and `val2017/`:

```bash
export PCB_DATA=/path/to/DsPCBSD+/Data_COCO      # or:
echo /path/to/DsPCBSD+/Data_COCO > dataset.path
python paths.py                                   # verifies the three subdirectories
```

The dataset itself is not redistributed here. `data/patches_*.csv` only holds relative file names,
window coordinates and labels, so it can be inspected without the images.

## Patch construction

A window is anchored on the largest defect box of an image, with edge length
`max(96, 1.15 × max(w, h) + 2)`; windows that would cross the image border are translated inward
rather than padded, then resized to 96×96. 85.5% of windows are exactly 96 px, the rest are
larger and downsampled.

Negative samples are drawn from the *same* image, at most two per image, with zero overlap and a
6 px margin around every box, split into a hard (nearest-to-box) and an easy (random) variant.
36% of images contain no legal negative window and are skipped. Negatives inherit their image's
split, so there is no cross-split leakage.

## Reproducing the results

```bash
python prepare_data.py --task multi9          # builds data/patches_multi9.csv
python prepare_data.py --task binary
python prepare_cache.py --all                 # optional, ~1.6 GB, makes epochs ~2x faster

python train.py --task binary --model lightpcb --epochs 1 --max-steps 30 --smoke   # sanity check

python run_all.py --stage table1              # 9-class results
python run_all.py --stage table2              # binary screening
python run_all.py --stage ablation            # width / SE / spectral-branch ablation
python run_seeds.py                           # seeds 42/43/44 (mean ± std)
python make_tables.py --write                 # LaTeX tables from results/summary.csv
```

A single run is short: ~1.3 s/epoch on the 9-class task and ~2.6 s/epoch on the binary task with
an RTX 3060, batch 128, bf16. All numbers in `results/*.json` were produced by these scripts.

Metrics: mAP and macro-F1 for `multi9`; accuracy, F1, AUC and **FPR at 95% recall** for `binary`.
The multi-label threshold is tuned on the validation split and stored in the `tuned` field of each
JSON, so both `best@0.5` and `tuned` remain auditable.

## Models and baselines

| model | params (this task) | MACs | CPU latency |
|---|---|---|---|
| LightPCB (ours, 9-class) | 34,250 | 2.86 M | ~1.34 ms |
| LightPCB (ours, binary) | 33,474 | 2.86 M | ~1.30 ms |
| ShuffleNetV2 ×0.5 | 351,017 | 7.26 M | ~4.6 ms |
| MobileNetV3-Small | 1,527,081 | 11.06 M | ~3.1 ms |
| ResNet18 | 11,181,129 | 333.1 M | ~11.6 ms |

Baselines are re-headed for the task rather than quoted from ImageNet. Latency is measured on CPU,
batch 1, single thread, averaged over 50 runs; MACs and parameters are counted on the actual model.

`--se` adds a squeeze-and-excitation block (51,770 parameters) and `--aux fft|hf` adds a spectral
gating branch; both are ablation options, not part of the default model.

## Audits

The scripts under `audit_*.py` back the limitations discussed in the paper and can be re-run
independently:

- `audit_data.py` — class counts, window-size distribution, negative-sample statistics.
- `audit_leak.py` — duplicate and near-duplicate checks, file-index adjacency between splits.
- `audit_shift.py` — mAP of a trained model when the sampling window is displaced by 20/40 px.
- `cv_eval.py` — 5-fold CV where the decision threshold is calibrated on inner folds only, so the
  validation split is never used to pick it.

## License

MIT. See [LICENSE](LICENSE).
