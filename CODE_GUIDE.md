# CODE_GUIDE – what every file does, in simple words

## What changed in this version

| File | Change | Why |
|---|---|---|
| `robustwaste/compare.py` | **NEW** | Builds all paper tables (clean, robustness, reject, unknown waste, ablation A–D) and 2 figures in one command |
| `robustwaste/train.py` | added `--no-pretrained` | Train without downloading ImageNet weights (offline / quick tests) |
| `robustwaste/infer.py` | added `--condition`, `--severity`; event now has `image_condition` | The IoT message in the proposed work contains the image condition |
| `robustwaste/ood.py` | now also saves `ood_summary.json` | Table 4 needs mean confidence and reject rate |
| `run_experiments.sh/.bat` | last line runs `compare` | Whole study in one script |
| `README.md` | new compare section | Usage |

Everything else is the original code, unchanged.

---

## The pipeline in 8 steps

1. `scripts/download_realwaste.py` – downloads the RealWaste zip from UCI and unzips it.
2. `scripts/prepare_realwaste.py` – merges 9 classes into 6 (Cardboard+Paper, Food Organics+Vegetation, …), splits 70/15/15 into `train/val/test`, and puts **Textile Trash** in `ood_textile/` (never trained on).
3. `robustwaste/train.py` – fine-tunes a pretrained model. `--degradation-aware` adds random low-light / blur / noise / occlusion to training images.
4. `robustwaste/evaluate.py` – clean test accuracy, macro P/R/F1, confusion matrix.
5. `robustwaste/threshold.py` – picks the confidence threshold on the **validation** set.
6. `robustwaste/robustness.py` – tests the model on degraded copies of the test set.
7. `robustwaste/ood.py` – feeds unseen textile images and counts how many are rejected.
8. `robustwaste/compare.py` – does steps 4–7 for all four models and writes the tables.

---

## `compare.py` explained block by block

```python
CONDITIONS = ["normal", "low_light", "blur", "noise", "occlusion"]
DEFAULT_RUNS = {"ResNet50": "resnet50_baseline", ...}
```
The 5 test conditions, and which folder in `outputs/` belongs to which model name in the tables.

```python
def get_predictions(model, loader, device):
```
Runs the model over a dataset. For each image: `softmax` turns raw scores into probabilities,
`max` gives the **confidence** (highest probability) and the **predicted class** (its position).
Returns three arrays: true labels, predicted labels, confidences.

```python
def pick_threshold(y, pred, conf, target):
```
Tries thresholds 0.50, 0.51, … 0.95. For each one: images with `conf >= t` are ACCEPTED.
It keeps the threshold that accepts the most images while accepted accuracy is still `>= target` (0.95).
If no threshold reaches the target, it falls back to the one with the best accepted accuracy and
reports `False` so you know the target was not met. Only **validation** data is passed in.

```python
def reject_stats(y, pred, conf, t):
```
For a threshold `t`: `coverage` = share accepted, `reject_rate` = 1 − coverage,
`accepted_accuracy` = accuracy only on accepted images, `wrong_accepted` = wrong images that still got through,
`wrong_accepted_rate` = that count divided by ALL images (the "unsafe error" rate).

```python
def cpu_latency_ms(model, image_size, runs=30):
```
Times one image at a time on the CPU (what an edge device does): 5 warm-up runs, then the average of 30.

```python
def main():
```
1. For each of the 4 models: load the checkpoint.
2. Compute the threshold from the validation set.
3. For each of the 5 conditions: predict on the test set, store accuracy/F1 (Table 2) and the reject statistics (Table 3). On the `normal` condition also store Table 1 (params, size, latency).
4. Predict on the textile images: mean confidence and reject rate (Table 4).
5. Save the CSV tables.
6. **Ablation:** A = baseline MobileNetV3 without reject, B = proposed model without reject, C = baseline with reject, D = proposed with reject. Averaged over the 4 degraded conditions.
7. Draw two comparison bar charts.

---

## The confidence rule (the core idea)

```
confidence = highest softmax probability
if confidence >= T  -> ACCEPT  (use the predicted class)
else                -> REJECT  (Unknown / manual inspection)
```
`T` comes from validation data only. The test set is never used to tune it.

---

## Run order (copy-paste)

```bash
pip install -r requirements.txt
python scripts/download_realwaste.py
python scripts/prepare_realwaste.py --source data/raw/realwaste --output data/processed --seed 42 --overwrite
python -m robustwaste.train --data-dir data/processed --model resnet50 --run-name resnet50_baseline --epochs 20
python -m robustwaste.train --data-dir data/processed --model efficientnet_b0 --run-name efficientnet_b0_baseline --epochs 20
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name mobilenet_v3_baseline --epochs 20
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name robustwaste_proposed --epochs 25 --degradation-aware
python -m robustwaste.compare --data-dir data/processed --outputs-dir outputs
```
Or simply run `run_experiments.sh` (Linux/macOS) or `run_experiments.bat` (Windows).
A GPU makes training far faster; on CPU only, reduce `--epochs` for a first test.

---

## Honest notes

- `compare.py` takes every number from your trained checkpoints; nothing is typed in by hand. Do not put numbers in the paper that these scripts did not produce.
- Textile waste may get a **confident** wrong prediction (probably `miscellaneous_trash`). If the OOD reject rate is low, report it as a limitation rather than hiding it.
- The same split and seed (42) are used for every model so the comparison is fair.


---

## Added in v3

| File | Change |
|---|---|
| `scripts/prepare_trashnet.py` | NEW. Splits TrashNet 63/7/30 (test = 758 images, same as the paper) |
| `robustwaste/compare.py` | NEW option `--suffix` so TrashNet runs (names ending `_trashnet`) are compared separately and never overwrite the RealWaste results |

See the README section "Head-to-head with TrashNet" for the exact commands.


---

## Added in v4

| File | Change |
|---|---|
| `robustwaste/robustness.py` | The degradation used in testing is now seeded from the image itself, so the same image always gets the same noise / occlusion. Before, results moved by about 1 point between runs |

TrashNet note: use `--epochs 40 --patience 12` for all four models. With patience 6 the proposed model stopped at epoch 9 (best epoch 3) because the validation set is very small.

---

## Added in v5 (accuracy options)

| Option | What it does |
|---|---|
| `train.py --label-smoothing 0.1` | Softens the training targets so the model is less over-confident. Usually a small free gain. |
| `train.py --image-size 288` | Larger input images (already existed). More detail, slower training and inference. |
| `compare.py --tta` | Test-time augmentation: classify each image and its mirror image, average the probabilities. Applied to every model inside the one script. Costs 2x inference time. |

Rules: pick settings by VALIDATION score only, never by looking at the test set; give all four models the same settings; say in the paper which options were used.

---

## Added in v6

`compare.py --extra "label=run_name"` adds more model variants (for example a degradation-aware ResNet50 as an
"accuracy variant") to every table, next to the four standard models. The `--suffix` is added to the run name as usual.

---

## Added in v7 (ensemble)

`compare.py --extra "label=run1,run2,run3"` now accepts several comma-separated runs. Their predicted probabilities are
averaged (an ensemble). Train the same model with different seeds (`train.py --seed 1`, `--seed 2`, ...) and list the runs.
The ensemble is slower (about 3x for 3 members), so report its latency, and decide the number of members in advance
(do not pick members by looking at the test set).

---

## Added in v8 (stronger backbones)

`train.py --model convnext_tiny` and `--model efficientnet_v2_s` are now available. They are heavier than MobileNetV3
(not edge models) and are meant as "accuracy variants" of the same framework. For ConvNeXt use a smaller learning rate
(`--lr 1e-4`), and `--batch-size 16` if the GPU runs out of memory. Add them to the tables with
`compare.py --extra "label=run_name"`.
