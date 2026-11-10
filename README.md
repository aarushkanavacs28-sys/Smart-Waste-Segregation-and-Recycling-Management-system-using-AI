# RobustWaste-AIoT

Reference implementation for the proposed conference work:

**RobustWaste-AIoT: A Confidence-Aware Edge AI and IoT Framework for Smart Waste Segregation under Real-World Visual Degradation**

This project is software-first. No physical waste-bin hardware or sensors are required for the main experiments.

## Results at a glance

Test accuracy on RealWaste (666 test images, 6 classes, seed 42). All models use the same split and settings
(288 px, label smoothing 0.1, up to 40 epochs, patience 12).

| Model | Clean accuracy | Mean accuracy under 4 degradations |
|---|---|---|
| **RobustWaste (MobileNetV3-Large, degradation-aware), main run** | **96.10%** | 95.4% |
| Same configuration, repeat run | 95.35% | see `results/realwaste_repeat_and_ensemble` |
| 3-seed ensemble of the above | 95.80% | see `results/realwaste_repeat_and_ensemble` |
| RobustWaste with ConvNeXt-Tiny (heavier accuracy variant) | 96.55% | 95.7% |
| EfficientNet-B0 | 95.95% | 76.1% |
| ResNet50 | 95.35% | 84.5% |
| MobileNetV3-Large, normal training | 95.05% | 89.9% |

Main findings: degradation-aware training keeps accuracy within about 1.7 points of the clean result under low light,
blur, noise and occlusion, while conventionally trained models lose up to 44 points. The lightweight model is about
5.5x smaller and 6.6x faster on CPU than ResNet50. Retraining the same configuration changes accuracy by about
0.75 points, so differences below one point are within run-to-run noise. Unseen textile waste is mostly accepted by
every model (only 5-14% rejected), which is reported as a limitation.

TrashNet (758 test images, 63/7/30 split as in Rajakumaran et al.): proposed 94.20%, reproduced ResNet50 94.33%.

## Repository layout

| Folder / file | Content |
|---|---|
| `robustwaste/` | Model, training, evaluation, robustness, reject threshold, comparison and IoT code |
| `scripts/` | Dataset download and preparation (RealWaste, TrashNet) |
| `dashboard/` | Streamlit dashboard for the MQTT events |
| `results/` | Result tables (CSV), figures and training logs of the final runs |
| `models/` | Trained lightweight proposed model (`robustwaste_mobilenetv3_proposed.pt`, about 16 MB) |
| `docs/` | Comparative analysis and proposed-work document |
| `CODE_GUIDE.md` | Plain-language explanation of every file and option |

Datasets are not included; the scripts download them. The ConvNeXt weights (106 MB) are too large for the repository
and are not included; add a release link here if you publish them.

Quick demo with the included model:

```bash
python -m robustwaste.infer --checkpoint models/robustwaste_mobilenetv3_proposed.pt --image path/to/photo.jpg --threshold 0.5
```

Note: the 96.10% main-run model file was not saved. The included model is a retrained copy with the same
configuration (test accuracy 95.35%). The tables in `results/realwaste_main_run` belong to the 96.10% run.

## Data sources

- RealWaste: Single, Iranmanesh and Raad, *Information* 14(12):633, 2023 (UCI Machine Learning Repository).
- TrashNet: Thung and Yang, Stanford CS229 project, 2016.

## Implemented pipeline

- Automatic RealWaste download helper
- RealWaste preparation into train/validation/test folders
- Six in-distribution classes:
  - `paper_cardboard`
  - `organic`
  - `glass`
  - `metal`
  - `plastic`
  - `miscellaneous_trash`
- `textile_trash` is held out completely as an **unknown/OOD** test set
- ResNet50, EfficientNet-B0, and MobileNetV3-Large baselines
- Proposed MobileNetV3-Large with **degradation-aware training**
- Robustness tests under:
  - normal images
  - low light
  - blur
  - Gaussian noise
  - partial occlusion
- Confidence-threshold selection on the validation set
- Confidence-aware ACCEPT/REJECT routing
- OOD rejection evaluation
- MQTT event publishing and subscriber logging
- Streamlit monitoring dashboard
- One-command comparison of all four models (`robustwaste.compare`) that generates every paper table, the A-D ablation and comparison figures
- CSV/JSON/PNG outputs suitable for paper tables and figures

## Recommended dataset

### Primary: RealWaste

RealWaste contains 4,752 images collected in an authentic landfill environment across nine material classes.

This project uses the following mapping:

| RealWaste class | Experimental use |
|---|---|
| Cardboard | paper_cardboard |
| Paper | paper_cardboard |
| Food Organics | organic |
| Vegetation | organic |
| Glass | glass |
| Metal | metal |
| Plastic | plastic |
| Miscellaneous Trash | miscellaneous_trash |
| Textile Trash | OOD/Unknown test only |

The held-out textile class is never seen during training, which makes it useful for evaluating the confidence-aware reject mechanism.

### Secondary benchmark: TrashNet

TrashNet can be added as a secondary benchmark. It has 2,527 images across six classes:
glass, paper, cardboard, plastic, metal, and trash.

For this paper, RealWaste is the recommended primary dataset because it better matches the real-world visual degradation story.

---

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bat
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

If needed, install the PyTorch build appropriate for your CUDA environment from the official PyTorch installation page.

---

## Download RealWaste

```bash
python scripts/download_realwaste.py
```

The script attempts to download the official UCI archive and extracts it to:

```text
data/raw/realwaste/
```

If automatic download is blocked, download and extract the UCI RealWaste dataset manually anywhere under that folder.

---

## Prepare dataset

```bash
python scripts/prepare_realwaste.py \
    --source data/raw/realwaste \
    --output data/processed \
    --seed 42
```

Output:

```text
data/processed/
├── train/
├── val/
├── test/
└── ood_textile/
```

Default split:
- 70% train
- 15% validation
- 15% test

---

> Offline machine or quick smoke test? Add `--no-pretrained` to any training command to skip the
> ImageNet weight download (accuracy will be lower; use real runs for the paper).

## Train baselines

ResNet50:

```bash
python -m robustwaste.train \
  --data-dir data/processed \
  --model resnet50 \
  --run-name resnet50_baseline \
  --epochs 20
```

EfficientNet-B0:

```bash
python -m robustwaste.train \
  --data-dir data/processed \
  --model efficientnet_b0 \
  --run-name efficientnet_b0_baseline \
  --epochs 20
```

MobileNetV3:

```bash
python -m robustwaste.train \
  --data-dir data/processed \
  --model mobilenet_v3_large \
  --run-name mobilenet_v3_baseline \
  --epochs 20
```

---

## Train proposed model

```bash
python -m robustwaste.train \
  --data-dir data/processed \
  --model mobilenet_v3_large \
  --run-name robustwaste_proposed \
  --epochs 25 \
  --degradation-aware
```

---

## Evaluate clean test performance

```bash
python -m robustwaste.evaluate \
  --data-dir data/processed \
  --checkpoint outputs/robustwaste_proposed/best.pt \
  --split test
```

Outputs:
- `metrics_test.json`
- `predictions_test.csv`
- `confusion_matrix_test.png`

---

## Select confidence threshold

The threshold is selected only from the validation set.

```bash
python -m robustwaste.threshold \
  --data-dir data/processed \
  --checkpoint outputs/robustwaste_proposed/best.pt \
  --target-accepted-accuracy 0.95
```

Outputs:
- `threshold.json`
- `threshold_scan.csv`

---

## Robustness evaluation

```bash
python -m robustwaste.robustness \
  --data-dir data/processed \
  --checkpoint outputs/robustwaste_proposed/best.pt
```

Outputs include:
- `robustness_results.csv`
- per-condition predictions
- `robustness_accuracy.png`
- `robustness_f1.png`

---

## OOD / unknown rejection evaluation

```bash
python -m robustwaste.ood \
  --data-dir data/processed \
  --checkpoint outputs/robustwaste_proposed/best.pt
```

The held-out `Textile Trash` class is used as unseen data.

---

## Compare all models (generates the paper tables)

Run this after the four models are trained (`resnet50_baseline`, `efficientnet_b0_baseline`,
`mobilenet_v3_baseline`, `robustwaste_proposed`):

```bash
python -m robustwaste.compare --data-dir data/processed --outputs-dir outputs
```

Everything is written to `outputs/comparison/`:

| File | Content |
|---|---|
| `table1_clean.csv` | accuracy, macro P/R/F1, parameters, size, single-image CPU latency |
| `table2_robustness.csv` | accuracy and macro-F1 per condition + drop vs normal |
| `table3_confidence_aware.csv` | coverage, reject rate, accepted accuracy, wrong-accepted rate per condition |
| `table4_ood.csv` | held-out textile: mean confidence and reject rate |
| `table5_ablation.csv` | configs A/B/C/D (normal vs degradation-aware, with/without reject) |
| `fig_robustness_comparison.png`, `fig_wrong_accepted_comparison.png` | figures for the paper |

The confidence threshold of every model is chosen on the validation set only
(target accepted accuracy 0.95, change with `--target-accepted-accuracy`).
A model whose folder is missing is skipped, so you can run it after only some trainings.

---

## Single-image inference

```bash
python -m robustwaste.infer \
  --checkpoint outputs/robustwaste_proposed/best.pt \
  --image path/to/example.jpg
```

Simulate a bad camera view (the event then carries `image_condition`):

```bash
python -m robustwaste.infer \
  --checkpoint outputs/robustwaste_proposed/best.pt \
  --image path/to/example.jpg \
  --condition low_light --severity 0.65
```

With a manual threshold:

```bash
python -m robustwaste.infer \
  --checkpoint outputs/robustwaste_proposed/best.pt \
  --image path/to/example.jpg \
  --threshold 0.78
```

---

## MQTT / virtual IoT

Publish an inference event:

```bash
python -m robustwaste.infer \
  --checkpoint outputs/robustwaste_proposed/best.pt \
  --image path/to/example.jpg \
  --mqtt-host localhost \
  --mqtt-port 1883 \
  --mqtt-topic robustwaste/events
```

Subscribe and log:

```bash
python -m robustwaste.mqtt_subscriber \
  --host localhost \
  --port 1883 \
  --topic robustwaste/events \
  --csv outputs/iot_events.csv
```

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard reads `outputs/iot_events.csv` by default.

---

## Recommended paper comparison

Train and compare:

1. ResNet50 baseline
2. EfficientNet-B0 baseline
3. MobileNetV3 baseline
4. Proposed MobileNetV3 + degradation-aware training

Report:
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Parameters
- Model size
- Mean inference latency
- Low-light robustness
- Blur robustness
- Noise robustness
- Occlusion robustness
- Confidence coverage/reject rate
- Accepted-sample accuracy
- OOD rejection rate

Do not fill paper tables until these scripts generate the experimental numbers.

## Methodological note

`Unknown/Reject` is a **decision state**, not a seventh training class.
This makes the confidence-aware rejection experiment cleaner.

The final test set must not be used to tune the confidence threshold.


---

## Head-to-head with TrashNet (same dataset and split as Rajakumaran et al.)

Accuracy numbers from different datasets cannot be compared fairly (RealWaste is harder than
TrashNet). To compare with a TrashNet paper, run your models on TrashNet with the same
63/7/30 split (1593/176/758 images in the paper; ours gives 1591/178/758).

```bash
git clone https://github.com/garythung/trashnet
unzip -q trashnet/data/dataset-resized.zip -d trashnet_raw
python scripts/prepare_trashnet.py --source trashnet_raw --output data/trashnet_processed --seed 42 --overwrite

# train all four with a _trashnet suffix (same settings for every model).
# --patience 12 matters: the TrashNet validation set is tiny (~176 images), so patience 6
# can stop training far too early.
python -m robustwaste.train --data-dir data/trashnet_processed --model resnet50 --run-name resnet50_baseline_trashnet --epochs 40 --patience 12
python -m robustwaste.train --data-dir data/trashnet_processed --model efficientnet_b0 --run-name efficientnet_b0_baseline_trashnet --epochs 40 --patience 12
python -m robustwaste.train --data-dir data/trashnet_processed --model mobilenet_v3_large --run-name mobilenet_v3_baseline_trashnet --epochs 40 --patience 12
python -m robustwaste.train --data-dir data/trashnet_processed --model mobilenet_v3_large --run-name robustwaste_proposed_trashnet --epochs 40 --patience 12 --degradation-aware

python -m robustwaste.compare --data-dir data/trashnet_processed --outputs-dir outputs --suffix _trashnet
```

Results go to `outputs/comparison_trashnet/`. TrashNet has no unknown-waste folder, so Table 4 is
skipped automatically. The `resnet50_baseline_trashnet` run is the fair reproduction of the
ResNet50 approach in the TrashNet paper.

---

## Accuracy options (v5)

```bash
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large \
  --run-name robustwaste_proposed_v2 --epochs 40 --patience 12 \
  --image-size 288 --label-smoothing 0.1 --degradation-aware
python -m robustwaste.compare --data-dir data/processed --outputs-dir outputs --suffix _v2          # no TTA
python -m robustwaste.compare --data-dir data/processed --outputs-dir outputs --suffix _v2 --tta    # with TTA
```
Train all four models with the same options and `_v2` run names so the comparison stays fair.
Copy `outputs/comparison_v2` somewhere safe before running the `--tta` version, which overwrites it.

### Ensemble of seeds (v7)

```bash
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name robustwaste_proposed_v2    --seed 42 --epochs 40 --patience 12 --image-size 288 --label-smoothing 0.1 --degradation-aware
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name robustwaste_proposed_s1_v2 --seed 1  --epochs 40 --patience 12 --image-size 288 --label-smoothing 0.1 --degradation-aware
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name robustwaste_proposed_s2_v2 --seed 2  --epochs 40 --patience 12 --image-size 288 --label-smoothing 0.1 --degradation-aware
python -m robustwaste.compare --data-dir data/processed --outputs-dir outputs --suffix _v2 \
  --extra "RobustWaste-Ens3 (3 seeds)=robustwaste_proposed,robustwaste_proposed_s1,robustwaste_proposed_s2"
```
