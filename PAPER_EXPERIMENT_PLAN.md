# Conference Paper Experiment Plan

## Proposed contribution

RobustWaste-AIoT combines:

1. lightweight MobileNetV3 waste classification,
2. degradation-aware training,
3. confidence-aware reject routing,
4. MQTT-based IoT monitoring.

## Main research question

Can a lightweight waste-classification model remain reliable under realistic visual degradation,
and can confidence-aware rejection reduce incorrect automatic segregation?

## Hypotheses

### H1
Degradation-aware MobileNetV3 should show smaller performance loss than conventional training
under low light, blur, noise, and occlusion.

### H2
A confidence threshold selected on the validation set should increase the accuracy of automatically
accepted samples, with a corresponding reject-rate tradeoff.

### H3
The held-out textile class provides a simple unseen/OOD test for whether uncertain waste can be rejected.

## Models

- ResNet50 baseline
- EfficientNet-B0 baseline
- MobileNetV3 baseline
- Proposed degradation-aware MobileNetV3

## RealWaste mapping

- Cardboard + Paper -> paper_cardboard
- Food Organics + Vegetation -> organic
- Glass -> glass
- Metal -> metal
- Plastic -> plastic
- Miscellaneous Trash -> miscellaneous_trash
- Textile Trash -> held-out OOD/unknown

## Tables to generate

### Table 1: clean classification
- Model
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Parameters
- Model size
- Latency

### Table 2: robustness
- Model
- Normal
- Low light
- Blur
- Noise
- Occlusion

### Table 3: confidence-aware decision
- Threshold
- Coverage
- Reject Rate
- Accepted Accuracy
- Wrong Accepted Samples

### Table 4: OOD evaluation
- OOD set
- Number of samples
- Mean confidence
- Reject rate

## Ablation

A. MobileNetV3 normal training + no reject  
B. MobileNetV3 degradation-aware training + no reject  
C. MobileNetV3 normal training + confidence reject  
D. MobileNetV3 degradation-aware training + confidence reject  

## Rules

- Use one fixed dataset split for all models.
- Select the confidence threshold on validation only.
- Keep the final test set untouched until evaluation.
- Do not fabricate results.
