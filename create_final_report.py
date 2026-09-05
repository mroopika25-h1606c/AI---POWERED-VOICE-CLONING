import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)


PREDICTIONS_FILE = "predictions/predictions.csv"
METRICS_FILE = "results/metrics.json"
RESULTS_FOLDER = Path("results")

RESULTS_FOLDER.mkdir(exist_ok=True)


predictions = pd.read_csv(PREDICTIONS_FILE)

with open(
    METRICS_FILE,
    "r",
    encoding="utf-8",
) as file:
    metrics = json.load(file)


true_labels = predictions["true_label"].map({
    "real": 0,
    "fake": 1,
})

predicted_labels = predictions["predicted_label"].map({
    "real": 0,
    "fake": 1,
})


# ---------------------------------------
# CONFUSION MATRIX
# ---------------------------------------

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["REAL", "FAKE"],
)

display.plot(
    cmap="Blues",
    values_format="d",
)

plt.title("Voice Cloning Detection Confusion Matrix")
plt.tight_layout()

confusion_path = (
    RESULTS_FOLDER / "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
)

plt.close()


# ---------------------------------------
# ROC CURVE
# ---------------------------------------

false_positive_rate, true_positive_rate, _ = roc_curve(
    true_labels,
    predictions["fake_score"],
)

roc_auc = auc(
    false_positive_rate,
    true_positive_rate,
)

plt.figure(figsize=(7, 6))

plt.plot(
    false_positive_rate,
    true_positive_rate,
    color="blue",
    linewidth=2,
    label=f"Model AUC = {roc_auc:.4f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    color="red",
    linestyle="--",
    label="Random model",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Voice Cloning Detection ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = RESULTS_FOLDER / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
)

plt.close()


# ---------------------------------------
# TESTING REPORT
# ---------------------------------------

report = f"""# VoiceGuard Model Testing Report

## Project

AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

## Dataset

- Dataset: ASVspoof 2019 LA
- Training samples: 1,000
- Validation samples: 200
- Testing samples: {metrics["total_test_files"]}
- Classes: REAL/BONAFIDE and FAKE/SPOOF
- Audio sample rate: 16 kHz
- Audio duration: 5 seconds
- Training device: CPU

## Model

- Model: AASIST-inspired voice spoofing classifier
- Input: Mel-spectrogram
- Input shape: 1 × 128 × 501
- Output: REAL or FAKE
- Checkpoint: latest_calibrated.pth

## Evaluation Results

- Accuracy: {metrics["accuracy_percent"]}%
- Precision: {metrics["precision_percent"]}%
- Fake Recall: {metrics["recall_percent"]}%
- F1-score: {metrics["f1_score_percent"]}%
- ROC-AUC: {metrics["roc_auc_percent"]}%
- Equal Error Rate: {metrics["eer_percent"]}%
- Average inference time: {metrics["average_inference_time_ms"]} ms

## Confusion Matrix Results

- Correctly detected real voices: {metrics["true_real"]}
- Real voices incorrectly blocked: {metrics["real_predicted_as_fake"]}
- Fake voices incorrectly allowed: {metrics["fake_predicted_as_real"]}
- Correctly detected fake voices: {metrics["true_fake"]}

## Confusion Matrix

{metrics["confusion_matrix"]}

## Conclusion

The prototype successfully performs end-to-end voice spoofing detection using ASVspoof 2019 LA audio.

The system preprocesses audio, generates an AI prediction and confidence score, and can send the result to the cybersecurity risk engine.

The current model is suitable as a university hackathon prototype. Further improvements can include additional training data, pretrained official AASIST weights, audio augmentation and testing against newer voice-cloning systems.
"""

report_path = RESULTS_FOLDER / "testing_report.md"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:
    file.write(report)


print("\nFINAL REPORT CREATED")
print(f"Confusion matrix: {confusion_path}")
print(f"ROC curve: {roc_path}")
print(f"Testing report: {report_path}")