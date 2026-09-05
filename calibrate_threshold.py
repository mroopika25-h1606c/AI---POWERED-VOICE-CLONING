from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from inference.predict import VoiceSpoofingDetector


VALIDATION_CSV = "metadata/LA_cpu_val.csv"
INPUT_CHECKPOINT = "checkpoints/latest.pth"
OUTPUT_CHECKPOINT = "checkpoints/latest_calibrated.pth"


validation_data = pd.read_csv(VALIDATION_CSV)

detector = VoiceSpoofingDetector(
    model_path=INPUT_CHECKPOINT,
    device="cpu",
)

true_labels = []
fake_scores = []

print("\nAnalysing validation audio...")

for number, row in validation_data.iterrows():
    result = detector.predict(row["audio_path"])

    true_label = 0 if row["label"] == "bonafide" else 1

    true_labels.append(true_label)
    fake_scores.append(result["fake_score"])

    print(
        f"{number + 1}/{len(validation_data)} "
        f"label={row['label']} "
        f"fake_score={result['fake_score']:.6f}"
    )


true_labels = np.array(true_labels)
fake_scores = np.array(fake_scores)

best_threshold = 0.5
best_balanced_accuracy = 0.0
best_accuracy = 0.0
best_f1 = 0.0

minimum_score = float(fake_scores.min())
maximum_score = float(fake_scores.max())

thresholds = np.linspace(
    minimum_score,
    maximum_score,
    1000,
)

for threshold in thresholds:
    predictions = (
        fake_scores >= threshold
    ).astype(int)

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predictions,
    )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    if balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = balanced_accuracy
        best_accuracy = accuracy
        best_f1 = f1
        best_threshold = float(threshold)


print("\nTHRESHOLD CALIBRATION RESULT")
print(f"Minimum fake score: {minimum_score:.6f}")
print(f"Maximum fake score: {maximum_score:.6f}")
print(f"Best threshold: {best_threshold:.6f}")
print(
    f"Balanced accuracy: "
    f"{best_balanced_accuracy * 100:.2f}%"
)
print(f"Accuracy: {best_accuracy * 100:.2f}%")
print(f"F1-score: {best_f1 * 100:.2f}%")


checkpoint = torch.load(
    INPUT_CHECKPOINT,
    map_location="cpu",
)

checkpoint["threshold"] = best_threshold
checkpoint["calibration"] = {
    "validation_balanced_accuracy": float(
        best_balanced_accuracy
    ),
    "validation_accuracy": float(best_accuracy),
    "validation_f1": float(best_f1),
    "validation_files": int(len(validation_data)),
}

torch.save(
    checkpoint,
    OUTPUT_CHECKPOINT,
)

print(
    f"\nCalibrated checkpoint saved: "
    f"{OUTPUT_CHECKPOINT}"
)