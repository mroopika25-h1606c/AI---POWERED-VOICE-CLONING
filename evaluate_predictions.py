import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


INPUT_FILE = "predictions/predictions.csv"
RESULT_FOLDER = Path("results")
RESULT_FOLDER.mkdir(exist_ok=True)


data = pd.read_csv(INPUT_FILE)

true_labels = data["true_label"].map({
    "real": 0,
    "fake": 1,
})

predicted_labels = data["predicted_label"].map({
    "real": 0,
    "fake": 1,
})

if true_labels.isna().any():
    raise ValueError("Invalid value in true_label")

if predicted_labels.isna().any():
    raise ValueError("Invalid value in predicted_label")


matrix = confusion_matrix(
    true_labels,
    predicted_labels,
)

true_negative = int(matrix[0, 0])
false_positive = int(matrix[0, 1])
false_negative = int(matrix[1, 0])
true_positive = int(matrix[1, 1])


false_positive_rates, true_positive_rates, thresholds = (
    roc_curve(
        true_labels,
        data["fake_score"],
    )
)

false_negative_rates = 1 - true_positive_rates

eer_index = np.nanargmin(
    np.abs(
        false_positive_rates
        - false_negative_rates
    )
)

eer = (
    false_positive_rates[eer_index]
    + false_negative_rates[eer_index]
) / 2


metrics = {
    "total_test_files": int(len(data)),
    "accuracy_percent": round(
        accuracy_score(
            true_labels,
            predicted_labels,
        ) * 100,
        2,
    ),
    "precision_percent": round(
        precision_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ) * 100,
        2,
    ),
    "recall_percent": round(
        recall_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ) * 100,
        2,
    ),
    "f1_score_percent": round(
        f1_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ) * 100,
        2,
    ),
    "roc_auc_percent": round(
        roc_auc_score(
            true_labels,
            data["fake_score"],
        ) * 100,
        2,
    ),
    "eer_percent": round(eer * 100, 2),
    "average_inference_time_ms": round(
        data["inference_time_ms"].mean(),
        2,
    ),
    "true_real": true_negative,
    "real_predicted_as_fake": false_positive,
    "fake_predicted_as_real": false_negative,
    "true_fake": true_positive,
    "confusion_matrix": matrix.tolist(),
}


output_path = RESULT_FOLDER / "metrics.json"

with open(
    output_path,
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(
        metrics,
        output_file,
        indent=4,
    )


print("\nMODEL EVALUATION")
print("------------------------------")

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")

print(f"\nSaved to: {output_path}")