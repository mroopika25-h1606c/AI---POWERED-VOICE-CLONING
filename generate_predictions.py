import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from inference.predict import VoiceSpoofingDetector


TEST_CSV = "metadata/LA_cpu_test.csv"
CHECKPOINT = "checkpoints/latest_calibrated.pth"
OUTPUT_FILE = "predictions/predictions.csv"


Path("predictions").mkdir(exist_ok=True)

test_data = pd.read_csv(TEST_CSV)

detector = VoiceSpoofingDetector(
    model_path=CHECKPOINT,
    device="cpu",
)

results = []

for _, row in tqdm(
    test_data.iterrows(),
    total=len(test_data),
    desc="Generating predictions",
):
    start_time = time.perf_counter()

    prediction_result = detector.predict(
        row["audio_path"]
    )

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    if row["label"] == "bonafide":
        true_label = "real"
    else:
        true_label = "fake"

    results.append({
        "audio_id": row["file_id"],
        "true_label": true_label,
        "predicted_label": (
            prediction_result["prediction"].lower()
        ),
        "fake_score": (
            prediction_result["fake_score"]
        ),
        "confidence": (
            prediction_result["confidence"]
        ),
        "threshold": (
            prediction_result["threshold"]
        ),
        "inference_time_ms": round(
            inference_time,
            2,
        ),
    })


predictions = pd.DataFrame(results)

predictions.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nPREDICTION GENERATION COMPLETED")
print(f"Total files: {len(predictions)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nPredicted labels:")
print(predictions["predicted_label"].value_counts())

print("\nCorrect labels:")
print(predictions["true_label"].value_counts())