from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from inference.predict import VoiceSpoofingDetector


TEST_CSV = "metadata/LA_cpu_test.csv"
CHECKPOINT = "checkpoints/latest_calibrated.pth"

ROBUSTNESS_FOLDER = Path("dataset/robustness_test")
RESULTS_FOLDER = Path("results")
PREDICTIONS_FOLDER = Path("predictions")

TEST_TYPES = [
    "clean",
    "noisy",
    "compressed",
    "low_volume",
    "resampled",
]


for test_type in TEST_TYPES:
    (
        ROBUSTNESS_FOLDER / test_type
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

RESULTS_FOLDER.mkdir(exist_ok=True)
PREDICTIONS_FOLDER.mkdir(exist_ok=True)


def add_noise(audio):
    random_generator = np.random.default_rng(42)

    noise = random_generator.normal(
        0,
        1,
        len(audio),
    )

    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)

    target_snr = 10

    noise_scale = np.sqrt(
        signal_power
        / (
            noise_power
            * (10 ** (target_snr / 10))
        )
    )

    result = audio + noise * noise_scale

    return np.clip(result, -1.0, 1.0)


def compress_audio(audio):
    result = np.round(audio * 127) / 127

    return np.clip(result, -1.0, 1.0)


def lower_volume(audio):
    return audio * 0.20


def resample_audio(audio):
    audio_8khz = librosa.resample(
        y=audio,
        orig_sr=16000,
        target_sr=8000,
    )

    return librosa.resample(
        y=audio_8khz,
        orig_sr=8000,
        target_sr=16000,
    )


test_data = pd.read_csv(TEST_CSV)

real_samples = test_data[
    test_data["label"] == "bonafide"
].sample(
    n=10,
    random_state=50,
)

fake_samples = test_data[
    test_data["label"] == "spoof"
].sample(
    n=10,
    random_state=50,
)

selected_samples = pd.concat(
    [real_samples, fake_samples],
    ignore_index=True,
)

detector = VoiceSpoofingDetector(
    model_path=CHECKPOINT,
    device="cpu",
)

all_results = []

print("\nCreating robustness test audio...")

for _, row in tqdm(
    selected_samples.iterrows(),
    total=len(selected_samples),
    desc="Testing original files",
):
    audio, _ = librosa.load(
        row["audio_path"],
        sr=16000,
        mono=True,
    )

    true_label = (
        "real"
        if row["label"] == "bonafide"
        else "fake"
    )

    audio_versions = {
        "clean": audio,
        "noisy": add_noise(audio),
        "compressed": compress_audio(audio),
        "low_volume": lower_volume(audio),
        "resampled": resample_audio(audio),
    }

    for test_type, modified_audio in audio_versions.items():
        output_filename = (
            f"{row['file_id']}_{test_type}.wav"
        )

        output_path = (
            ROBUSTNESS_FOLDER
            / test_type
            / output_filename
        )

        sf.write(
            output_path,
            modified_audio,
            16000,
        )

        result = detector.predict(
            str(output_path.resolve())
        )

        all_results.append({
            "audio_id": row["file_id"],
            "test_type": test_type,
            "true_label": true_label,
            "predicted_label": (
                result["prediction"].lower()
            ),
            "fake_score": result["fake_score"],
            "confidence": result["confidence"],
        })


predictions = pd.DataFrame(all_results)

predictions_file = (
    PREDICTIONS_FOLDER
    / "robustness_predictions.csv"
)

predictions.to_csv(
    predictions_file,
    index=False,
)


metric_results = []

for test_type in TEST_TYPES:
    test_results = predictions[
        predictions["test_type"] == test_type
    ]

    correct_labels = test_results[
        "true_label"
    ].map({
        "real": 0,
        "fake": 1,
    })

    model_labels = test_results[
        "predicted_label"
    ].map({
        "real": 0,
        "fake": 1,
    })

    metric_results.append({
        "test_type": test_type,
        "total_files": len(test_results),
        "accuracy_percent": round(
            accuracy_score(
                correct_labels,
                model_labels,
            ) * 100,
            2,
        ),
        "precision_percent": round(
            precision_score(
                correct_labels,
                model_labels,
                zero_division=0,
            ) * 100,
            2,
        ),
        "recall_percent": round(
            recall_score(
                correct_labels,
                model_labels,
                zero_division=0,
            ) * 100,
            2,
        ),
        "f1_score_percent": round(
            f1_score(
                correct_labels,
                model_labels,
                zero_division=0,
            ) * 100,
            2,
        ),
    })


metrics = pd.DataFrame(metric_results)

metrics_file = (
    RESULTS_FOLDER
    / "robustness_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False,
)

print("\nROBUSTNESS TESTING COMPLETED")
print(metrics.to_string(index=False))

print(f"\nPredictions saved: {predictions_file}")
print(f"Metrics saved: {metrics_file}")