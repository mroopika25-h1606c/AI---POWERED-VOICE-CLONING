from pathlib import Path

import torch

from inference.predict import VoiceSpoofingDetector


CHECKPOINT_PATH = Path(
    "checkpoints/latest.pth"
)

WAV_FOLDER = Path(
    "dataset/robustness_test/clean"
)


if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint missing: {CHECKPOINT_PATH}"
    )


wav_files = list(
    WAV_FOLDER.glob("*.wav")
)

if not wav_files:
    raise FileNotFoundError(
        f"No WAV file found inside: {WAV_FOLDER}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint


fc1_key = next(
    (
        key
        for key in state_dict
        if key.endswith("fc1.weight")
    ),
    None,
)

if fc1_key is None:
    raise KeyError(
        "fc1.weight was not found in checkpoint"
    )


print(
    f"Checkpoint fc1.weight shape: "
    f"{tuple(state_dict[fc1_key].shape)}"
)


detector = VoiceSpoofingDetector(
    model_path=str(CHECKPOINT_PATH),
    device="cpu",
)


print("MODEL LOADED OK")


test_audio = wav_files[0]

print(f"Test WAV: {test_audio}")


result = detector.predict(
    str(test_audio.resolve())
)


print("\nPREDICTION RESULT")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")


real_probability = result["probabilities"]["real"]
fake_probability = result["probabilities"]["fake"]


# Support both 0–1 and 0–100 probability formats
if real_probability <= 1:
    real_probability *= 100

if fake_probability <= 1:
    fake_probability *= 100


print(
    f"REAL probability: "
    f"{real_probability:.2f}%"
)

print(
    f"FAKE probability: "
    f"{fake_probability:.2f}%"
)