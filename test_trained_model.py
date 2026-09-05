import pandas as pd

from inference.predict import VoiceSpoofingDetector

TEST_CSV = "metadata/LA_cpu_test.csv"
CHECKPOINT = "checkpoints/best_calibrated.pth"

test_data = pd.read_csv(TEST_CSV)

real_audio = test_data[
    test_data["label"] == "bonafide"
].iloc[0]

fake_audio = test_data[
    test_data["label"] == "spoof"
].iloc[0]

detector = VoiceSpoofingDetector(
    model_path=CHECKPOINT,
    device="cpu",
)

print("\nREAL AUDIO")
print("Correct label: REAL")
print(detector.predict(real_audio["audio_path"]))

print("\nFAKE AUDIO")
print("Correct label: FAKE")
print(detector.predict(fake_audio["audio_path"]))