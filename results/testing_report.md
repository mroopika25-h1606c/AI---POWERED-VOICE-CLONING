# VoiceGuard Model Testing Report

## Project

AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

## Dataset

- Dataset: ASVspoof 2019 LA
- Training samples: 1,000
- Validation samples: 200
- Testing samples: 400
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

- Accuracy: 81.0%
- Precision: 90.26%
- Fake Recall: 69.5%
- F1-score: 78.53%
- ROC-AUC: 83.72%
- Equal Error Rate: 21.75%
- Average inference time: 93.31 ms

## Confusion Matrix Results

- Correctly detected real voices: 185
- Real voices incorrectly blocked: 15
- Fake voices incorrectly allowed: 61
- Correctly detected fake voices: 139

## Confusion Matrix

[[185, 15], [61, 139]]

## Conclusion

The prototype successfully performs end-to-end voice spoofing detection using ASVspoof 2019 LA audio.

The system preprocesses audio, generates an AI prediction and confidence score, and can send the result to the cybersecurity risk engine.

The current model is suitable as a university hackathon prototype. Further improvements can include additional training data, pretrained official AASIST weights, audio augmentation and testing against newer voice-cloning systems.
