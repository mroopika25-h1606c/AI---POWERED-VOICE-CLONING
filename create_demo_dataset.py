"""
Demo Dataset Generator for Voice Spoofing Detection

Creates a synthetic demo dataset locally for testing/development purposes.
Useful for:
- Testing the training pipeline without downloading ~12 GB
- Verifying data loading code works correctly
- Quick iterations during development

This creates realistic mel-spectrograms mimicking ASVspoof 2019 LA data.

Usage:
    python create_demo_dataset.py --num-train 100 --num-test 20
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
AUDIO_DURATION = 5  # seconds
NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION


class DemoAudioGenerator:
    """Generate synthetic audio samples mimicking real and spoofed voice."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def generate_bonafide(self, duration=5.0):
        """
        Generate synthetic bonafide (real) voice-like audio.
        Uses multiple frequency components to simulate natural speech.
        """
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Simulate speech formants (F1, F2, F3 typical for human voice)
        f1 = 700  # Hz
        f2 = 1220  # Hz
        f3 = 2600  # Hz
        
        # Create formant-like signal with varying amplitude (speech-like)
        envelope = 0.5 * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))  # Slow modulation
        
        signal = (
            0.4 * envelope * np.sin(2 * np.pi * f1 * t) +
            0.3 * envelope * np.sin(2 * np.pi * f2 * t) +
            0.2 * envelope * np.sin(2 * np.pi * f3 * t)
        )
        
        # Add realistic noise
        noise = 0.01 * np.random.randn(num_samples)
        signal = signal + noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        return signal.astype(np.float32)
    
    def generate_spoof(self, duration=5.0):
        """
        Generate synthetic spoofed (AI-generated/synthesized) voice audio.
        Characteristics: more periodic, less natural variation, artifacts.
        """
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Synthesized voice tends to have more stable pitch and less variation
        base_freq = 180 + np.random.randint(-50, 50)  # Simulated pitch
        
        # More periodic and uniform signal (less natural)
        signal = np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonic content (characteristic of vocoders/synthesis)
        for harmonic in range(2, 5):
            signal += 0.3 * np.sin(2 * np.pi * base_freq * harmonic * t) / harmonic
        
        # Add synthesis artifacts (slight distortions)
        signal += 0.02 * np.sin(2 * np.pi * 3000 * t)  # Vocoder artifact
        
        # Add less natural noise (more high-frequency)
        noise = 0.01 * np.random.randn(num_samples)
        noise_filtered = np.convolve(noise, np.ones(50)/50, mode='same')
        signal = signal + noise_filtered
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        return signal.astype(np.float32)
    
    def generate_dataset(self, output_dir, num_bonafide, num_spoof, prefix="LA"):
        """Generate complete dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        protocol_lines = []
        
        # Generate bonafide samples
        logger.info(f"Generating {num_bonafide} bonafide samples...")
        for i in tqdm(range(num_bonafide)):
            file_id = f"{prefix}_{i+1:04d}_0000000"
            audio = self.generate_bonafide()
            audio_path = output_dir / f"{file_id}.flac"
            sf.write(str(audio_path), audio, self.sample_rate)
            
            # Protocol format: file_id spoof_label
            protocol_lines.append(f"{file_id} - - - bonafide\n")
        
        # Generate spoof samples
        logger.info(f"Generating {num_spoof} spoof samples...")
        for i in tqdm(range(num_spoof)):
            file_id = f"{prefix}_{num_bonafide + i + 1:04d}_0000000"
            audio = self.generate_spoof()
            audio_path = output_dir / f"{file_id}.flac"
            sf.write(str(audio_path), audio, self.sample_rate)
            
            # Assign random spoof type (A02-A10)
            spoof_type = f"A{np.random.randint(2, 10):02d}"
            protocol_lines.append(f"{file_id} - - {spoof_type} spoof\n")
        
        # Write protocol file
        protocol_path = output_dir / "protocol.txt"
        with open(protocol_path, 'w') as f:
            f.writelines(protocol_lines)
        
        logger.info(f"✓ Generated {num_bonafide + num_spoof} samples")
        logger.info(f"✓ Protocol file: {protocol_path}")
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo dataset for voice spoofing detection"
    )
    parser.add_argument("--num-train-real", type=int, default=50,
                       help="Number of bonafide training samples")
    parser.add_argument("--num-train-fake", type=int, default=50,
                       help="Number of spoof training samples")
    parser.add_argument("--num-test-real", type=int, default=20,
                       help="Number of bonafide test samples")
    parser.add_argument("--num-test-fake", type=int, default=20,
                       help="Number of spoof test samples")
    parser.add_argument("--output-dir", type=str,
                       default="dataset/LA_demo",
                       help="Output directory for demo dataset")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Demo Dataset Generator for Voice Spoofing Detection")
    print("="*70)
    
    generator = DemoAudioGenerator(sample_rate=SAMPLE_RATE)
    
    # Create training and test directories
    train_dir = Path(args.output_dir) / "train"
    test_dir = Path(args.output_dir) / "test"
    
    print(f"\n📁 Training set:")
    print(f"   Real (bonafide): {args.num_train_real}")
    print(f"   Fake (spoof):    {args.num_train_fake}")
    
    print(f"\n📁 Test set:")
    print(f"   Real (bonafide): {args.num_test_real}")
    print(f"   Fake (spoof):    {args.num_test_fake}")
    
    # Generate datasets
    print(f"\n🔄 Generating training set...")
    generator.generate_dataset(
        train_dir,
        args.num_train_real,
        args.num_train_fake,
        prefix="LA_train"
    )
    
    print(f"\n🔄 Generating test set...")
    generator.generate_dataset(
        test_dir,
        args.num_test_real,
        args.num_test_fake,
        prefix="LA_test"
    )
    
    print("\n" + "="*70)
    print("✅ Demo dataset created successfully!")
    print("="*70)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples:  {args.num_train_real + args.num_train_fake}")
    print(f"   Test samples:      {args.num_test_real + args.num_test_fake}")
    print(f"   Total:             {args.num_train_real + args.num_train_fake + args.num_test_real + args.num_test_fake}")
    print(f"\n📁 Location: {args.output_dir}/")
    
    print(f"\n👉 Next step: python organize_dataset.py --dataset {args.output_dir}")


if __name__ == "__main__":
    main()
