"""
Test Script for Voice Spoofing Detection

This script tests the inference pipeline and shows usage examples.
Run this to verify everything is working correctly.

Usage:
    python test_inference.py
"""

import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference.predict import VoiceSpoofingDetector, predict_audio
from preprocessing.audio_preprocessing import AudioPreprocessor, preprocess_audio_file


def create_test_audio(output_path: str, duration: float = 3.0, frequency: float = 440.0):
    """
    Create a simple test audio file (sine wave).
    
    Args:
        output_path (str): Path to save the test audio
        duration (float): Duration in seconds
        frequency (float): Frequency in Hz
    """
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    sf.write(output_path, audio, sample_rate)
    print(f"✓ Created test audio: {output_path}")
    return output_path


def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("\n" + "="*70)
    print("TEST 1: Audio Preprocessing")
    print("="*70)
    
    # Create test audio
    test_audio_path = project_root / "test_audio.wav"
    create_test_audio(str(test_audio_path), duration=5.0)
    
    # Test preprocessing
    try:
        mel_spec, metadata = preprocess_audio_file(str(test_audio_path))
        
        print(f"✓ Preprocessing successful!")
        print(f"  - Mel-spectrogram shape: {mel_spec.shape}")
        print(f"  - Duration: {metadata['original_duration']:.2f}s")
        print(f"  - Sample rate: {metadata['sample_rate']} Hz")
        print(f"  - Min value: {mel_spec.min():.2f}")
        print(f"  - Max value: {mel_spec.max():.2f}")
        
        # Cleanup
        test_audio_path.unlink()
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing failed: {str(e)}")
        return False


def test_inference():
    """Test the inference pipeline."""
    print("\n" + "="*70)
    print("TEST 2: Voice Spoofing Detection Inference")
    print("="*70)
    
    # Create test audio
    test_audio_path = project_root / "test_audio.wav"
    create_test_audio(str(test_audio_path), duration=5.0)
    
    try:
        # Initialize detector
        print("📍 Initializing detector...")
        detector = VoiceSpoofingDetector()
        
        # Run prediction
        print("📍 Running inference...")
        result = detector.predict(str(test_audio_path))
        
        print(f"✓ Inference successful!")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']}%")
        print(f"  - Probabilities: {result['probabilities']}")
        
        # Cleanup
        test_audio_path.unlink()
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_function():
    """Test the convenience function."""
    print("\n" + "="*70)
    print("TEST 3: Convenience Function (Direct Usage)")
    print("="*70)
    
    # Create test audio
    test_audio_path = project_root / "test_audio.wav"
    create_test_audio(str(test_audio_path), duration=5.0)
    
    try:
        print("📍 Using predict_audio() convenience function...")
        result = predict_audio(str(test_audio_path))
        
        print(f"✓ Direct function call successful!")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']}%")
        
        # Cleanup
        test_audio_path.unlink()
        return True
        
    except Exception as e:
        print(f"✗ Direct function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference():
    """Test batch inference."""
    print("\n" + "="*70)
    print("TEST 4: Batch Inference")
    print("="*70)
    
    # Create multiple test audios
    test_paths = []
    for i in range(3):
        path = project_root / f"test_audio_{i}.wav"
        create_test_audio(str(path), duration=3.0, frequency=440 + i*50)
        test_paths.append(str(path))
    
    try:
        print("📍 Initializing detector...")
        detector = VoiceSpoofingDetector()
        
        print("📍 Running batch inference...")
        results = detector.predict_batch(test_paths)
        
        print(f"✓ Batch inference successful!")
        print(f"  - Processed {len(results)} files")
        
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"    [{i+1}] {result['prediction']} ({result['confidence']}%)")
            else:
                print(f"    [{i+1}] ERROR: {result['error']}")
        
        # Cleanup
        for path in test_paths:
            Path(path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Batch inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_usage_examples():
    """Print usage examples for Member 3."""
    print("\n" + "="*70)
    print("USAGE EXAMPLES FOR FASTAPI INTEGRATION (Member 3)")
    print("="*70)
    
    examples = """
1. IMPORT AND USE IN YOUR CODE:
   
   from inference.predict import VoiceSpoofingDetector
   
   detector = VoiceSpoofingDetector()
   result = detector.predict("audio.wav")
   print(result)
   # Output: {'prediction': 'REAL', 'confidence': 98.5, ...}

2. USE CONVENIENCE FUNCTION:
   
   from inference.predict import predict_audio
   
   result = predict_audio("audio.wav")
   print(f"Prediction: {result['prediction']}")

3. USE IN FASTAPI (See api.py for full example):
   
   from fastapi import FastAPI, File, UploadFile
   from inference.predict import VoiceSpoofingDetector
   
   app = FastAPI()
   detector = VoiceSpoofingDetector()
   
   @app.post("/predict")
   async def predict(file: UploadFile):
       result = detector.predict(file.filename)
       return result

4. BATCH PROCESSING:
   
   detector = VoiceSpoofingDetector()
   results = detector.predict_batch(["audio1.wav", "audio2.wav", "audio3.wav"])
   
   for result in results:
       print(f"{result['audio_path']}: {result['prediction']}")

5. WITH PRETRAINED MODEL:
   
   detector = VoiceSpoofingDetector(model_path="models/aasist_best.pth")
   result = detector.predict("audio.wav")

6. CUSTOM DEVICE:
   
   # Force CPU (useful if CUDA causes issues)
   detector = VoiceSpoofingDetector(device="cpu")
   
   # Or use GPU if available
   detector = VoiceSpoofingDetector(device="cuda")
"""
    
    print(examples)


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " VOICE SPOOFING DETECTION - INFERENCE PIPELINE TEST ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    # Run tests
    results['preprocessing'] = test_preprocessing()
    results['inference'] = test_inference()
    results['direct_function'] = test_direct_function()
    results['batch_inference'] = test_batch_inference()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Inference pipeline is ready.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    # Print usage examples
    print_usage_examples()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
