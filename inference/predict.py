"""
AASIST Model Inference Pipeline for Voice Spoofing Detection

This is the CORE inference module that Member 3 can import and use from FastAPI.
Clean, simple, production-ready.

Usage Example:
    from inference.predict import VoiceSpoofingDetector
    
    detector = VoiceSpoofingDetector()
    result = detector.predict("audio.wav")
    print(result)
    # Output: {'prediction': 'REAL', 'confidence': 98.5, 'probabilities': {...}}
"""

import numpy as np
import torch
import json
import logging
from typing import Dict, Union, Optional, Tuple
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import from preprocessing module
try:
    from preprocessing.audio_preprocessing import AudioPreprocessor
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from preprocessing.audio_preprocessing import AudioPreprocessor


class AASISTModel(torch.nn.Module):
    """
    Simplified AASIST-inspired architecture for voice spoofing detection.
    
    This model uses:
    - Mel-spectrogram input (128 mel bins, ~157 time steps)
    - Convolutional frontend for feature extraction
    - Attention mechanism for temporal modeling
    - Binary classification output (Real vs. Spoof)
    """
    
    def __init__(self, num_mels: int = 128, num_classes: int = 2):
        super(AASISTModel, self).__init__()
        
        self.num_mels = num_mels
        self.num_classes = num_classes
        
        # Convolutional frontend
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        
        # Attention layer
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=128 * 4 * 4,
            num_heads=8,
            batch_first=True
        )
        
        # Classification head
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256, num_classes)
    
    def forward(self, x):
        """Forward pass through the model."""
        # Convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.pool3(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten and attention
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Apply attention
        x_expanded = x.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.attention(x_expanded, x_expanded, x_expanded)
        x = attn_output.squeeze(1)
        
        # Classification head
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class VoiceSpoofingDetector:
    """
    Main inference class for voice spoofing detection.
    
    This is the PRIMARY interface that Member 3 should use from FastAPI.
    
    Features:
    - Simple predict() method that takes audio path → returns prediction
    - Automatic preprocessing
    - Clean JSON output
    - GPU support (auto-detects)
    - Error handling and logging
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the voice spoofing detector.
        
        Args:
            model_path (str, optional): Path to pretrained model checkpoint.
                                       If None, uses untrained model (for demo).
            device (str, optional): Device to run on ('cuda' or 'cpu'). 
                                   Auto-detects if None.
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🎙️ VoiceSpoofingDetector initialized on device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=128, duration=5.0)
        
        # Initialize model
        self.model = AASISTModel(num_mels=128, num_classes=2)
        self.model.to(self.device)
        self.model.eval()
        
        # Load pretrained weights if provided
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("⚠️ No pretrained model loaded. Using untrained model for demo only.")
    
    def load_model(self, model_path: str):
        """
        Load pretrained model checkpoint.
        
        Args:
            model_path (str): Path to .pth checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"✓ Loaded pretrained model: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    @torch.no_grad()
    def predict(self, audio_path: str, return_raw: bool = False) -> Dict:
        """
        Predict whether audio is REAL or FAKE (spoofed).
        
        This is the MAIN FUNCTION that Member 3 should call from FastAPI.
        
        Args:
            audio_path (str): Path to audio file (WAV, FLAC, MP3, etc.)
            return_raw (bool): If True, also return raw model outputs
            
        Returns:
            Dict with keys:
                - 'prediction': str, 'REAL' or 'FAKE'
                - 'confidence': float, 0-100
                - 'probabilities': dict with 'real' and 'fake' probabilities
                - 'audio_path': str, input file path
                - 'raw_scores': dict (optional, if return_raw=True)
                
        Example:
            result = detector.predict("sample.wav")
            # Output:
            # {
            #     'prediction': 'REAL',
            #     'confidence': 98.5,
            #     'probabilities': {'real': 0.985, 'fake': 0.015},
            #     'audio_path': 'sample.wav'
            # }
        """
        try:
            # Step 1: Preprocess audio
            mel_spec, metadata = self.preprocessor.preprocess(audio_path, verbose=False)
            
            # Step 2: Prepare input tensor
            # Add channel and batch dimensions: (1, 128, time) → (1, 1, 128, time)
            input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Step 3: Run inference
            logits = self.model(input_tensor)
            
            # Step 4: Convert to probabilities
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Classes: 0 = Real (Bonafide), 1 = Fake (Spoof)
            prob_real = float(probs[0])
            prob_fake = float(probs[1])
            
            # Step 5: Determine prediction
            prediction = "REAL" if prob_real > prob_fake else "FAKE"
            confidence = max(prob_real, prob_fake) * 100
            
            # Step 6: Format output
            result = {
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'real': round(prob_real, 4),
                    'fake': round(prob_fake, 4)
                },
                'audio_path': str(audio_path)
            }
            
            # Add raw scores if requested
            if return_raw:
                result['raw_scores'] = {
                    'real_logit': float(logits[0, 0].item()),
                    'fake_logit': float(logits[0, 1].item())
                }
            
            logger.info(f"✓ Prediction: {prediction} ({confidence:.1f}%)")
            return result
            
        except FileNotFoundError as e:
            logger.error(f"✗ File not found: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"✗ Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, audio_paths: list) -> list:
        """
        Predict on multiple audio files.
        
        Args:
            audio_paths (list): List of audio file paths
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"✗ Failed to process {audio_path}: {str(e)}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e),
                    'prediction': 'ERROR'
                })
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS FOR FASTAPI INTEGRATION
# ============================================================================

# Global detector instance (lazy-loaded)
_detector_instance = None

def get_detector(model_path: Optional[str] = None) -> VoiceSpoofingDetector:
    """
    Get or create the detector instance (singleton pattern).
    
    Useful for FastAPI to avoid reloading the model on every request.
    
    Args:
        model_path (str, optional): Path to model checkpoint
        
    Returns:
        VoiceSpoofingDetector: Detector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = VoiceSpoofingDetector(model_path=model_path)
    return _detector_instance


def predict_audio(audio_path: str) -> Dict:
    """
    Standalone convenience function for single prediction.
    
    Usage:
        from inference.predict import predict_audio
        result = predict_audio("audio.wav")
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        Dict: Prediction result
    """
    detector = get_detector()
    return detector.predict(audio_path)
