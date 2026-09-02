"""
Dataset Loader for Voice Spoofing Detection

Provides PyTorch Dataset and DataLoader classes for efficient data loading during training.

Usage:
    from data.dataset_loader import VoiceSpoofingDataset
    from torch.utils.data import DataLoader
    
    dataset = VoiceSpoofingDataset("metadata/LA_demo_train.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch_idx, (mel_specs, labels) in enumerate(dataloader):
        # mel_specs: (batch_size, 1, 128, time_steps)
        # labels: (batch_size,) with values 0 (bonafide) or 1 (spoof)
        pass
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from tqdm import tqdm

# Import preprocessing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.audio_preprocessing import AudioPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceSpoofingDataset(Dataset):
    """
    PyTorch Dataset for voice spoofing detection.
    
    Loads audio files from CSV metadata and preprocesses them on-the-fly.
    
    CSV Format:
        file_id, label, label_int, audio_path, split, ...
    """
    
    def __init__(self, csv_path, sample_rate=16000, duration=5.0, 
                 cache_preprocessed=False, verbose=True):
        """
        Initialize dataset.
        
        Args:
            csv_path (str): Path to metadata CSV file
            sample_rate (int): Sample rate for audio (default: 16000)
            duration (float): Audio duration in seconds (default: 5.0)
            cache_preprocessed (bool): Cache preprocessed spectrograms in memory
            verbose (bool): Print progress messages
        """
        self.csv_path = Path(csv_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.cache_preprocessed = cache_preprocessed
        self.verbose = verbose
        
        # Load metadata
        self.metadata = pd.read_csv(self.csv_path)
        
        if verbose:
            logger.info(f"📊 Loaded metadata: {self.csv_path}")
            logger.info(f"   Total samples: {len(self.metadata)}")
            logger.info(f"   Bonafide: {(self.metadata['label'] == 'bonafide').sum()}")
            logger.info(f"   Spoof: {(self.metadata['label'] == 'spoof').sum()}")
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            duration=duration
        )
        
        # Cache for preprocessed spectrograms
        self.cache = {} if cache_preprocessed else None
        
        # Preprocess all samples if caching
        if cache_preprocessed and verbose:
            logger.info("📍 Caching preprocessed spectrograms...")
            for idx in tqdm(range(len(self.metadata))):
                self._load_and_preprocess(idx)
    
    def _load_and_preprocess(self, idx):
        """Load and preprocess audio at index."""
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        # Get audio path
        audio_path = self.metadata.iloc[idx]['audio_path']
        
        # Preprocess
        mel_spec, metadata = self.preprocessor.preprocess(audio_path, verbose=False)
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = mel_spec
        
        return mel_spec
    
    def __len__(self):
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (mel_spectrogram, label):
                - mel_spectrogram: Tensor of shape (1, 128, time_steps)
                - label: Integer label (0 = bonafide, 1 = spoof)
        """
        # Load and preprocess audio
        mel_spec = self._load_and_preprocess(idx)
        
        # Get label
        label = int(self.metadata.iloc[idx]['label_int'])
        
        # Convert to tensor and add channel dimension
        # mel_spec shape: (128, time_steps)
        # output shape: (1, 128, time_steps)
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        label_tensor = torch.LongTensor([label])
        
        return mel_spec_tensor, label_tensor.squeeze()


class VoiceSpoofingDataLoader:
    """Wrapper around PyTorch DataLoader for convenience."""
    
    def __init__(self, csv_path, batch_size=32, shuffle=True, 
                 num_workers=0, sample_rate=16000, duration=5.0,
                 cache_preprocessed=False):
        """
        Initialize data loader.
        
        Args:
            csv_path (str): Path to metadata CSV
            batch_size (int): Batch size
            shuffle (bool): Shuffle data
            num_workers (int): Number of worker processes (0 = main process)
            sample_rate (int): Sample rate
            duration (float): Audio duration
            cache_preprocessed (bool): Cache spectrograms in memory
        """
        self.dataset = VoiceSpoofingDataset(
            csv_path,
            sample_rate=sample_rate,
            duration=duration,
            cache_preprocessed=cache_preprocessed,
            verbose=True
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"🔄 Created DataLoader:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Shuffle: {shuffle}")
        logger.info(f"   Total batches: {len(self.dataloader)}")
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.dataloader)


def create_loaders(dataset_path="dataset/LA_demo", metadata_dir="metadata",
                   batch_size=32, num_workers=0):
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_path (str): Path to dataset directory
        metadata_dir (str): Path to metadata CSV directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' data loaders
    """
    
    # Determine dataset name
    dataset_name = Path(dataset_path).name
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = Path(metadata_dir) / f"{dataset_name}_{split}.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            continue
        
        logger.info(f"\n📋 Creating {split.upper()} loader from {csv_path}...")
        
        loaders[split] = VoiceSpoofingDataLoader(
            str(csv_path),
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Only shuffle training set
            num_workers=num_workers,
            cache_preprocessed=(split == 'train')  # Cache training data
        )
    
    return loaders


if __name__ == "__main__":
    # Test script
    print("\n" + "="*70)
    print("Dataset Loader Test")
    print("="*70)
    
    # Create loaders
    loaders = create_loaders(
        dataset_path="dataset/LA_demo",
        metadata_dir="metadata",
        batch_size=8
    )
    
    # Test loading one batch from training set
    if 'train' in loaders:
        print("\n📌 Loading sample batch from training set...")
        for batch_idx, (mel_specs, labels) in enumerate(loaders['train']):
            print(f"\nBatch {batch_idx}:")
            print(f"   Mel-spectrograms shape: {mel_specs.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels: {labels.tolist()}")
            print(f"   Label meaning: {['REAL' if l == 0 else 'FAKE' for l in labels.tolist()]}")
            
            if batch_idx == 0:
                break
    
    print("\n✅ Data loader test successful!")
