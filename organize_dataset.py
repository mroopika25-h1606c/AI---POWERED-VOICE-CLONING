"""
Dataset Organization & Processing Script

Organizes ASVspoof 2019 LA dataset into train/val/test splits.
Creates CSV metadata files for easy data loading.

Usage:
    python organize_dataset.py --dataset dataset/LA_demo
    python organize_dataset.py --dataset dataset/LA  (for full dataset)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organize ASVspoof dataset and create metadata CSV files."""
    
    def __init__(self, dataset_path, output_path="metadata"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Output path: {self.output_path}")
    
    def parse_protocol_file(self, protocol_path):
        """
        Parse protocol file to get audio file labels.
        
        Protocol format (ASVspoof 2019):
        file_id - - - label
        
        Examples:
        LA_0001_0000000 - - - bonafide
        LA_0002_0000000 - - A02 spoof
        """
        data = []
        
        if not protocol_path.exists():
            logger.warning(f"Protocol file not found: {protocol_path}")
            return []
        
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    file_id = parts[0]
                    label = parts[-1]  # Last column is label (bonafide/spoof)
                    
                    data.append({
                        'file_id': file_id,
                        'label': label,
                        'label_int': 0 if label == 'bonafide' else 1
                    })
        
        return data
    
    def create_metadata_csv(self, dataset_name="LA"):
        """
        Create metadata CSV files for train/test sets.
        
        Returns:
            dict: Statistics about the dataset
        """
        all_data = []
        
        # Check for train and test directories
        train_dir = self.dataset_path / "train"
        test_dir = self.dataset_path / "test"
        
        splits = {}
        
        # Process training set
        if train_dir.exists():
            logger.info(f"📁 Processing training set from {train_dir}...")
            protocol_file = train_dir / "protocol.txt"
            train_data = self.parse_protocol_file(protocol_file)
            
            for item in train_data:
                item['split'] = 'train'
                item['dataset_name'] = dataset_name
                
                # Construct full audio path
                audio_file = train_dir / f"{item['file_id']}.flac"
                if audio_file.exists():
                    item['audio_path'] = str(audio_file)
                else:
                    logger.warning(f"Audio file not found: {audio_file}")
                    continue
                
                all_data.append(item)
            
            splits['train'] = len(train_data)
            logger.info(f"✓ Training set: {len(train_data)} samples")
        
        # Process test set
        if test_dir.exists():
            logger.info(f"📁 Processing test set from {test_dir}...")
            protocol_file = test_dir / "protocol.txt"
            test_data = self.parse_protocol_file(protocol_file)
            
            for item in test_data:
                item['split'] = 'test'
                item['dataset_name'] = dataset_name
                
                # Construct full audio path
                audio_file = test_dir / f"{item['file_id']}.flac"
                if audio_file.exists():
                    item['audio_path'] = str(audio_file)
                else:
                    logger.warning(f"Audio file not found: {audio_file}")
                    continue
                
                all_data.append(item)
            
            splits['test'] = len(test_data)
            logger.info(f"✓ Test set: {len(test_data)} samples")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            logger.error("No data found! Check dataset directory structure.")
            return {}
        
        # Split into train and validation (80/20 from training set)
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        if len(train_df) > 0:
            # Split training data into train and validation
            np.random.seed(42)
            train_indices = np.random.choice(
                len(train_df),
                size=int(0.8 * len(train_df)),
                replace=False
            )
            
            actual_train = train_df.iloc[train_indices].copy()
            validation = train_df.drop(train_indices).copy()
            
            # Update split labels
            actual_train['split'] = 'train'
            validation['split'] = 'val'
            
            # Combine with test
            final_df = pd.concat([actual_train, validation, test_df], ignore_index=True)
        else:
            final_df = df
        
        # Save complete metadata CSV
        csv_path = self.output_path / f"{dataset_name}_metadata.csv"
        final_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved metadata: {csv_path}")
        
        # Save split-specific CSVs
        for split in ['train', 'val', 'test']:
            split_df = final_df[final_df['split'] == split]
            if len(split_df) > 0:
                split_csv = self.output_path / f"{dataset_name}_{split}.csv"
                split_df.to_csv(split_csv, index=False)
                logger.info(f"✓ Saved {split} CSV: {split_csv} ({len(split_df)} samples)")
        
        # Print statistics
        stats = {
            'total_samples': len(final_df),
            'train_samples': len(final_df[final_df['split'] == 'train']),
            'val_samples': len(final_df[final_df['split'] == 'val']),
            'test_samples': len(final_df[final_df['split'] == 'test']),
            'bonafide_count': (final_df['label'] == 'bonafide').sum(),
            'spoof_count': (final_df['label'] == 'spoof').sum(),
        }
        
        return stats
    
    def print_statistics(self, stats):
        """Print dataset statistics."""
        if not stats:
            return
        
        print("\n" + "="*70)
        print("Dataset Statistics")
        print("="*70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total samples:  {stats['total_samples']}")
        print(f"   Bonafide:       {stats['bonafide_count']} ({100*stats['bonafide_count']/stats['total_samples']:.1f}%)")
        print(f"   Spoof:          {stats['spoof_count']} ({100*stats['spoof_count']/stats['total_samples']:.1f}%)")
        
        print(f"\n📈 Split Distribution:")
        print(f"   Training:       {stats['train_samples']} samples")
        print(f"   Validation:     {stats['val_samples']} samples")
        print(f"   Test:           {stats['test_samples']} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Organize ASVspoof 2019 LA dataset and create metadata CSV files"
    )
    parser.add_argument("--dataset", type=str, default="dataset/LA_demo",
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="metadata",
                       help="Output directory for CSV files")
    parser.add_argument("--dataset-name", type=str, default="LA",
                       help="Dataset name prefix for CSV files")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Dataset Organization Script")
    print("="*70)
    
    organizer = DatasetOrganizer(args.dataset, args.output)
    stats = organizer.create_metadata_csv(args.dataset_name)
    organizer.print_statistics(stats)
    
    print("\n✅ Dataset organization complete!")
    print(f"📁 CSV files created in: {args.output}/")
    print("\n👉 Next: Use these CSV files to train your model")
    print("   Example: python train.py --dataset dataset/LA_demo --metadata metadata/LA_train.csv")


if __name__ == "__main__":
    main()
