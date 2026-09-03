"""
Main Training Script for Voice Spoofing Detection

Trains the AASIST model on the ASVspoof 2019 LA dataset.

Usage:
    python train.py --dataset dataset/LA_demo --metadata metadata/LA_demo_train.csv
    python train.py --dataset dataset/LA --metadata metadata/LA_train.csv --epochs 100
    
Args:
    --dataset: Path to dataset directory
    --metadata-dir: Directory containing CSV metadata files
    --dataset-name: Prefix of CSV files (e.g., "LA_demo")
    --epochs: Number of training epochs
    --batch-size: Batch size
    --lr: Learning rate
    --device: Device to use (cuda or cpu)
    --checkpoint-dir: Directory to save checkpoints
    --log-dir: Directory to save logs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
import json
from tqdm import tqdm
import numpy as np

# Import project modules
from models.aasist import AASIST
from data.dataset_loader import VoiceSpoofingDataLoader, create_loaders
from utils.training_utils import (
    MetricTracker, compute_metrics, CheckpointManager, 
    plot_training_curves, plot_confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training loop manager for voice spoofing detection"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None,
                 device='cpu', lr=1e-3, checkpoint_dir='checkpoints'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader (optional)
            device: Device to train on
            lr: Learning rate
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,patience=5)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Metrics tracking
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        logger.info(f"✓ Trainer initialized on device: {device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (mel_specs, labels) in enumerate(progress_bar):
            mel_specs = mel_specs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(mel_specs)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        epoch_loss /= len(self.train_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = epoch_loss
        
        self.train_metrics.update(
            loss=epoch_loss,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            auc_roc=metrics['auc_roc']
        )
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        for mel_specs, labels in progress_bar:
            mel_specs = mel_specs.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(mel_specs)
            loss = self.criterion(logits, labels)
            
            epoch_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        epoch_loss /= len(self.val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = epoch_loss
        
        self.val_metrics.update(
            loss=epoch_loss,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            auc_roc=metrics['auc_roc']
        )
        
        return metrics
    
    @torch.no_grad()
    def test(self):
        """Evaluate on test set"""
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return None
        
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.test_loader, desc="Testing", leave=False)
        
        for mel_specs, labels in progress_bar:
            mel_specs = mel_specs.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(mel_specs)
            loss = self.criterion(logits, labels)
            
            epoch_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        epoch_loss /= len(self.test_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = epoch_loss
        
        # Save confusion matrix
        pred_classes = np.argmax(all_preds, axis=1)
        plot_confusion_matrix(all_labels, pred_classes, 'confusion_matrix.png')
        
        return metrics, all_labels, all_preds
    
    def train(self, num_epochs, save_interval=5):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
        """
        logger.info(f"\n🎯 Starting training for {num_epochs} epochs...")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Train batches: {len(self.train_loader)}")
        logger.info(f"   Val batches: {len(self.val_loader)}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*70}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['loss']:.4f} | "
                       f"Acc: {train_metrics['accuracy']:.4f} | "
                       f"F1: {train_metrics['f1']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f} | "
                       f"Acc: {val_metrics['accuracy']:.4f} | "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1'])
            
            # Save checkpoint
            if epoch % save_interval == 0:
                checkpoint_data = {
                    'train': train_metrics,
                    'val': val_metrics,
                    'epoch': epoch
                }
                is_best = self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    checkpoint_data,
                    metric_name='val_f1'
                )
        
        logger.info(f"\n✅ Training completed!")
        
        # Plot training curves
        plot_training_curves(
            self.train_metrics.to_dict(),
            self.val_metrics.to_dict(),
            'training_curves.png'
        )
        
        return self.train_metrics, self.val_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train AASIST model for voice spoofing detection"
    )
    
    parser.add_argument("--dataset", type=str, default="dataset/LA_demo",
                       help="Path to dataset directory")
    parser.add_argument("--metadata-dir", type=str, default="metadata",
                       help="Directory containing CSV metadata files")
    parser.add_argument("--dataset-name", type=str, default="LA_demo",
                       help="Dataset name prefix for CSV files")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"\n{'='*70}")
    logger.info("AASIST Model Training")
    logger.info(f"{'='*70}\n")
    
    # Create data loaders
    logger.info("📊 Loading dataset...")
    loaders = create_loaders(
        dataset_path=args.dataset,
        metadata_dir=args.metadata_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if 'train' not in loaders:
        logger.error("Training data not found!")
        return
    
    # Create model
    logger.info("\n🤖 Creating AASIST model...")
    model = AASIST(num_classes=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'].dataloader,
        val_loader=loaders['val'].dataloader if 'val' in loaders else loaders['train'].dataloader,
        test_loader=loaders['test'].dataloader if 'test' in loaders else None,
        device=args.device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    train_metrics, val_metrics = trainer.train(args.epochs, save_interval=5)
    
    # Test
    logger.info("\n📊 Testing on test set...")
    if 'test' in loaders:
        test_results = trainer.test()
        if test_results:
            test_metrics, test_labels, test_preds = test_results
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"Test Acc: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1: {test_metrics['f1']:.4f}")
            logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    logger.info("\n✅ Training pipeline completed!")


if __name__ == "__main__":
    main()
