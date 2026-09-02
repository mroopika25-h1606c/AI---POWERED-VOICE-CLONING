"""
Training Utilities for Voice Spoofing Detection

Provides utilities for:
- Computing metrics (accuracy, precision, recall, F1, EER)
- Tracking training progress
- Saving/loading checkpoints
- Visualization
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class MetricTracker:
    """Tracks metrics during training and validation"""
    
    def __init__(self):
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc_roc': []
        }
    
    def update(self, loss=None, accuracy=None, precision=None, 
               recall=None, f1=None, auc_roc=None):
        """Update metrics"""
        if loss is not None:
            self.metrics['loss'].append(loss)
        if accuracy is not None:
            self.metrics['accuracy'].append(accuracy)
        if precision is not None:
            self.metrics['precision'].append(precision)
        if recall is not None:
            self.metrics['recall'].append(recall)
        if f1 is not None:
            self.metrics['f1'].append(f1)
        if auc_roc is not None:
            self.metrics['auc_roc'].append(auc_roc)
    
    def get_last(self):
        """Get last values of all metrics"""
        return {k: v[-1] if v else None for k, v in self.metrics.items()}
    
    def get_average(self):
        """Get average values of all metrics"""
        return {k: np.mean(v) if v else None for k, v in self.metrics.items()}
    
    def to_dict(self):
        """Convert to dictionary"""
        return self.metrics


def compute_metrics(preds, labels):
    """
    Compute all metrics given predictions and labels.
    
    Args:
        preds: List or array of predicted probabilities or class indices
        labels: List or array of true labels (0 or 1)
        
    Returns:
        dict: Dictionary with all computed metrics
    """
    # Convert to numpy arrays
    preds = np.array(preds)
    labels = np.array(labels)
    
    # If preds are probabilities, get predicted class
    if preds.ndim == 2:
        pred_probs = preds
        pred_classes = np.argmax(preds, axis=1)
    else:
        pred_classes = preds
        pred_probs = None
    
    # Compute classification metrics
    accuracy = accuracy_score(labels, pred_classes)
    precision = precision_score(labels, pred_classes, average='binary', zero_division=0)
    recall = recall_score(labels, pred_classes, average='binary', zero_division=0)
    f1 = f1_score(labels, pred_classes, average='binary', zero_division=0)
    
    # Compute AUC-ROC if we have probabilities
    auc_roc = None
    if pred_probs is not None and pred_probs.shape[1] == 2:
        try:
            auc_roc = roc_auc_score(labels, pred_probs[:, 1])
        except:
            auc_roc = None
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc) if auc_roc else None
    }
    
    return metrics


def compute_eer(labels, pred_probs):
    """
    Compute Equal Error Rate (EER)
    
    Args:
        labels: True labels (0 or 1)
        pred_probs: Predicted probabilities for positive class
        
    Returns:
        float: Equal Error Rate
    """
    fpr, fnr, _ = roc_curve(labels, pred_probs)
    
    # Find EER (where FPR == FNR)
    eer_idx = np.argmin(np.abs(fpr + fnr - 1))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return float(eer)


class CheckpointManager:
    """Manages model checkpoint saving and loading"""
    
    def __init__(self, checkpoint_dir='checkpoints', keep_best_only=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_only = keep_best_only
        self.best_metric = None
        self.best_checkpoint = None
    
    def save(self, model, optimizer, epoch, metrics, metric_name='val_f1', is_best=False):
        """
        Save checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            metric_name: Name of metric to track best checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        logger.info(f"✓ Saved latest checkpoint: {latest_path}")
        
        # Save best checkpoint
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            
            if self.best_metric is None or metric_value > self.best_metric:
                self.best_metric = metric_value
                self.best_checkpoint = self.checkpoint_dir / 'best.pth'
                torch.save(checkpoint, self.best_checkpoint)
                logger.info(f"🌟 Saved best checkpoint: {self.best_checkpoint}")
                logger.info(f"   {metric_name}: {metric_value:.4f}")
                return True
        
        return False
    
    def load_best(self, model, optimizer=None):
        """Load best checkpoint"""
        if not self.best_checkpoint or not self.best_checkpoint.exists():
            logger.warning("No best checkpoint found")
            return None
        
        checkpoint = torch.load(self.best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"✓ Loaded best checkpoint: {self.best_checkpoint}")
        return checkpoint
    
    def load_latest(self, model, optimizer=None):
        """Load latest checkpoint"""
        latest_path = self.checkpoint_dir / 'latest.pth'
        if not latest_path.exists():
            logger.warning("No latest checkpoint found")
            return None
        
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"✓ Loaded latest checkpoint: {latest_path}")
        return checkpoint


def plot_training_curves(train_metrics, val_metrics, output_path='training_curves.png'):
    """
    Plot training and validation curves
    
    Args:
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training and Validation Metrics', fontsize=16)
    
    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
        if metric in train_metrics and train_metrics[metric]:
            ax.plot(train_metrics[metric], label='Train', marker='o')
        
        if metric in val_metrics and val_metrics[metric]:
            ax.plot(val_metrics[metric], label='Val', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    logger.info(f"✓ Saved training curves: {output_path}")
    plt.close()


def plot_confusion_matrix(labels, preds, output_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        labels: True labels
        preds: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Spoof'])
    ax.set_yticklabels(['Real', 'Spoof'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    logger.info(f"✓ Saved confusion matrix: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Training Utilities Test")
    print("="*70)
    
    # Test metric computation
    labels = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    preds = np.array([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8],
                      [0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    
    print("\n📊 Computing metrics...")
    metrics = compute_metrics(preds, labels)
    
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key:.<20} {value:.4f}")
    
    print("\n✅ Training utilities test completed!")
