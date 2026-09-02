"""Quick training test"""
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset_loader import VoiceSpoofingDataLoader
from models.aasist import AASIST
import logging

logging.basicConfig(level=logging.WARNING)

print("\n" + "="*70)
print("AASIST Training Pipeline Test")
print("="*70)

# Create data loaders
print("\n📊 Loading data...")
train_loader = VoiceSpoofingDataLoader(
    'metadata/LA_demo_train.csv',
    batch_size=8,
    cache_preprocessed=False,
    shuffle=False
)
print(f"✓ Loaded {len(train_loader.dataset)} training samples")

# Create model
print("\n🤖 Creating model...")
model = AASIST(num_classes=2)
device = 'cpu'
model.to(device)
print(f"✓ Model with {sum(p.numel() for p in model.parameters()):,} parameters")

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train for 1 epoch
print("\n🎯 Training epoch...")
model.train()
total_loss = 0
num_batches = 0

for batch_idx, (mel_specs, labels) in enumerate(train_loader):
    mel_specs = mel_specs.to(device)
    labels = labels.to(device)
    
    # Forward
    logits = model(mel_specs)
    loss = criterion(logits, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    num_batches += 1
    
    if (batch_idx + 1) % 4 == 0:
        print(f"  Batch {batch_idx+1}/8 - Loss: {loss.item():.4f}")

avg_loss = total_loss / num_batches
print(f"\n✓ Epoch complete - Average Loss: {avg_loss:.4f}")

print("\n✅ Training pipeline test PASSED!")
print("\nReady to run: python train.py --dataset dataset/LA_demo --epochs 10")
