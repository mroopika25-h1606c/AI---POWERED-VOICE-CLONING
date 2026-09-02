"""Quick test of data loader"""
from data.dataset_loader import VoiceSpoofingDataLoader

print('\nCreating train loader...')
train_loader = VoiceSpoofingDataLoader('metadata/LA_demo_train.csv', batch_size=8, cache_preprocessed=False)

print(f'\n✓ DataLoader created successfully!')
print(f'  Total samples: {len(train_loader.dataset)}')
print(f'  Batch size: 8')
print(f'  Total batches: {len(train_loader)}')

# Load one batch
print(f'\nLoading first batch...')
for mel_specs, labels in train_loader:
    print(f'✓ Batch loaded!')
    print(f'  Mel-spectrograms shape: {mel_specs.shape}')
    print(f'  Labels shape: {labels.shape}')
    print(f'  Labels: {labels.tolist()}')
    label_names = ['REAL' if l == 0 else 'FAKE' for l in labels.tolist()]
    print(f'  Label names: {label_names}')
    break

print(f'\n✅ Data loader test PASSED!')
