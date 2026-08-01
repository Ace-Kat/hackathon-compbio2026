"""
Chromatin State Prediction - Improved CNN Model
Standalone Python script for local training

SETUP INSTRUCTIONS:
===================
1. Install Python 3.10+ from https://www.python.org/downloads/

2. Open terminal/command prompt and run:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy pandas scikit-learn tqdm matplotlib

3. Place your data files in the same folder as this script:
   - trainsequences.csv
   - trainlabels.csv
   - testsequences.csv

4. Run the script:
   python train_cnn.py

"""

import os
import math
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIGURATION - EDIT THESE PATHS!
# ============================================================
class Config:
    # Data paths - UPDATE THESE!
    train_sequences_path = 'trainsequences.csv'
    train_labels_path = 'trainlabels.csv'
    test_sequences_path = 'testsequences.csv'
    
    # Data
    seq_length = 200
    num_classes = 18
    
    # Training
    batch_size = 128          # Reduce to 64 if you run out of GPU memory
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 0.01
    warmup_epochs = 3
    label_smoothing = 0.05
    gradient_clip = 1.0
    
    # Cross-validation
    n_folds = 5
    
    # Augmentation
    use_reverse_complement = True
    rc_prob = 0.5

config = Config()


# ============================================================
# SETUP
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# DATA LOADING
# ============================================================
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

def encode_sequence(seq):
    return np.array([NUC_TO_IDX[nuc] for nuc in seq], dtype=np.int64)

def reverse_complement(seq):
    return ''.join(COMPLEMENT[nuc] for nuc in reversed(seq))

def load_data(sequences_path, labels_path=None):
    print(f"Loading sequences from {sequences_path}...")
    with open(sequences_path, 'r') as f:
        sequences = [line.strip().upper() for line in f]
    print(f"Loaded {len(sequences)} sequences")
    
    labels = None
    if labels_path:
        print(f"Loading labels from {labels_path}...")
        with open(labels_path, 'r') as f:
            labels = [int(line.strip()) for line in f]
        print(f"Loaded {len(labels)} labels")
    
    return sequences, labels


class DNADataset(Dataset):
    def __init__(self, sequences, labels=None, augment=False, rc_prob=0.5):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.rc_prob = rc_prob
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        if self.augment and random.random() < self.rc_prob:
            seq = reverse_complement(seq)
        
        encoded = encode_sequence(seq)
        
        if self.labels is not None:
            label = self.labels[idx] - 1  # Convert to 0-indexed
            return torch.tensor(encoded), torch.tensor(label)
        return torch.tensor(encoded)


class DNADatasetTTA(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_rc = reverse_complement(seq)
        return torch.tensor(encode_sequence(seq)), torch.tensor(encode_sequence(seq_rc))


# ============================================================
# MODEL
# ============================================================
class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = F.gelu(x)
        return x


class ImprovedCNNModel(nn.Module):
    """Deeper CNN model with residual connections."""
    
    def __init__(self, num_classes=18, dropout=0.1):
        super().__init__()
        
        # Embedding: 4 nucleotides -> 128 dimensions
        self.embedding = nn.Embedding(4, 128)
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks at 256 channels
        self.res_blocks_256 = nn.Sequential(
            ResidualBlock(256, kernel_size=9, dropout=dropout),
            ResidualBlock(256, kernel_size=9, dropout=dropout),
            ResidualBlock(256, kernel_size=7, dropout=dropout),
        )
        
        # Expand to 512 channels
        self.expand_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        
        # Residual blocks at 512 channels
        self.res_blocks_512 = nn.Sequential(
            ResidualBlock(512, kernel_size=5, dropout=dropout),
            ResidualBlock(512, kernel_size=5, dropout=dropout),
            ResidualBlock(512, kernel_size=3, dropout=dropout),
        )
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        
        x = self.initial_conv(x)
        x = self.res_blocks_256(x)
        x = self.expand_conv(x)
        x = self.res_blocks_512(x)
        
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = self.classifier(x)
        return x


# ============================================================
# TRAINING UTILITIES
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_preds)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        loss = (-smooth_target * log_preds).sum(dim=-1).mean()
        return loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(sequences)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc='Validating'):
            sequences, labels = sequences.to(device), labels.to(device)
            
            with autocast():
                outputs = model(sequences)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), correct / total


# ============================================================
# TRAINING LOOP
# ============================================================
def train_fold(fold, train_idx, val_idx, train_sequences, train_labels, config, device):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}/{config.n_folds}")
    print(f"{'='*50}")
    
    train_seqs = [train_sequences[i] for i in train_idx]
    train_labs = [train_labels[i] for i in train_idx]
    val_seqs = [train_sequences[i] for i in val_idx]
    val_labs = [train_labels[i] for i in val_idx]
    
    train_dataset = DNADataset(train_seqs, train_labs, augment=config.use_reverse_complement, rc_prob=config.rc_prob)
    val_dataset = DNADataset(val_seqs, val_labs, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    
    model = ImprovedCNNModel(num_classes=config.num_classes).to(device)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    scaler = GradScaler()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"*** New best model! Val Acc: {val_acc*100:.2f}% ***")
            torch.save(best_model_state, f'cnn_model_fold{fold}.pt')
    
    model.load_state_dict(best_model_state)
    return model, best_val_acc


def predict_with_tta(models, test_sequences, device, batch_size=256):
    """Ensemble prediction with test-time augmentation."""
    test_dataset = DNADatasetTTA(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    all_probs = []
    
    for model in models:
        model.eval()
        model_probs = []
        
        with torch.no_grad():
            for seq_orig, seq_rc in tqdm(test_loader, desc='Predicting'):
                seq_orig = seq_orig.to(device)
                seq_rc = seq_rc.to(device)
                
                with autocast():
                    logits_orig = model(seq_orig)
                    probs_orig = F.softmax(logits_orig, dim=-1)
                    
                    logits_rc = model(seq_rc)
                    probs_rc = F.softmax(logits_rc, dim=-1)
                
                probs = (probs_orig + probs_rc) / 2
                model_probs.append(probs.cpu().numpy())
        
        model_probs = np.concatenate(model_probs, axis=0)
        all_probs.append(model_probs)
    
    ensemble_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(ensemble_probs, axis=1) + 1  # Back to 1-indexed
    
    return predictions


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    train_sequences, train_labels = load_data(config.train_sequences_path, config.train_labels_path)
    test_sequences, _ = load_data(config.test_sequences_path)
    
    # Train with K-Fold
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    
    fold_models = []
    fold_accuracies = []
    
    labels_array = np.array(train_labels)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_sequences, labels_array)):
        model, val_acc = train_fold(
            fold, train_idx, val_idx,
            train_sequences, train_labels,
            config, device
        )
        fold_models.append(model)
        fold_accuracies.append(val_acc)
    
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    for fold, acc in enumerate(fold_accuracies):
        print(f"Fold {fold + 1}: {acc*100:.2f}%")
    print(f"Mean: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    
    # Generate predictions
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    predictions = predict_with_tta(fold_models, test_sequences, device)
    
    # Save predictions
    output_file = 'predictions.csv'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"\nPredictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")
    
    # Create zip
    import zipfile
    with zipfile.ZipFile('submission_cnn.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file)
    
    print("Created submission_cnn.zip - upload this to Codabench!")
