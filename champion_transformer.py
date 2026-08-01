"""
🏆 Champion Transformer Model for Chromatin State Prediction
Run: python champion_transformer.py
"""

import os
import math
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data
    seq_length = 200
    num_classes = 18
    
    # Model Architecture (same as winning model)
    vocab_size = 4
    embed_dim = 256
    num_heads = 8
    num_transformer_layers = 8
    ff_dim = 1024
    dropout = 0.05
    
    # CNN parameters
    cnn_channels = [128, 256, 256]
    cnn_kernels = [15, 9, 5]
    
    # Training
    batch_size = 128
    epochs = 30  # Same as winning model
    learning_rate = 5e-4
    weight_decay = 0.01
    warmup_epochs = 5
    label_smoothing = 0.05
    gradient_clip = 1.0
    
    # Cross-validation
    n_folds = 5
    
    # Augmentation
    use_reverse_complement = True
    rc_prob = 0.5
    
    # Paths - UPDATE THESE!
    train_sequences_path = 'trainsequences.csv'
    train_labels_path = 'trainlabels.csv'
    test_sequences_path = 'testsequences.csv'

config = Config()

# TRY DIFFERENT SEEDS: 42, 45, 123, 2024
SEED = 2007

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Using seed: {SEED}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================
# DATA LOADING
# ============================================
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


# ============================================
# DNA ENCODING
# ============================================
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

def encode_sequence(seq):
    return np.array([NUC_TO_IDX[nuc] for nuc in seq], dtype=np.int64)

def reverse_complement(seq):
    return ''.join(COMPLEMENT[nuc] for nuc in reversed(seq))


# ============================================
# DATASET
# ============================================
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
            label = self.labels[idx] - 1  # 0-indexed
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


# ============================================
# MODEL COMPONENTS
# ============================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttentionRoPE(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cos, sin):
        x = x + self.dropout(self.attention(self.norm1(x), cos, sin))
        x = x + self.ff(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


# ============================================
# MAIN MODEL
# ============================================
class ChromatinStatePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        self.conv_layers = nn.ModuleList()
        in_channels = config.embed_dim
        for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernels):
            self.conv_layers.append(ConvBlock(in_channels, out_channels, kernel_size, config.dropout))
            in_channels = out_channels
        
        self.cnn_proj = nn.Linear(config.cnn_channels[-1], config.embed_dim)
        
        self.rope = RotaryPositionalEmbedding(
            config.embed_dim // config.num_heads,
            max_seq_len=config.seq_length
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_transformer_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        x = self.embedding(x)
        
        x = x.transpose(1, 2)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.transpose(1, 2)
        
        x = self.cnn_proj(x)
        
        cos, sin = self.rope(seq_len)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, cos, sin)
        
        x = self.final_norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        
        return logits


# ============================================
# TRAINING UTILITIES
# ============================================
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


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, config):
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


# ============================================
# TRAINING LOOP
# ============================================
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
    
    model = ChromatinStatePredictor(config).to(device)
    
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
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, config)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"*** New best model! Val Acc: {val_acc*100:.2f}% ***")
    
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, f'champion_model4_fold{fold}.pt')
    
    return model, best_val_acc


def predict_with_tta(models, test_sequences, device, batch_size=256):
    """Generate predictions with test-time augmentation."""
    test_dataset = DNADatasetTTA(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    all_probs = []
    
    for model_idx, model in enumerate(models):
        print(f"Predicting with model {model_idx + 1}/{len(models)}...")
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
    
    return predictions, ensemble_probs


# ============================================
# MAIN
# ============================================
def main():
    print("Loading data...")
    train_sequences, train_labels = load_data(config.train_sequences_path, config.train_labels_path)
    test_sequences, _ = load_data(config.test_sequences_path)
    
    # Test model
    print("\nTesting model architecture...")
    model = ChromatinStatePredictor(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    test_input = torch.randint(0, 4, (2, 200)).to(device)
    test_output = model(test_input)
    print(f"Input: {test_input.shape} -> Output: {test_output.shape}")
    del model
    
    # Train with K-Fold
    print("\nStarting K-Fold training...")
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    
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
    print("\nGenerating predictions...")
    predictions, probabilities = predict_with_tta(fold_models, test_sequences, device)
    print(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    output_file = 'predictions.csv'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Saved to {output_file}")
    
    # Save probabilities for ensembling
    np.save('probabilities.npy', probabilities)
    print("Saved probabilities.npy")
    
    # Create zip
    with zipfile.ZipFile('submission_champion.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file)
    print("Created submission_champion.zip")
    
    # Print distribution
    print("\nPrediction distribution:")
    counts = Counter(predictions)
    for state in sorted(counts.keys()):
        print(f"  State {state}: {counts[state]}")


if __name__ == '__main__':
    main()
