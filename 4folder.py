"""
Load trained fold models and generate predictions
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import zipfile

# ============================================
# CONFIG (must match training)
# ============================================
class Config:
    seq_length = 200
    num_classes = 18
    vocab_size = 4
    embed_dim = 256
    num_heads = 8
    num_transformer_layers = 8
    ff_dim = 1024
    dropout = 0.05
    cnn_channels = [128, 256, 256]
    cnn_kernels = [15, 9, 5]

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
# MODEL (copy from training code)
# ============================================
import torch.nn as nn

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
        self.rope = RotaryPositionalEmbedding(config.embed_dim // config.num_heads, max_seq_len=config.seq_length)
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
# LOAD MODELS AND PREDICT
# ============================================
def load_models(model_paths, config, device):
    """Load trained models from .pt files."""
    models = []
    for path in model_paths:
        print(f"Loading {path}...")
        model = ChromatinStatePredictor(config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models")
    return models

def predict_with_tta(models, test_sequences, device, batch_size=256):
    """Generate predictions with test-time augmentation."""
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
    
    return predictions, ensemble_probs

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    # Load test sequences
    print("Loading test sequences...")
    with open('testsequences.csv', 'r') as f:
        test_sequences = [line.strip().upper() for line in f]
    print(f"Loaded {len(test_sequences)} test sequences")
    
    # Load your 4 trained fold models
    model_paths = [
        'transformer_model_fold0.pt',
        'transformer_model_fold1.pt',
        'transformer_model_fold2.pt',
        'transformer_model_fold3.pt',
        # 'transformer_model_fold4.pt',  # Skip if not trained
    ]
    
    models = load_models(model_paths, config, device)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions, probabilities = predict_with_tta(models, test_sequences, device)
    print(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    output_file = 'predictions.csv'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Saved to {output_file}")
    
    # Create zip
    with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file)
    print("Created submission.zip")
    
    # Print distribution
    from collections import Counter
    print("\nPrediction distribution:")
    counts = Counter(predictions)
    for state in sorted(counts.keys()):
        print(f"  State {state}: {counts[state]}")