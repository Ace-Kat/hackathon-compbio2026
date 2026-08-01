"""
ULTIMATE K-MER MODEL FOR CHROMATIN STATE PREDICTION
====================================================
Based on research from:
- Gapped k-mers (gkm-SVM) - better than regular k-mers
- CpG island features - critical for promoter detection
- Positional k-mers - where patterns occur matters
- DNA structural features - flexibility, bendability
- Transcription factor motifs - TATA, CCAAT, GC-box, etc.

Run: python ultimate_kmer_model.py
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

import zipfile
import pickle

# ============================================
# CONFIGURATION
# ============================================
TRAIN_SEQ_PATH = 'trainsequences.csv'
TRAIN_LABELS_PATH = 'trainlabels.csv'
TEST_SEQ_PATH = 'testsequences.csv'

# ============================================
# DATA LOADING
# ============================================
def load_data():
    print("Loading data...")
    with open(TRAIN_SEQ_PATH, 'r') as f:
        train_sequences = [line.strip().upper() for line in f]
    with open(TRAIN_LABELS_PATH, 'r') as f:
        train_labels = [int(line.strip()) for line in f]
    with open(TEST_SEQ_PATH, 'r') as f:
        test_sequences = [line.strip().upper() for line in f]
    
    print(f"Train: {len(train_sequences)} sequences")
    print(f"Test: {len(test_sequences)} sequences")
    return train_sequences, train_labels, test_sequences

# ============================================
# FEATURE 1: REGULAR K-MERS
# ============================================
def get_all_kmers(k):
    """Generate all possible k-mers."""
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in product(bases, repeat=k)]

def count_kmers(sequence, k):
    """Count k-mer frequencies."""
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

def get_kmer_features(sequence, k_values=[3, 4, 5]):
    """Extract normalized k-mer frequencies."""
    features = []
    for k in k_values:
        all_kmers = get_all_kmers(k)
        kmer_counts = count_kmers(sequence, k)
        total = len(sequence) - k + 1
        for kmer in all_kmers:
            features.append(kmer_counts.get(kmer, 0) / total)
    return features

# ============================================
# FEATURE 2: GAPPED K-MERS (Research-backed!)
# Better for regulatory sequence prediction
# ============================================
def get_gapped_kmer_features(sequence, k=5, gap_positions=[2]):
    """
    Gapped k-mers: k-mers with wildcards at certain positions.
    Example: k=5, gap at position 2: "AC*GT" matches "ACAGT", "ACCGT", "ACGGT", "ACTGT"
    
    This is more robust than regular k-mers for longer patterns.
    """
    features = {}
    bases = 'ACGT'
    
    for gap_pos in gap_positions:
        # Generate all gapped k-mers (using 'N' as wildcard in name)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            # Replace the gap position with wildcard
            gapped = kmer[:gap_pos] + 'N' + kmer[gap_pos+1:]
            features[gapped] = features.get(gapped, 0) + 1
    
    # Normalize
    total = sum(features.values()) if features else 1
    return {k: v/total for k, v in features.items()}

def extract_gapped_kmer_vector(sequence, k=5, gap_positions=[2]):
    """Convert gapped k-mers to fixed-length feature vector."""
    gkmer_counts = get_gapped_kmer_features(sequence, k, gap_positions)
    
    # Create all possible gapped k-mers
    bases = 'ACGT'
    all_gkmers = []
    for gap_pos in gap_positions:
        for combo in product(bases, repeat=k-1):
            gkmer = ''.join(combo[:gap_pos]) + 'N' + ''.join(combo[gap_pos:])
            all_gkmers.append(gkmer)
    
    return [gkmer_counts.get(gk, 0) for gk in all_gkmers]

# ============================================
# FEATURE 3: POSITIONAL K-MERS
# Where patterns occur matters!
# ============================================
def get_positional_kmer_features(sequence, k=4, n_bins=4):
    """
    Count k-mers in different regions of the sequence.
    TATA at start vs TATA at end have different meanings!
    """
    features = []
    seq_len = len(sequence)
    bin_size = seq_len // n_bins
    
    all_kmers = get_all_kmers(k)
    
    for bin_idx in range(n_bins):
        start = bin_idx * bin_size
        end = start + bin_size if bin_idx < n_bins - 1 else seq_len
        region = sequence[start:end]
        
        kmer_counts = count_kmers(region, k)
        total = len(region) - k + 1 if len(region) >= k else 1
        
        for kmer in all_kmers:
            features.append(kmer_counts.get(kmer, 0) / total)
    
    return features

# ============================================
# FEATURE 4: CpG ISLAND FEATURES
# Critical for promoter detection!
# ============================================
def get_cpg_features(sequence):
    """
    CpG islands are regions with high CpG density.
    Associated with promoters and active regulatory regions.
    
    Traditional CpG island definition:
    - Length > 200bp
    - GC content > 50%
    - Observed/Expected CpG ratio > 0.6
    """
    length = len(sequence)
    
    # Count nucleotides
    c_count = sequence.count('C')
    g_count = sequence.count('G')
    a_count = sequence.count('A')
    t_count = sequence.count('T')
    
    # CpG dinucleotide count
    cpg_count = sequence.count('CG')
    
    # GC content
    gc_content = (c_count + g_count) / length
    
    # CpG observed/expected ratio
    # Expected CpG = (C_freq * G_freq) * length
    expected_cpg = (c_count / length) * (g_count / length) * (length - 1)
    cpg_oe_ratio = cpg_count / expected_cpg if expected_cpg > 0 else 0
    
    # CpG density
    cpg_density = cpg_count / (length - 1)
    
    # TpG and CpA (products of CpG deamination - indicates methylation history)
    tpg_count = sequence.count('TG')
    cpa_count = sequence.count('CA')
    
    # CpG suppression index
    cpg_suppression = cpg_count / ((tpg_count + cpa_count) / 2 + 1)
    
    # Sliding window CpG analysis
    window_size = 50
    cpg_windows = []
    for i in range(0, length - window_size, window_size // 2):
        window = sequence[i:i+window_size]
        cpg_windows.append(window.count('CG'))
    
    cpg_max = max(cpg_windows) if cpg_windows else 0
    cpg_min = min(cpg_windows) if cpg_windows else 0
    cpg_std = np.std(cpg_windows) if cpg_windows else 0
    
    return [
        gc_content,
        cpg_oe_ratio,
        cpg_density,
        cpg_suppression,
        tpg_count / (length - 1),
        cpa_count / (length - 1),
        cpg_max,
        cpg_min,
        cpg_std,
    ]

# ============================================
# FEATURE 5: TRANSCRIPTION FACTOR MOTIFS
# Known regulatory sequence patterns
# ============================================
# Important motifs for chromatin states
TF_MOTIFS = {
    # Core promoter elements
    'TATA_box': ['TATAAA', 'TATAWA', 'TATAWAR'],  # W = A or T, R = A or G
    'TATA_like': ['TATAA', 'TATAT', 'TATTA'],
    'Initiator': ['YYANWYY'],  # Y = C or T, N = any, W = A or T
    'GC_box': ['GGGCGG', 'CCGCCC'],
    'CCAAT_box': ['CCAAT', 'ATTGG'],
    'DPE': ['RGWYV'],  # Downstream promoter element
    
    # CpG-related
    'CpG_cluster': ['CGCG', 'GCGC', 'CGCGCG'],
    
    # Enhancer motifs
    'AP1': ['TGACTCA', 'TGAGTCA'],
    'CREB': ['TGACGTCA'],
    'NFkB': ['GGGACTTTCC', 'GGGRNNYYCC'],
    'SP1': ['GGGCGG', 'GGCGGG'],
    
    # Repressor motifs  
    'Polycomb': ['GCGC'],  # Associated with repressed chromatin
    
    # CTCF (insulator)
    'CTCF': ['CCGCGNGGNGGCAG'],
}

def count_motif_matches(sequence, motif):
    """Count occurrences of a motif (with IUPAC wildcards)."""
    # Expand IUPAC codes
    iupac = {
        'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
        'R': 'AG', 'Y': 'CT', 'W': 'AT', 'S': 'GC',
        'M': 'AC', 'K': 'GT', 'B': 'CGT', 'D': 'AGT',
        'H': 'ACT', 'V': 'ACG', 'N': 'ACGT'
    }
    
    count = 0
    motif_len = len(motif)
    
    for i in range(len(sequence) - motif_len + 1):
        subseq = sequence[i:i+motif_len]
        match = True
        for j, (s, m) in enumerate(zip(subseq, motif)):
            if s not in iupac.get(m, m):
                match = False
                break
        if match:
            count += 1
    
    return count

def get_motif_features(sequence):
    """Extract transcription factor motif counts."""
    features = []
    length = len(sequence)
    
    for motif_name, patterns in TF_MOTIFS.items():
        total_count = 0
        for pattern in patterns:
            total_count += count_motif_matches(sequence, pattern)
        features.append(total_count / length)  # Normalize by length
    
    return features

# ============================================
# FEATURE 6: DINUCLEOTIDE BIAS
# Important for chromatin structure
# ============================================
def get_dinucleotide_features(sequence):
    """
    Dinucleotide frequencies capture DNA structural properties.
    Certain dinucleotides affect DNA flexibility and protein binding.
    """
    dinucs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
              'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    
    length = len(sequence) - 1
    counts = Counter(sequence[i:i+2] for i in range(length))
    
    features = [counts.get(d, 0) / length for d in dinucs]
    
    # Dinucleotide transition probabilities
    # P(XY) / P(X) - probability of Y given X
    nuc_counts = Counter(sequence[:-1])
    total = sum(nuc_counts.values())
    
    transitions = []
    for first in 'ACGT':
        first_count = nuc_counts.get(first, 0)
        for second in 'ACGT':
            dinuc = first + second
            dinuc_count = counts.get(dinuc, 0)
            if first_count > 0:
                transitions.append(dinuc_count / first_count)
            else:
                transitions.append(0)
    
    return features + transitions

# ============================================
# FEATURE 7: DNA STRUCTURAL FEATURES
# Bendability, flexibility affect protein binding
# ============================================
# Dinucleotide structural parameters (from research)
BENDABILITY = {
    'AA': 0.026, 'AC': 0.037, 'AG': 0.014, 'AT': 0.032,
    'CA': 0.055, 'CC': 0.042, 'CG': 0.027, 'CT': 0.014,
    'GA': 0.045, 'GC': 0.044, 'GG': 0.042, 'GT': 0.037,
    'TA': 0.068, 'TC': 0.045, 'TG': 0.055, 'TT': 0.026,
}

PROPELLER_TWIST = {
    'AA': -18.66, 'AC': -13.10, 'AG': -14.00, 'AT': -15.01,
    'CA': -9.45, 'CC': -8.11, 'CG': -10.03, 'CT': -14.00,
    'GA': -13.48, 'GC': -11.08, 'GG': -8.11, 'GT': -13.10,
    'TA': -11.85, 'TC': -13.48, 'TG': -9.45, 'TT': -18.66,
}

def get_structural_features(sequence):
    """Calculate DNA structural properties."""
    features = []
    length = len(sequence) - 1
    
    # Bendability
    bend_values = [BENDABILITY.get(sequence[i:i+2], 0.04) for i in range(length)]
    features.extend([
        np.mean(bend_values),
        np.std(bend_values),
        np.max(bend_values),
        np.min(bend_values),
    ])
    
    # Propeller twist
    twist_values = [PROPELLER_TWIST.get(sequence[i:i+2], -12) for i in range(length)]
    features.extend([
        np.mean(twist_values),
        np.std(twist_values),
        np.max(twist_values),
        np.min(twist_values),
    ])
    
    # Flexibility (AA, TA, AT are most flexible)
    flexible_dinucs = ['AA', 'TT', 'AT', 'TA']
    flex_count = sum(1 for i in range(length) if sequence[i:i+2] in flexible_dinucs)
    features.append(flex_count / length)
    
    # Rigidity (GC, CG are rigid)
    rigid_dinucs = ['GC', 'CG', 'GG', 'CC']
    rigid_count = sum(1 for i in range(length) if sequence[i:i+2] in rigid_dinucs)
    features.append(rigid_count / length)
    
    return features

# ============================================
# FEATURE 8: SEQUENCE COMPLEXITY
# Low complexity = heterochromatin/quiescent
# ============================================
def get_complexity_features(sequence):
    """Measure sequence complexity."""
    length = len(sequence)
    
    # Shannon entropy
    counts = Counter(sequence)
    probs = [c/length for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Linguistic complexity (unique k-mers / possible k-mers)
    complexities = []
    for k in [2, 3, 4]:
        unique_kmers = len(set(sequence[i:i+k] for i in range(length - k + 1)))
        possible_kmers = min(4**k, length - k + 1)
        complexities.append(unique_kmers / possible_kmers)
    
    # Repeat density
    repeat_count = sum(1 for i in range(length-1) if sequence[i] == sequence[i+1])
    
    # Longest homopolymer run
    max_run = 1
    current_run = 1
    for i in range(1, length):
        if sequence[i] == sequence[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    # Purine/Pyrimidine balance
    purines = sequence.count('A') + sequence.count('G')
    pyrimidines = sequence.count('C') + sequence.count('T')
    pur_pyr_ratio = purines / (pyrimidines + 1)
    
    return [entropy] + complexities + [repeat_count / length, max_run / length, pur_pyr_ratio]

# ============================================
# FEATURE 9: NUCLEOSOME POSITIONING
# Affects chromatin accessibility
# ============================================
def get_nucleosome_features(sequence):
    """
    Features related to nucleosome positioning.
    Certain dinucleotide patterns favor or disfavor nucleosome formation.
    """
    length = len(sequence)
    
    # AA/TT dinucleotides with ~10bp periodicity favor nucleosome positioning
    # Count AA/TT in different phase positions
    aa_tt_counts = []
    for phase in range(10):
        count = 0
        for i in range(phase, length - 1, 10):
            if sequence[i:i+2] in ['AA', 'TT']:
                count += 1
        aa_tt_counts.append(count)
    
    # Nucleosome disfavoring sequences (poly-A/T tracts)
    poly_a = max(len(s) for s in sequence.split('A') if s == '') if 'AA' in sequence else 0
    poly_t = max(len(s) for s in sequence.split('T') if s == '') if 'TT' in sequence else 0
    
    # GC content affects nucleosome stability
    gc = (sequence.count('G') + sequence.count('C')) / length
    
    return [np.mean(aa_tt_counts), np.std(aa_tt_counts), np.max(aa_tt_counts), gc]

# ============================================
# COMBINE ALL FEATURES
# ============================================
def extract_all_features(sequence):
    """Extract all feature types."""
    features = []
    
    # 1. Regular k-mers (k=3,4): 64 + 256 = 320 features
    features.extend(get_kmer_features(sequence, k_values=[3, 4]))
    
    # 2. Positional k-mers (k=3, 4 bins): 64 * 4 = 256 features
    features.extend(get_positional_kmer_features(sequence, k=3, n_bins=4))
    
    # 3. CpG features: 9 features
    features.extend(get_cpg_features(sequence))
    
    # 4. TF motif features: ~15 features
    features.extend(get_motif_features(sequence))
    
    # 5. Dinucleotide features: 16 + 16 = 32 features
    features.extend(get_dinucleotide_features(sequence))
    
    # 6. Structural features: 10 features
    features.extend(get_structural_features(sequence))
    
    # 7. Complexity features: 7 features
    features.extend(get_complexity_features(sequence))
    
    # 8. Nucleosome features: 4 features
    features.extend(get_nucleosome_features(sequence))
    
    return features

# ============================================
# TRAINING
# ============================================
def train_model(X_train, y_train, X_val, y_val):
    """Train multiple models and pick the best."""
    models = {}
    
    # LightGBM (usually best for this type of data)
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    lgb_acc = lgb_model.score(X_val, y_val)
    models['lgb'] = (lgb_model, lgb_acc)
    print(f"LightGBM Val Accuracy: {lgb_acc*100:.2f}%")
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    xgb_acc = xgb_model.score(X_val, y_val)
    models['xgb'] = (xgb_model, xgb_acc)
    print(f"XGBoost Val Accuracy: {xgb_acc*100:.2f}%")
    
    # Pick best model
    best_name = max(models, key=lambda x: models[x][1])
    print(f"\nBest model: {best_name} ({models[best_name][1]*100:.2f}%)")
    
    return models[best_name][0], models

# ============================================
# MAIN
# ============================================
def main():
    # Load data
    train_sequences, train_labels, test_sequences = load_data()
    
    # Convert labels to 0-indexed
    y = np.array(train_labels) - 1
    
    # Extract features
    print("\nExtracting features for training data...")
    X = np.array([extract_all_features(seq) for seq in tqdm(train_sequences)])
    print(f"Feature matrix shape: {X.shape}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    best_model, all_models = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Retrain on full data
    print("\n" + "="*50)
    print("RETRAINING ON FULL DATA")
    print("="*50)
    X_full_scaled = scaler.fit_transform(X)
    best_model.fit(X_full_scaled, y)
    
    # Extract test features
    print("\nExtracting features for test data...")
    X_test = np.array([extract_all_features(seq) for seq in tqdm(test_sequences)])
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    print("\nGenerating predictions...")
    predictions = best_model.predict(X_test_scaled) + 1  # Back to 1-indexed
    
    # Save
    output_file = 'predictions_ultimate.csv'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Saved {len(predictions)} predictions to {output_file}")
    
    # Create zip
    with zipfile.ZipFile('submission_ultimate.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file)
    print("Created submission_ultimate.zip")
    
    # Save model and scaler
    with open('ultimate_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler}, f)
    print("Saved model to ultimate_model.pkl")
    
    # Print distribution
    print("\nPrediction distribution:")
    counts = Counter(predictions)
    for state in sorted(counts.keys()):
        print(f"  State {state}: {counts[state]}")

if __name__ == '__main__':
    main()
