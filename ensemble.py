"""
Ensemble Script for Chromatin State Prediction
Combines predictions from multiple models to improve accuracy.

Usage:
1. First, save PROBABILITIES (not just predictions) from each model
2. Run this script to combine them
"""

import numpy as np
import zipfile
from collections import Counter

# ============================================
# OPTION 1: Ensemble from Probability Files
# (Best method - if you saved probabilities)
# ============================================

def ensemble_from_probabilities(prob_files, weights=None):
    """
    Ensemble by averaging probabilities from multiple models.
    
    Args:
        prob_files: List of .npy files containing probability arrays
        weights: Optional weights for each model (must sum to 1)
    
    Returns:
        predictions: Final predictions (1-indexed)
    """
    all_probs = []
    
    for f in prob_files:
        probs = np.load(f)
        print(f"Loaded {f}: shape {probs.shape}")
        all_probs.append(probs)
    
    if weights is None:
        weights = [1/len(all_probs)] * len(all_probs)
    
    # Weighted average
    ensemble_probs = np.zeros_like(all_probs[0])
    for probs, weight in zip(all_probs, weights):
        ensemble_probs += probs * weight
    
    # Get predictions (add 1 for 1-indexed labels)
    predictions = np.argmax(ensemble_probs, axis=1) + 1
    
    return predictions, ensemble_probs


# ============================================
# OPTION 2: Ensemble from Prediction Files
# (If you only have final predictions)
# ============================================

def load_predictions(filepath):
    """Load predictions from CSV file."""
    with open(filepath, 'r') as f:
        preds = [int(line.strip()) for line in f]
    return np.array(preds)


def ensemble_from_predictions_voting(pred_files):
    """
    Ensemble by majority voting from multiple prediction files.
    
    Args:
        pred_files: List of prediction CSV files
    
    Returns:
        predictions: Final predictions after voting
    """
    all_preds = []
    
    for f in pred_files:
        preds = load_predictions(f)
        print(f"Loaded {f}: {len(preds)} predictions")
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)  # Shape: (n_models, n_samples)
    
    # Majority voting
    final_preds = []
    for i in range(all_preds.shape[1]):
        votes = all_preds[:, i]
        most_common = Counter(votes).most_common(1)[0][0]
        final_preds.append(most_common)
    
    return np.array(final_preds)


def ensemble_from_predictions_soft(pred_files, n_classes=18):
    """
    Soft voting: Convert predictions to pseudo-probabilities.
    Better than hard voting when models have similar accuracy.
    
    Args:
        pred_files: List of prediction CSV files
        n_classes: Number of classes (18)
    
    Returns:
        predictions: Final predictions
    """
    all_preds = []
    
    for f in pred_files:
        preds = load_predictions(f)
        print(f"Loaded {f}: {len(preds)} predictions")
        all_preds.append(preds)
    
    n_samples = len(all_preds[0])
    n_models = len(all_preds)
    
    # Convert to pseudo-probabilities
    pseudo_probs = np.zeros((n_samples, n_classes))
    
    for preds in all_preds:
        for i, pred in enumerate(preds):
            pseudo_probs[i, pred - 1] += 1  # pred is 1-indexed
    
    # Normalize
    pseudo_probs /= n_models
    
    # Get final predictions
    final_preds = np.argmax(pseudo_probs, axis=1) + 1
    
    return final_preds, pseudo_probs


# ============================================
# OPTION 3: Weighted Ensemble
# (Give more weight to better models)
# ============================================

def weighted_ensemble(pred_files, accuracies):
    """
    Weight models by their validation accuracy.
    
    Args:
        pred_files: List of prediction CSV files
        accuracies: Validation accuracy of each model
    
    Returns:
        predictions: Final predictions
    """
    # Normalize accuracies to weights
    weights = np.array(accuracies)
    weights = weights / weights.sum()
    
    print(f"Model weights: {weights}")
    
    all_preds = []
    for f in pred_files:
        preds = load_predictions(f)
        all_preds.append(preds)
    
    n_samples = len(all_preds[0])
    n_classes = 18
    
    # Weighted pseudo-probabilities
    pseudo_probs = np.zeros((n_samples, n_classes))
    
    for preds, weight in zip(all_preds, weights):
        for i, pred in enumerate(preds):
            pseudo_probs[i, pred - 1] += weight
    
    final_preds = np.argmax(pseudo_probs, axis=1) + 1
    
    return final_preds


# ============================================
# SAVE PREDICTIONS
# ============================================

def save_predictions(predictions, output_file='predictions_ensemble.csv'):
    """Save predictions to CSV and create zip for submission."""
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Saved {len(predictions)} predictions to {output_file}")
    
    # Create zip
    zip_file = output_file.replace('.csv', '.zip')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file)
    
    print(f"Created {zip_file} for submission")
    
    # Print distribution
    print("\nPrediction distribution:")
    counts = Counter(predictions)
    for state in sorted(counts.keys()):
        print(f"  State {state}: {counts[state]}")


# ============================================
# MAIN - EDIT THIS SECTION
# ============================================

if __name__ == '__main__':
    
    # -----------------------------------------
    # EDIT THESE PATHS TO YOUR PREDICTION FILES
    # -----------------------------------------
    
    pred_files = [
        'predictions_transformer.csv',  # Your transformer model
        'predictions_kmer.csv',         # Your k-mer model
        # 'predictions_teammate.csv',   # Add teammate's when ready
    ]
    
    # Validation accuracies (for weighted ensemble)
    accuracies = [
        0.196,  # Transformer: 19.6%
        0.162,  # K-mer: 16.2%
        # 0.XX,  # Teammate's accuracy
    ]
    
    # -----------------------------------------
    # CHOOSE ENSEMBLE METHOD
    # -----------------------------------------
    
    print("="*50)
    print("ENSEMBLE PREDICTIONS")
    print("="*50)
    
    # Method 1: Simple voting (equal weight)
    print("\n--- Method 1: Majority Voting ---")
    try:
        preds_vote = ensemble_from_predictions_voting(pred_files)
        save_predictions(preds_vote, 'predictions_ensemble_vote.csv')
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 2: Soft voting
    print("\n--- Method 2: Soft Voting ---")
    try:
        preds_soft, _ = ensemble_from_predictions_soft(pred_files)
        save_predictions(preds_soft, 'predictions_ensemble_soft.csv')
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 3: Weighted by accuracy
    print("\n--- Method 3: Weighted Ensemble ---")
    try:
        preds_weighted = weighted_ensemble(pred_files, accuracies)
        save_predictions(preds_weighted, 'predictions_ensemble_weighted.csv')
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("Done! Submit the ensemble zip files to Codabench")
    print("="*50)
