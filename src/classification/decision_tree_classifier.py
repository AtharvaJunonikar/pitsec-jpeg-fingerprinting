#!/usr/bin/env python3
"""
PITSEC: Decision Tree Classifier

Builds a single Decision Tree classifier on combined features dataset.
Interpretable model - good for understanding how the classifier makes decisions.

Input: output/all_features_combined.csv (35 features + labels)
Output: decision_tree.pkl, visualizations, metrics

Accuracy: 97-98%
Best for: Understanding decision logic, interpretability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay
)
import seaborn as sns

# ===== CONFIGURATION =====

INPUT_CSV = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/all_features_combined_ssim_fixed.csv")
OUTPUT_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/classifier_decision_tree")
LOG_FILE = OUTPUT_DIR / "decision_tree.log"

FEATURE_COLS = [
    # SSIM (8)
    'diff_C0', 'diff_C1', 'diff_C2', 'diff_C3',
    'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3',
    # DCT (15)
    'ac_energy_y', 'dc_variance_y', 'ac_variance_y', 'zero_ratio_y', 'energy_conc_y',
    'ac_energy_cr', 'dc_variance_cr', 'ac_variance_cr', 'zero_ratio_cr', 'energy_conc_cr',
    'ac_energy_cb', 'dc_variance_cb', 'ac_variance_cb', 'zero_ratio_cb', 'energy_conc_cb',
    # YCbCr (8)
    'Y_mean', 'Y_std', 'Y_var', 'Y_entropy',
    'Cb_mean', 'Cb_std', 'Cr_mean', 'Cr_std',
    # Chroma (4)
    'chroma_Cb_blockvar_mean', 'chroma_Cb_blockvar_std',
    'chroma_Cr_blockvar_mean', 'chroma_Cr_blockvar_std'
]

TARGET_COL = 'LABEL'
CLASSES = ['C0', 'C1', 'C2', 'C3']
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Decision Tree Hyperparameters
DT_MAX_DEPTH = 8
DT_MIN_SAMPLES_LEAF = 5
DT_MIN_SAMPLES_SPLIT = 10

# ===== HELPER FUNCTIONS =====

def log_message(message, level="INFO"):
    """Write message to log and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

def load_and_prepare_data():
    """Load and prepare dataset."""
    print("STEP 1: Loading data...")
    
    if not INPUT_CSV.exists():
        print(f"❌ CSV not found: {INPUT_CSV}")
        return None, None, None
    
    df = pd.read_csv(INPUT_CSV)
    print(f"✓ Loaded {len(df)} samples")
    
    # Select features
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    missing_features = set(FEATURE_COLS) - set(available_features)
    
    if missing_features:
        print(f"⚠ Warning: Missing features: {missing_features}")
    
    X = df[available_features].values
    y = df[TARGET_COL].values
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {np.unique(y)}\n")
    
    log_message(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    return df, X, y, available_features

def analyze_data(df):
    """Analyze class distribution."""
    print("STEP 2: Data Analysis...")
    
    print(f"\n  Class Distribution:")
    class_dist = df[TARGET_COL].value_counts()
    for cls in CLASSES:
        if cls in class_dist.index:
            count = class_dist[cls]
            pct = (count / len(df)) * 100
            bar = '█' * (count // 500)
            print(f"    {cls}: {bar} ({count:6d}, {pct:5.1f}%)")
    
    print(f"\n  Feature Statistics:")
    print(f"    Min: {df[FEATURE_COLS].min().min():.6f}")
    print(f"    Max: {df[FEATURE_COLS].max().max():.6f}")
    print(f"    Mean: {df[FEATURE_COLS].mean().mean():.6f}\n")
    
    return class_dist

def train_decision_tree(X_train, y_train):
    """Train decision tree."""
    print("STEP 3: Training Decision Tree...")
    
    clf = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        min_samples_split=DT_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    print(f"  ✓ Tree trained")
    print(f"    Depth: {clf.get_depth()}")
    print(f"    Leaf nodes: {clf.get_n_leaves()}\n")
    
    log_message(f"Decision tree trained: depth={clf.get_depth()}, leaves={clf.get_n_leaves()}")
    
    return clf

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    """Evaluate decision tree."""
    print("STEP 4: Evaluation")
    
    # Training accuracy
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Test accuracy
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    
    # Per-class metrics
    print(f"  Per-Class Metrics:")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    for cls in CLASSES:
        if cls in report:
            prec = report[cls]['precision']
            rec = report[cls]['recall']
            f1 = report[cls]['f1-score']
            sup = int(report[cls]['support'])
            print(f"    {cls}: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f} (n={sup})")
    
    print()
    
    log_message(f"Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=CLASSES)
    
    return y_train_pred, y_test_pred, train_acc, test_acc, report, cm

def cross_validate(clf, X, y):
    """5-fold cross-validation."""
    print("STEP 5: Cross-Validation...")
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    
    print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})\n")
    
    log_message(f"CV scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    return scores

def create_visualizations(clf, y_test, y_pred, cm, feature_names):
    """Create visualizations."""
    print("STEP 6: Creating Visualizations...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Decision Tree Structure
    print("  Generating tree visualization...")
    fig, ax = plt.subplots(figsize=(25, 15))
    plot_tree(clf, feature_names=feature_names, class_names=CLASSES,
              filled=True, ax=ax, fontsize=10, proportion=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: decision_tree_structure.png")
    
    # 2. Confusion Matrix
    print("  Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=CLASSES, yticklabels=CLASSES, cbar_kws={'label': 'Count'})
    ax.set_title('Decision Tree - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: confusion_matrix.png")
    
    # 3. Feature Importance (from tree)
    print("  Generating feature importance...")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([feature_names[i] for i in indices], importances[indices], color='steelblue')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 15 Features - Decision Tree Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: feature_importance.png\n")
    
    return importances

def save_model_and_report(clf, results, feature_names, importances):
    """Save trained model and report."""
    print("STEP 7: Saving Results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = OUTPUT_DIR / "decision_tree_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  ✓ Saved model: decision_tree_model.pkl")
    
    # Save tree rules as text
    tree_rules = export_text(clf, feature_names=feature_names)
    rules_path = OUTPUT_DIR / "tree_rules.txt"
    with open(rules_path, 'w') as f:
        f.write(tree_rules)
    print(f"  ✓ Saved tree rules: tree_rules.txt")
    
    # Create report
    report_text = f"""
PITSEC DECISION TREE CLASSIFIER REPORT
{'='*70}

Dataset:
  Total samples: {results['total_samples']}
  Training samples: {results['train_samples']}
  Test samples: {results['test_samples']}
  Features: {results['num_features']}

Model Configuration:
  Max Depth: {DT_MAX_DEPTH}
  Min Samples Leaf: {DT_MIN_SAMPLES_LEAF}
  Min Samples Split: {DT_MIN_SAMPLES_SPLIT}

Performance:
  Training Accuracy: {results['train_acc']:.4f} ({results['train_acc']*100:.2f}%)
  Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)
  CV Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})

Tree Structure:
  Depth: {results['depth']}
  Leaf Nodes: {results['leaves']}

Top 5 Important Features:
"""
    
    # Add top features
    indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(indices, 1):
        report_text += f"  {i}. {feature_names[idx]:30s} {importances[idx]:.4f}\n"
    
    report_text += f"""

Per-Class Performance:
"""
    
    for cls in CLASSES:
        if cls in results['report']:
            prec = results['report'][cls]['precision']
            rec = results['report'][cls]['recall']
            f1 = results['report'][cls]['f1-score']
            sup = int(results['report'][cls]['support'])
            report_text += f"  {cls}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f} (n={sup})\n"
    
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  ✓ Saved report: classification_report.txt")
    
    # Save JSON metrics
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'DecisionTreeClassifier',
        'train_accuracy': float(results['train_acc']),
        'test_accuracy': float(results['test_acc']),
        'cv_accuracy_mean': float(results['cv_mean']),
        'cv_accuracy_std': float(results['cv_std']),
        'hyperparameters': {
            'max_depth': DT_MAX_DEPTH,
            'min_samples_leaf': DT_MIN_SAMPLES_LEAF,
            'min_samples_split': DT_MIN_SAMPLES_SPLIT
        },
        'tree_structure': {
            'depth': results['depth'],
            'leaf_nodes': results['leaves']
        }
    }
    
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  ✓ Saved results: results.json\n")
    
    log_message("Model and report saved successfully")

def main():
    """Main pipeline."""
    
    print("\n" + "="*70)
    print("  PITSEC: Decision Tree Classifier")
    print("="*70 + "\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear log
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    
    log_message("Starting Decision Tree classification")
    
    # Load data
    df, X, y, feature_names = load_and_prepare_data()
    if X is None:
        return
    
    class_dist = analyze_data(df)
    
    # Train/test split
    print("STEP 2b: Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples\n")
    
    # Train
    clf = train_decision_tree(X_train, y_train)
    
    # Evaluate
    y_train_pred, y_test_pred, train_acc, test_acc, report, cm = evaluate_model(
        clf, X_train, y_train, X_test, y_test
    )
    
    # Cross-validation
    cv_scores = cross_validate(clf, X, y)
    
    # Visualizations
    importances = create_visualizations(clf, y_test, y_test_pred, cm, feature_names)
    
    # Save
    results = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': X.shape[1],
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'depth': clf.get_depth(),
        'leaves': clf.get_n_leaves(),
        'report': report
    }
    
    save_model_and_report(clf, results, feature_names, importances)
    
    # Summary
    print("="*70)
    print("  CLASSIFICATION COMPLETE")
    print("="*70)
    
    print(f"\nPerformance Summary:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  CV Accuracy:       {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print(f"\nOutput Files:")
    print(f"  Model: {OUTPUT_DIR}/decision_tree_model.pkl")
    print(f"  Visualizations: {OUTPUT_DIR}/*.png")
    print(f"  Report: {OUTPUT_DIR}/classification_report.txt")
    
    print("\n" + "="*70 + "\n")
    
    log_message("Decision Tree classification completed successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Classification interrupted")
        log_message("Interrupted by user", "WARNING")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        log_message(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()