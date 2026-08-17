#!/usr/bin/env python3
"""
Comprehensive evaluation of Decision Tree and Random Forest classifiers
Generates: Confusion Matrix, F1 Score, Precision, Recall, ROC-AUC, Classification Report
Includes visualizations and JSON export of all metrics
"""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

# Configuration
DATA_CSV = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/all_features_combined.csv")  # Update path as needed
OUT_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/model_evaluation_results_3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'LABEL'

FEATURE_COLS = [
    'diff_C0', 'diff_C1', 'diff_C2', 'diff_C3',
    'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3',
    'ac_energy_y', 'dc_variance_y', 'ac_variance_y', 'zero_ratio_y', 'energy_conc_y',
    'ac_energy_cr', 'dc_variance_cr', 'ac_variance_cr', 'zero_ratio_cr', 'energy_conc_cr',
    'ac_energy_cb', 'dc_variance_cb', 'ac_variance_cb', 'zero_ratio_cb', 'energy_conc_cb',
    'Y_mean', 'Y_std', 'Y_var', 'Y_entropy',
    'Cb_mean', 'Cb_std', 'Cr_mean', 'Cr_std',
    'chroma_Cb_blockvar_mean', 'chroma_Cb_blockvar_std', 'chroma_Cr_blockvar_mean', 'chroma_Cr_blockvar_std'
]


def load_data():
    """Load and prepare data"""
    if not DATA_CSV.exists():
        raise SystemExit(f"CSV not found: {DATA_CSV}")
    
    df = pd.read_csv(DATA_CSV)
    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].values
    y = df[TARGET].values
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}\n")
    
    return X, y, features


def compute_metrics(y_true, y_pred, y_proba=None, model_name="", class_labels=None):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for ROC-AUC)
        model_name: Name of the model
        class_labels: List of class names
    
    Returns:
        Dictionary with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report (as dict for JSON)
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision_micro': float(precision_micro),
        'precision_macro': float(precision_macro),
        'recall_micro': float(recall_micro),
        'recall_macro': float(recall_macro),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': clf_report,
        'precision_per_class': precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else precision_per_class,
        'recall_per_class': recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else recall_per_class,
        'f1_per_class': f1_per_class.tolist() if isinstance(f1_per_class, np.ndarray) else f1_per_class,
    }
    
    # ROC-AUC (one-vs-rest for multiclass)
    if y_proba is not None and len(np.unique(y_true)) > 2:
        try:
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            # Ensure y_proba has same shape as y_true_bin
            if y_proba.shape[1] == y_true_bin.shape[1]:
                roc_auc_ovr = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                roc_auc_ovo = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovo')
                metrics['roc_auc_ovr'] = float(roc_auc_ovr)
                metrics['roc_auc_ovo'] = float(roc_auc_ovo)
        except Exception as e:
            print(f"  Warning: Could not compute ROC-AUC: {e}")
    elif y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            metrics['roc_auc'] = float(roc_auc)
        except Exception as e:
            print(f"  Warning: Could not compute ROC-AUC: {e}")
    
    return metrics


def print_metrics(metrics):
    """Pretty print metrics"""
    print(f"\n{'='*70}")
    print(f"Model: {metrics['model_name']}")
    print(f"{'='*70}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"\nPrecision (Micro):  {metrics['precision_micro']:.4f}")
    print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"\nRecall (Micro):     {metrics['recall_micro']:.4f}")
    print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"\nF1 Score (Micro):   {metrics['f1_micro']:.4f}")
    print(f"F1 Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted):{metrics['f1_weighted']:.4f}")
    
    if 'roc_auc_ovr' in metrics:
        print(f"\nROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
        print(f"ROC-AUC (OvO):      {metrics['roc_auc_ovo']:.4f}")
    elif 'roc_auc' in metrics:
        print(f"\nROC-AUC:            {metrics['roc_auc']:.4f}")
    
    print(f"\n{'Confusion Matrix':-^70}")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    print(f"{'='*70}\n")


def plot_confusion_matrix(cm, model_name, class_labels=None):
    """Plot and save confusion matrix as heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Use class labels if provided, else use numeric indices
    if class_labels is None:
        class_labels = [str(i) for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path = OUT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix plot: {save_path}")
    plt.close()


def plot_metrics_comparison(all_metrics, metric_names=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']):
    """Plot comparison of metrics across models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        models = [m['model_name'] for m in all_metrics]
        values = [m.get(metric, 0) for m in all_metrics]
        
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Comparison: Key Metrics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = OUT_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot: {save_path}")
    plt.close()


def plot_per_class_metrics(all_metrics, class_labels=None):
    """Plot per-class precision, recall, F1 for each model"""
    if class_labels is None:
        # Assume 4 classes (C0, C1, C2, C3) based on the feature names
        class_labels = ['C0', 'C1', 'C2', 'C3']
    
    fig, axes = plt.subplots(len(all_metrics), 3, figsize=(15, 4*len(all_metrics)))
    
    if len(all_metrics) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, metrics in enumerate(all_metrics):
        model_name = metrics['model_name']
        
        # Precision per class
        ax = axes[row_idx, 0]
        precision = metrics['precision_per_class']
        ax.bar(class_labels, precision, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} - Precision per Class', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Recall per class
        ax = axes[row_idx, 1]
        recall = metrics['recall_per_class']
        ax.bar(class_labels, recall, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} - Recall per Class', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # F1 per class
        ax = axes[row_idx, 2]
        f1 = metrics['f1_per_class']
        ax.bar(class_labels, f1, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} - F1 Score per Class', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = OUT_DIR / "per_class_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class metrics plot: {save_path}")
    plt.close()


def train_and_evaluate_dt(X_train, y_train, X_test, y_test):
    """Train Decision Tree with specified parameters"""
    print("\n" + "="*70)
    print("Training Decision Tree Classifier (Grid Search Parameters)")
    print("="*70)
    
    # Using best parameters from GridSearch
    dt = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=2,
        criterion='entropy',
        random_state=42
    )
    
    t0 = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)
    
    metrics = compute_metrics(y_test, y_pred, y_proba, model_name="Decision Tree")
    metrics['train_time_sec'] = float(train_time)
    
    print(f"Training time: {train_time:.2f} seconds")
    print_metrics(metrics)
    
    # Save model
    model_path = OUT_DIR / 'decision_tree_model.joblib'
    joblib.dump(dt, model_path)
    print(f"Saved Decision Tree model: {model_path}")
    
    return dt, metrics, y_pred, y_proba


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest with specified parameters"""
    print("\n" + "="*70)
    print("Training Random Forest Classifier")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1
    )
    
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    metrics = compute_metrics(y_test, y_pred, y_proba, model_name="Random Forest")
    metrics['train_time_sec'] = float(train_time)
    
    print(f"Training time: {train_time:.2f} seconds")
    print_metrics(metrics)
    
    # Save model
    model_path = OUT_DIR / 'random_forest_model.joblib'
    joblib.dump(rf, model_path)
    print(f"Saved Random Forest model: {model_path}")
    
    return rf, metrics, y_pred, y_proba


def plot_feature_importance(rf_model, features, top_k=15):
    """Plot feature importance from Random Forest"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.bar(range(top_k), importances[indices], align='center', color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.xticks(range(top_k), [features[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = OUT_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot: {save_path}")
    plt.close()


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION: Decision Tree vs Random Forest")
    print("="*70)
    
    # Load data
    X, y, features = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    # Train models
    dt_model, dt_metrics, dt_pred, dt_proba = train_and_evaluate_dt(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics, rf_pred, rf_proba = train_and_evaluate_rf(X_train, y_train, X_test, y_test)
    
    all_metrics = [dt_metrics, rf_metrics]
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    
    class_labels = np.unique(y_test)
    
    plot_confusion_matrix(np.array(dt_metrics['confusion_matrix']), "Decision Tree", class_labels=class_labels)
    plot_confusion_matrix(np.array(rf_metrics['confusion_matrix']), "Random Forest", class_labels=class_labels)
    plot_metrics_comparison(all_metrics, metric_names=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
    plot_per_class_metrics(all_metrics, class_labels=class_labels)
    plot_feature_importance(rf_model, features, top_k=15)
    
    # Save results to JSON
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)
    
    results_summary = {
        'models': all_metrics,
        'test_set_size': X_test.shape[0],
        'train_set_size': X_train.shape[0],
        'num_features': X_train.shape[1],
        'classes': class_labels.tolist()
    }
    
    with open(OUT_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved evaluation results: {OUT_DIR / 'evaluation_results.json'}")
    
    # Save detailed comparison
    comparison = {
        'Decision_Tree': dt_metrics,
        'Random_Forest': rf_metrics,
        'winner_by_accuracy': 'Random Forest' if rf_metrics['accuracy'] > dt_metrics['accuracy'] else 'Decision Tree',
        'winner_by_f1_macro': 'Random Forest' if rf_metrics['f1_macro'] > dt_metrics['f1_macro'] else 'Decision Tree',
        'accuracy_difference': float(abs(rf_metrics['accuracy'] - dt_metrics['accuracy'])),
        'f1_macro_difference': float(abs(rf_metrics['f1_macro'] - dt_metrics['f1_macro']))
    }
    
    with open(OUT_DIR / 'model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved model comparison: {OUT_DIR / 'model_comparison.json'}")
    
    # Save model metadata (feature names, classes)
    model_metadata = {
        'feature_names': features,
        'classes': class_labels.tolist(),
        'n_features': len(features),
        'n_classes': len(class_labels),
        'feature_cols_used': FEATURE_COLS
    }
    
    with open(OUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"Saved model metadata: {OUT_DIR / 'model_metadata.json'}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Decision Tree Accuracy:     {dt_metrics['accuracy']:.4f}")
    print(f"Random Forest Accuracy:     {rf_metrics['accuracy']:.4f}")
    print(f"Accuracy Difference:        {abs(rf_metrics['accuracy'] - dt_metrics['accuracy']):.4f}")
    print(f"\nDecision Tree F1 (Macro):   {dt_metrics['f1_macro']:.4f}")
    print(f"Random Forest F1 (Macro):   {rf_metrics['f1_macro']:.4f}")
    print(f"F1 Difference:              {abs(rf_metrics['f1_macro'] - dt_metrics['f1_macro']):.4f}")
    print(f"\nWinner by Accuracy:         {'Random Forest' if rf_metrics['accuracy'] > dt_metrics['accuracy'] else 'Decision Tree'}")
    print(f"Winner by F1 Score:         {'Random Forest' if rf_metrics['f1_macro'] > dt_metrics['f1_macro'] else 'Decision Tree'}")
    print(f"\nAll results saved to:       {OUT_DIR}")
    print("\n" + "="*70)
    print("SAVED FILES")
    print("="*70)
    print("Models:")
    print(f"  - {OUT_DIR / 'decision_tree_model.joblib'}")
    print(f"  - {OUT_DIR / 'random_forest_model.joblib'}")
    print("\nMetrics & Results:")
    print(f"  - {OUT_DIR / 'evaluation_results.json'}")
    print(f"  - {OUT_DIR / 'model_comparison.json'}")
    print(f"  - {OUT_DIR / 'model_metadata.json'}")
    print("\nVisualizations:")
    print(f"  - {OUT_DIR / 'confusion_matrix_Decision Tree.png'}")
    print(f"  - {OUT_DIR / 'confusion_matrix_Random Forest.png'}")
    print(f"  - {OUT_DIR / 'metrics_comparison.png'}")
    print(f"  - {OUT_DIR / 'per_class_metrics.png'}")
    print(f"  - {OUT_DIR / 'feature_importance.png'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
