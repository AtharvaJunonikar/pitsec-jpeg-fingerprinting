#!/usr/bin/env python3
"""
PITSEC: Dataset Failure Diagnostics

Investigates the 3 validation failures in detail:
  1. FAIL: No Duplicates
  2. FAIL: Class Balance
  3. FAIL: Outliers

Provides actionable fixes for each issue.

Usage:
    python diagnose_failures.py <csv_path>

Example:
    python diagnose_failures.py output/all_features_combined_ssim_fixed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ===== CONFIGURATION =====

FEATURE_COLS = [
    'diff_C0', 'diff_C1', 'diff_C2', 'diff_C3',
    'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3',
    'ac_energy_y', 'dc_variance_y', 'ac_variance_y', 'zero_ratio_y', 'energy_conc_y',
    'ac_energy_cr', 'dc_variance_cr', 'ac_variance_cr', 'zero_ratio_cr', 'energy_conc_cr',
    'ac_energy_cb', 'dc_variance_cb', 'ac_variance_cb', 'zero_ratio_cb', 'energy_conc_cb',
    'Y_mean', 'Y_std', 'Y_var', 'Y_entropy',
    'Cb_mean', 'Cb_std', 'Cr_mean', 'Cr_std',
    'chroma_Cb_blockvar_mean', 'chroma_Cb_blockvar_std',
    'chroma_Cr_blockvar_mean', 'chroma_Cr_blockvar_std'
]

TARGET_COL = 'LABEL'
CLASSES = ['C0', 'C1', 'C2', 'C3']

# ===== FAILURE 1: DUPLICATES =====

def diagnose_duplicates(df):
    """Diagnose duplicate rows."""
    print("\n" + "="*70)
    print("FAILURE 1: DUPLICATES DETECTED")
    print("="*70)
    
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    
    # Check complete duplicates
    complete_dups = df.duplicated(keep=False).sum()
    print(f"\nComplete Duplicates: {complete_dups} rows")
    
    if complete_dups > 0:
        print(f"\nShowing first 5 complete duplicates:")
        dup_mask = df.duplicated(keep=False)
        for i, (idx, row) in enumerate(df[dup_mask].head(10).iterrows()):
            if i % 2 == 0:
                print(f"\n  Row {idx}: {row['file']} (LABEL={row[TARGET_COL]})")
                print(f"    diff_C0={row['diff_C0']:.6f}, ac_energy_y={row['ac_energy_y']:.2f}")
    
    # Check feature-only duplicates
    feature_dups = df[available_features].duplicated(keep=False).sum()
    print(f"\nFeature-Only Duplicates: {feature_dups} rows")
    
    if feature_dups > 0:
        print(f"\nThis means DIFFERENT FILES have IDENTICAL FEATURES!")
        print(f"This is the real problem - indicates data leakage!")
        
        # Find which files have same features
        feature_dups_mask = df[available_features].duplicated(keep=False)
        dup_groups = df[feature_dups_mask].groupby(available_features, dropna=False)
        
        print(f"\nDuplicate Feature Groups (first 3):")
        for i, (features, group) in enumerate(dup_groups):
            if i >= 3:
                break
            print(f"\n  Group {i+1}: {len(group)} rows with identical features")
            for idx, row in group.iterrows():
                print(f"    - {row['file']} (LABEL={row[TARGET_COL]})")
        
        print(f"\n⚠️  ANALYSIS:")
        print(f"    Different images compressed with same version should have")
        print(f"    SLIGHTLY DIFFERENT features due to image content differences.")
        print(f"    Identical features across different images = DATA LEAKAGE!")
    
    # Check duplicates by file
    print(f"\nDuplicate File Analysis:")
    file_counts = df['file'].value_counts()
    multi_count = (file_counts > 1).sum()
    
    if multi_count > 0:
        print(f"  Files appearing > 1 time: {multi_count}")
        print(f"  Top duplicated files:")
        for fname, count in file_counts[file_counts > 1].head(10).items():
            print(f"    {fname}: {count} times")
    else:
        print(f"  All files unique ✓")
    
    return feature_dups > 0

# ===== FAILURE 2: CLASS BALANCE =====

def diagnose_class_balance(df):
    """Diagnose class imbalance."""
    print("\n" + "="*70)
    print("FAILURE 2: CLASS IMBALANCE DETECTED")
    print("="*70)
    
    class_dist = df[TARGET_COL].value_counts()
    total = len(df)
    
    print(f"\nClass Distribution:")
    print(f"{'Class':10s} {'Count':10s} {'Percentage':12s} {'Ratio (vs min)':15s}")
    print("-" * 50)
    
    min_count = class_dist.min()
    max_count = class_dist.max()
    ratio = max_count / min_count
    
    for cls in CLASSES:
        if cls in class_dist.index:
            count = class_dist[cls]
            pct = (count / total) * 100
            ratio_val = count / min_count
            print(f"{cls:10s} {count:10d} {pct:11.2f}% {ratio_val:14.2f}x")
    
    print(f"\nImbalance Ratio: {ratio:.2f}:1 (max:min)")
    print(f"Threshold for FAIL: > 5:1")
    print(f"Your ratio: {ratio:.2f}:1 {'(FAIL)' if ratio > 5 else '(WARNING if > 3)'}")
    
    # Analyze what's causing imbalance
    print(f"\n⚠️  ANALYSIS:")
    
    if 'version' in df.columns:
        print(f"\nClass Distribution by Version:")
        version_dist = df.groupby('version')[TARGET_COL].value_counts().unstack(fill_value=0)
        print(version_dist.head(10))
        
        # Check if all versions are equally represented
        print(f"\nVersion Counts:")
        version_counts = df['version'].value_counts()
        print(version_counts)
        
        # Expected: each version should appear ~2500 times (if 2500 images × 24 versions)
        # But distributed across 4 classes
        avg_per_version = len(df) / df['version'].nunique()
        print(f"\nAverage samples per version: {avg_per_version:.0f}")
        print(f"Expected per version: ~2500 (if 2500 images)")
    
    # Check if imbalance is in the data or expected
    print(f"\n✓ Is this expected?")
    print(f"  - If 24 versions distributed across 4 clusters unequally: YES")
    print(f"  - Example: C0 might have 9 versions, C1 has 11, etc.")
    print(f"  - This is OK if WITHIN each class, images are balanced")
    
    return ratio > 5.0

# ===== FAILURE 3: OUTLIERS =====

def diagnose_outliers(df):
    """Diagnose outliers in detail."""
    print("\n" + "="*70)
    print("FAILURE 3: OUTLIERS DETECTED")
    print("="*70)
    
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    
    print(f"\nAnalyzing {len(available_features)} features for outliers...")
    print(f"(Using 3×IQR threshold for extreme outliers)\n")
    
    outlier_summary = {}
    
    for feat in available_features:
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = df[(df[feat] < lower_bound) | (df[feat] > upper_bound)]
        outlier_pct = (len(outliers) / len(df)) * 100
        
        if len(outliers) > 0:
            outlier_summary[feat] = {
                'count': len(outliers),
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_val': df[feat].min(),
                'max_val': df[feat].max()
            }
    
    # Sort by count
    sorted_outliers = sorted(outlier_summary.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"Features with Outliers (top 10):")
    print(f"{'Feature':30s} {'Count':8s} {'Percent':10s}")
    print("-" * 50)
    
    for feat, info in sorted_outliers[:10]:
        print(f"{feat:30s} {info['count']:8d} {info['percentage']:9.2f}%")
    
    # Total outlier percentage
    total_outlier_points = sum(info['count'] for info in outlier_summary.values())
    total_points = len(df) * len(available_features)
    total_pct = (total_outlier_points / total_points) * 100
    
    print(f"\nTotal Outlier Points: {total_outlier_points} / {total_points}")
    print(f"Percentage: {total_pct:.4f}%")
    print(f"Threshold for FAIL: > 1.0%")
    print(f"Your percentage: {total_pct:.4f}% {'(FAIL)' if total_pct > 1.0 else '(PASS)'}")
    
    # Analyze if outliers are expected
    print(f"\n⚠️  ANALYSIS:")
    print(f"\nOutliers might be EXPECTED if:")
    print(f"  1. Features have very different scales")
    print(f"  2. Some versions produce extreme compression artifacts")
    print(f"  3. Some images are edge cases")
    
    # Check by class
    print(f"\nOutliers by Class:")
    for cls in CLASSES:
        cls_data = df[df[TARGET_COL] == cls]
        cls_outliers = 0
        
        for feat in available_features:
            Q1 = cls_data[feat].quantile(0.25)
            Q3 = cls_data[feat].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = cls_data[(cls_data[feat] < lower_bound) | (cls_data[feat] > upper_bound)]
            cls_outliers += len(outliers)
        
        cls_pct = (cls_outliers / (len(cls_data) * len(available_features))) * 100
        print(f"  {cls}: {cls_pct:.2f}%")
    
    return total_pct > 1.0

# ===== REMEDIATION =====

def recommend_fixes(has_dup, has_imbalance, has_outliers):
    """Recommend fixes based on failures."""
    print("\n" + "="*70)
    print("REMEDIATION & FIX RECOMMENDATIONS")
    print("="*70)
    
    print("\n🔧 ACTIONS TO TAKE (in order):\n")
    
    severity = 0
    
    if has_dup:
        severity += 3
        print("1. CRITICAL - Handle Duplicates:")
        print("""
   Problem: Duplicate feature rows = DATA LEAKAGE
   
   Action A (Recommended): Remove duplicates
   -----------
   import pandas as pd
   df = pd.read_csv('all_features_combined_ssim_fixed.csv')
   
   # Remove complete duplicates
   df_clean = df.drop_duplicates(subset=FEATURE_COLS, keep='first')
   
   # Save cleaned data
   df_clean.to_csv('all_features_combined_deduplicated.csv', index=False)
   print(f"Removed {len(df) - len(df_clean)} duplicate rows")
   
   # Re-run validation
   python dataset_validation.py all_features_combined_deduplicated.csv
   -----------
   
   Action B (Alternative): Investigate cause
   -----------
   # Why are features identical for different images?
   # Possible reasons:
   #   - Images are actually the same (copy/paste error)
   #   - Compression is deterministic (expected)
   #   - Feature extraction has precision issues (rounding)
   
   # If features SHOULD differ:
   #   - Check feature extraction code
   #   - Verify image data is different
   #   - Increase feature precision
   -----------
""")
    
    if has_imbalance:
        severity += 2
        print("2. IMPORTANT - Fix Class Imbalance:")
        print("""
   Problem: Classes not equally distributed
   
   Action A (Quick Fix): Use class_weight='balanced'
   -----------
   # In your classifier:
   from sklearn.ensemble import RandomForestClassifier
   
   clf = RandomForestClassifier(
       class_weight='balanced',  # ← Add this
       n_estimators=200
   )
   -----------
   
   Action B (Better Fix): Use SMOTE
   -----------
   from imblearn.over_sampling import SMOTE
   
   X_train_balanced, y_train_balanced = SMOTE().fit_resample(X_train, y_train)
   clf.fit(X_train_balanced, y_train_balanced)
   -----------
   
   Action C (Best Fix): Check if imbalance is expected
   -----------
   # If you have:
   # - 9 versions in C0
   # - 11 versions in C1
   # - 2 versions in C2
   # - 2 versions in C3
   #
   # This explains the imbalance!
   # The ratio is EXPECTED, not a data error.
   # Use class_weight='balanced' to handle it.
   -----------
""")
    
    if has_outliers:
        severity += 1
        print("3. MODERATE - Handle Outliers:")
        print("""
   Problem: > 1% outliers detected
   
   Action A (Quick Fix): Standardize features
   -----------
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   # Use X_scaled for training
   -----------
   
   Action B (Better Fix): Robust scaling
   -----------
   from sklearn.preprocessing import RobustScaler
   
   scaler = RobustScaler()  # Less affected by outliers
   X_scaled = scaler.fit_transform(X)
   -----------
   
   Action C (Alternative): Remove extreme outliers
   -----------
   # Remove if value > 99th percentile
   Q99 = df[features].quantile(0.99)
   df_filtered = df[(df[features] <= Q99).all(axis=1)]
   
   # But be careful - you lose data!
   print(f"Removed {len(df) - len(df_filtered)} rows")
   -----------
""")
    
    print("\n" + "="*70)
    print("SEVERITY ASSESSMENT")
    print("="*70)
    
    if severity >= 5:
        print(f"\n🔴 CRITICAL ({severity}/6)")
        print("""
Your dataset has significant issues!

Recommended action:
  1. Fix duplicates FIRST (most important)
  2. Handle class imbalance with class_weight='balanced'
  3. Use StandardScaler for outliers
  4. Re-run validation
  5. Re-train classifier
  6. Compare accuracy before/after fixes

Do NOT deploy until accuracy is verified after fixes!
""")
    elif severity >= 3:
        print(f"\n🟡 MODERATE ({severity}/6)")
        print("""
Your dataset has some issues but they're manageable.

Recommended action:
  1. Apply remediation steps above
  2. Use appropriate sklearn options (class_weight, scaling)
  3. Re-run validation
  4. Re-train classifier
  5. Verify accuracy improves or stays same

You can likely still deploy after fixes.
""")
    else:
        print(f"\n🟢 MINOR ({severity}/6)")
        print("""
Your dataset issues are minimal.

Recommended action:
  1. Apply preventive measures (class_weight, scaling)
  2. Monitor model performance
  3. Good to deploy but keep monitoring

Your 98%+ accuracy is likely legitimate!
""")

# ===== VISUALIZATION =====

def create_diagnostic_plots(df, output_dir='output'):
    """Create diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating diagnostic plots...")
    
    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    df[TARGET_COL].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Class Distribution (Checking for Imbalance)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_class_distribution.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: diagnostic_class_distribution.png")
    
    # 2. Outlier distribution for key features
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, feat in enumerate(available_features[:6]):
        ax = axes[idx // 3, idx % 3]
        
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        
        outliers = df[(df[feat] < lower) | (df[feat] > upper)]
        
        ax.hist(df[feat], bins=50, alpha=0.7, label='Normal', color='blue')
        if len(outliers) > 0:
            ax.hist(outliers[feat], bins=20, alpha=0.7, label='Outliers', color='red')
        
        ax.axvline(lower, color='red', linestyle='--', alpha=0.5, label='Bounds')
        ax.axvline(upper, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{feat}\n({len(outliers)} outliers, {len(outliers)/len(df)*100:.2f}%)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_outliers.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: diagnostic_outliers.png")

# ===== MAIN =====

def main():
    """Run all diagnostics."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_failures.py <csv_path>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("  PITSEC: Failure Diagnostics")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Run diagnostics
    has_dup = diagnose_duplicates(df)
    has_imbalance = diagnose_class_balance(df)
    has_outliers = diagnose_outliers(df)
    
    # Create plots
    create_diagnostic_plots(df)
    
    # Recommendations
    recommend_fixes(has_dup, has_imbalance, has_outliers)
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
