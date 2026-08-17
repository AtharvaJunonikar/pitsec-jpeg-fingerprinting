#!/usr/bin/env python3
"""Compare classifiers: GridSearch DT, RandomForest, XGBoost, SMOTE, Feature selection

Writes results to output/classifier_comparison
"""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_CSV = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/all_features_combined.csv")
OUT_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/classifier_comparison")
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
    if not DATA_CSV.exists():
        raise SystemExit(f"CSV not found: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].values
    y = df[TARGET].values
    return X, y, features


def eval_and_save(name, clf, X_train, y_train, X_test, y_test, features, results):
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'accuracy': float(acc),
        'train_time_sec': float(train_time),
        'report': report,
        'confusion_matrix': cm.tolist()
    }
    # save model metrics immediately
    with open(OUT_DIR / f"{name}_metrics.json", 'w') as f:
        json.dump(results[name], f, indent=2)
    print(f"[{name}] acc={acc:.4f}, train_time={train_time:.1f}s")
    return results


def try_grid_dt(X_train, y_train, X_test, y_test, features, results):
    from sklearn.tree import DecisionTreeClassifier
    print("Running GridSearchCV for DecisionTree (coarse grid)")
    param_grid = {
        'max_depth': [8, 12, 16, None],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print('Best DT params:', gs.best_params_, 'best_score:', gs.best_score_)
    best = gs.best_estimator_
    return eval_and_save('DecisionTree_Grid', best, X_train, y_train, X_test, y_test, features, results)


def try_random_forest(X_train, y_train, X_test, y_test, features, results):
    from sklearn.ensemble import RandomForestClassifier
    print('Training RandomForest baseline')
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
    return eval_and_save('RandomForest', rf, X_train, y_train, X_test, y_test, features, results)


def try_xgboost(X_train, y_train, X_test, y_test, features, results):
    try:
        import xgboost as xgb
    except Exception as e:
        print('xgboost not installed:', e)
        return results
    clf = xgb.XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    return eval_and_save('XGBoost', clf, X_train, y_train, X_test, y_test, features, results)


def try_smote_then_rf(X_train, y_train, X_test, y_test, features, results):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        print('imbalanced-learn not installed:', e)
        return results
    print('Applying SMOTE to training set')
    sm = SMOTE(random_state=42, n_jobs=-1)
    Xb, yb = sm.fit_resample(X_train, y_train)
    print('SMOTE shapes:', Xb.shape, yb.shape)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    return eval_and_save('SMOTE_RandomForest', rf, Xb, yb, X_test, y_test, features, results)


def feature_selection_and_retrain(best_model_name, results, X_train, y_train, X_test, y_test, features):
    # Use a RandomForest for importance if available in results (else train one)
    from sklearn.ensemble import RandomForestClassifier
    print('Computing feature importances with RandomForest')
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    topk = min(15, len(idx))
    top_idx = idx[:topk]
    print('Top features:', [features[i] for i in top_idx])
    Xtr_sel = X_train[:, top_idx]
    Xte_sel = X_test[:, top_idx]

    # Retrain best model type if exists in results
    if 'RandomForest' in results or best_model_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        return eval_and_save('RF_top_features', clf, Xtr_sel, y_train, Xte_sel, y_test, [features[i] for i in top_idx], results)
    else:
        print('Skipping feature-retrain for non-RF best model; training RF on top features for comparison')
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        return eval_and_save('RF_top_features', clf, Xtr_sel, y_train, Xte_sel, y_test, [features[i] for i in top_idx], results)


def main():
    X, y, features = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results = {}

    # 1) Grid search DT
    try:
        results = try_grid_dt(X_train, y_train, X_test, y_test, features, results)
    except Exception as e:
        print('Grid DT failed:', e)

    # 2) Random Forest
    try:
        results = try_random_forest(X_train, y_train, X_test, y_test, features, results)
    except Exception as e:
        print('RF failed:', e)

    # 3) XGBoost
    try:
        results = try_xgboost(X_train, y_train, X_test, y_test, features, results)
    except Exception as e:
        print('XGBoost run failed:', e)

    # 4) SMOTE + RF
    try:
        results = try_smote_then_rf(X_train, y_train, X_test, y_test, features, results)
    except Exception as e:
        print('SMOTE+RF failed:', e)

    # 5) Feature selection + retrain
    try:
        results = feature_selection_and_retrain('RandomForest', results, X_train, y_train, X_test, y_test, features)
    except Exception as e:
        print('Feature selection failed:', e)

    # Save summary
    with open(OUT_DIR / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('All experiments complete. Results saved to', OUT_DIR)


if __name__ == '__main__':
    main()
