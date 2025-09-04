#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Script: 2-PCA.py
# Author: Bryan Ambrósio
# Date: 2025-09-02
# Summary:
#   - Loads the cleaned dataset produced by 1-EDA.py
#   - Automatically detects "long" (018019s, ...) or "short" (0.18s, ...) names
#   - Uses EXACTLY 8 Vang_* features (0.18s, 0.25s, 0.30s, 0.35s)
#   - Ignores extra columns (e.g., Estabilidade), checks NaN/inf
#   - Stratified split (safe fallback if needed), StandardScaler, PCA(2)
#   - Saves figures, PCA data, artifacts; writes features_used.txt
#
# Output folders:
#   - ../PCA_visualization/
#   - ../data_PCA/
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from datetime import datetime

# ----------------------------- Configuration -------------------------------
TARGET = "min_UGs"
VALID_CLASSES = set(range(0, 8))

# "Long" pattern (as in EDA before renaming)
FEATURES_LONG = [
    "Vang_XES_018019s",      "Vang_RioVTEST_018019s",
    "Vang_XES_025026s",      "Vang_RioVTEST_025026s",
    "Vang_XES_030031s",      "Vang_RioVTEST_030031s",
    "Vang_XES_035036s",      "Vang_RioVTEST_035036s",
]

# "Short" pattern (as in your cleaned CSV shown in logs)
FEATURES_SHORT = [
    "Vang_XES_0.18s",        "Vang_RioVTEST_0.18s",
    "Vang_XES_0.25s",        "Vang_RioVTEST_0.25s",
    "Vang_XES_0.30s",        "Vang_RioVTEST_0.30s",
    "Vang_XES_0.35s",        "Vang_RioVTEST_0.35s",
]

# Maps short -> long (only for canonical display if desired)
SHORT_TO_LONG = {
    "Vang_XES_0.18s":        "Vang_XES_018019s",
    "Vang_RioVTEST_0.18s":   "Vang_RioVTEST_018019s",
    "Vang_XES_0.25s":        "Vang_XES_025026s",
    "Vang_RioVTEST_0.25s":   "Vang_RioVTEST_025026s",
    "Vang_XES_0.30s":        "Vang_XES_030031s",
    "Vang_RioVTEST_0.30s":   "Vang_RioVTEST_030031s",
    "Vang_XES_0.35s":        "Vang_XES_035036s",
    "Vang_RioVTEST_0.35s":   "Vang_RioVTEST_035036s",
}


# ----------------------------- Plotting helper ------------------------------
def plot_pca_scatter(X_pca: np.ndarray, y: np.ndarray, title: str, outpath: Path) -> None:
    """
    Scatter-plot helper for 2D PCA outputs, colored by class (min_UGs).
    """
    plt.figure(figsize=(7.2, 5.4))
    classes = np.unique(y)
    cmap = plt.get_cmap("tab10")
    for k in sorted(classes):
        mk = (y == k)
        plt.scatter(
            X_pca[mk, 0], X_pca[mk, 1],
            s=20, alpha=0.85, label=str(k),
            color=cmap(int(k) % 10), edgecolors="none"
        )
    plt.xlabel("PC1", fontsize=11)
    plt.ylabel("PC2", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(title="min_UGs", ncol=4, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ------------------------- Robust stratified splitter ------------------------
def safe_stratified_split(X, y, test_size=0.25, random_state=42):
    """
    Attempts a stratified split; if any class has < 2 samples,
    falls back to a non-stratified split to avoid sklearn errors.
    """
    counts = pd.Series(y).value_counts()
    if (counts < 2).any():
        print("[Warning] Some class has < 2 samples; using NON-stratified split.")
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ------------------------------ Feature picker ------------------------------
def pick_features(df: pd.DataFrame) -> list[str]:
    """
    Automatically selects the 8 features according to available names.
    Prefers short names if both conventions are present.
    """
    have_long = all(c in df.columns for c in FEATURES_LONG)
    have_short = all(c in df.columns for c in FEATURES_SHORT)

    if have_long and have_short:
        print("[Info] Both conventions detected; using SHORT names (0.xx s).")
        return FEATURES_SHORT
    if have_short:
        print("[Info] SHORT names detected (0.xx s).")
        return FEATURES_SHORT
    if have_long:
        print("[Info] LONG names detected (xx-xx ms ranges).")
        return FEATURES_LONG

    # Neither complete set is present: show what's missing in each pattern
    missing_long = [c for c in FEATURES_LONG if c not in df.columns]
    missing_short = [c for c in FEATURES_SHORT if c not in df.columns]
    raise KeyError(
        "Could not locate the expected 8 features.\n"
        f"Missing (LONG pattern): {missing_long}\n"
        f"Missing (SHORT pattern): {missing_short}\n"
        "Please check the cleaned CSV header."
    )


# --------------------------------- Main -------------------------------------
def main() -> None:
    # -------------------------------------------------------------------------
    # 1) Paths and data loading
    # -------------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    data_cleaned_dir = script_dir.parent / "data_cleaned"
    input_csv = data_cleaned_dir / "tabelaestabilidade_minUG_cleaned.csv"

    if not input_csv.exists():
        raise FileNotFoundError(
            f"File not found: {input_csv}\n"
            f"Please run 1-EDA.py first to generate the cleaned CSV."
        )

    print(f"Loading cleaned dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    print("\nDataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # -------------------------------------------------------------------------
    # 2) Target/features and sanity checks
    # -------------------------------------------------------------------------
    if TARGET not in df.columns:
        raise KeyError(f"Column '{TARGET}' not found in dataset.")

    # Selects the correct naming convention automatically
    feats = pick_features(df)

    # Keep only target + 8 features; ignore extras (e.g., Estabilidade)
    df = df[[TARGET] + feats].copy()

    # Ensure target integrity and valid classes
    df[TARGET] = df[TARGET].astype(int)
    mask_valid = df[TARGET].isin(VALID_CLASSES)
    if not mask_valid.all():
        removed = int((~mask_valid).sum())
        print(f"[Warning] Removing {removed} rows with labels outside 0..7.")
        df = df[mask_valid].copy()

    # Drop rows with NaN/inf in features/target
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + [TARGET])
    removed_nan = before - len(df)
    if removed_nan > 0:
        print(f"[Warning] Removed {removed_nan} rows containing NaN/inf in features/target.")

    # Final matrices
    X = df[feats].values.astype(float)
    y = df[TARGET].values.astype(int)

    # -------------------------------------------------------------------------
    # 3) Train/test split (stratified with safe fallback)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = safe_stratified_split(
        X, y, test_size=0.25, random_state=42
    )

    print("\nShapes:")
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_test: ", X_test.shape,  "| y_test: ", y_test.shape)

    # -------------------------------------------------------------------------
    # 4) Standardization (fit on train) + PCA(2) (fit on train)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)   # fit only on train
    X_test_pca  = pca.transform(X_test_scaled)        # transform test with same mapping

    var_exp = pca.explained_variance_ratio_
    print(f"\nExplained variance ratio (PC1, PC2): {var_exp.tolist()} | Sum = {var_exp.sum():.4f}")

    # -------------------------------------------------------------------------
    # 5) Figures -> ../PCA_visualization/
    # -------------------------------------------------------------------------
    out_fig = script_dir.parent / "PCA_visualization"
    out_fig.mkdir(parents=True, exist_ok=True)

    plot_pca_scatter(
        X_train_pca, y_train,
        "Training set in PCA space (PC1 vs PC2)",
        out_fig / "pca_train_scatter.png"
    )
    plot_pca_scatter(
        X_test_pca, y_test,
        "Test set in PCA space (PC1 vs PC2)",
        out_fig / "pca_test_scatter.png"
    )

    with open(out_fig / "pca_explained_variance.txt", "w", encoding="utf-8") as f:
        f.write(f"Explained variance ratio (PC1, PC2): {var_exp.tolist()}\n")
        f.write(f"Sum: {float(var_exp.sum()):.6f}\n")

    # -------------------------------------------------------------------------
    # 6) Save PCA-transformed dataframes and artifacts for next steps (e.g., SVM)
    # -------------------------------------------------------------------------
    out_pca = script_dir.parent / "data_PCA"
    out_pca.mkdir(parents=True, exist_ok=True)

    df_train_pca = pd.DataFrame({
        "PC1": X_train_pca[:, 0],
        "PC2": X_train_pca[:, 1],
        "min_UGs": y_train,
        "split": "train",
    })
    df_test_pca = pd.DataFrame({
        "PC1": X_test_pca[:, 0],
        "PC2": X_test_pca[:, 1],
        "min_UGs": y_test,
        "split": "test",
    })
    df_all_pca = pd.concat([df_train_pca, df_test_pca], ignore_index=True)

    # CSV outputs
    df_train_pca.to_csv(out_pca / "train_pca.csv", index=False)
    df_test_pca.to_csv(out_pca / "test_pca.csv", index=False)
    df_all_pca.to_csv(out_pca / "all_pca.csv", index=False)

    # Parquet outputs (optional, requires pyarrow or fastparquet)
    try:
        df_train_pca.to_parquet(out_pca / "train_pca.parquet", index=False)
        df_test_pca.to_parquet(out_pca / "test_pca.parquet", index=False)
        df_all_pca.to_parquet(out_pca / "all_pca.parquet", index=False)
    except Exception as e:
        print("\n[Warning] Failed to save Parquet (install 'pyarrow' or 'fastparquet').")
        print("Error:", e)

    # Persist scaler and PCA model for downstream use
    joblib.dump(scaler, out_pca / "scaler.joblib")
    joblib.dump(pca,    out_pca / "pca.joblib")

    # -------------------------------------------------------------------------
    # 7) Save features_used.txt (traceability)
    # -------------------------------------------------------------------------
    features_txt = out_pca / "features_used.txt"
    with open(features_txt, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Target: {TARGET}\n")
        f.write("Features (order, as in CSV):\n")
        for i, col in enumerate(feats, start=1):
            long_name = SHORT_TO_LONG.get(col, col)
            f.write(f"  {i:02d}. {col} (canon: {long_name})\n")
        f.write("\nShapes:\n")
        f.write(f"  X_train: {X_train.shape} | y_train: {y_train.shape}\n")
        f.write(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}\n")
        f.write("\nExplained variance ratio (PC1, PC2): ")
        f.write(f"{var_exp.tolist()} | Sum = {float(var_exp.sum()):.6f}\n")

    # -------------------------------------------------------------------------
    # 8) Final logs
    # -------------------------------------------------------------------------
    print("\n=== PCA data saved to ===")
    for p in sorted(out_pca.glob("*")):
        print(" -", p)

    print("\nFigures saved to:", out_fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
