#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Script: 1-EDA.py
# Author: Bryan Ambrósio
# Date: 2025-09-02
# Summary:
#   - Loads the spreadsheet "tabelaestabilidade_minUG.xlsx" from ../data_raw/
#   - Uses target = min_UGs and features Vang_* at 0.18s, 0.25s, 0.30s, 0.35s
#   - Handles decimal comma via read_excel(decimal=",")
#   - Diagnostics: missing columns, empty columns, NaNs
#   - Removes rows with invalid target (outside 0..7)
#   - Removes rows with ("Estabilidade"="Instavel" and "min_UGs"=0)
#   - Removes rows with ANY negative feature
#   - Generates SCATTERPLOTS pre/post-cleaning and HISTOGRAMS (target=0)
#   - Saves cleaned data in ../data_cleaned/ (CSV and Parquet)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pathlib import Path

# ------------------------ Main configurations -------------------------------
# Target column name
TARGET = "min_UGs"

# Selected features: voltage angle (Vang) at specific timestamps
FEATURES = [
    "Vang_XES_018019s",      "Vang_RioVTEST_018019s",
    "Vang_XES_025026s",      "Vang_RioVTEST_025026s",
    "Vang_XES_030031s",      "Vang_RioVTEST_030031s",
    "Vang_XES_035036s",      "Vang_RioVTEST_035036s",
]

# Mapping to rename feature names into a human-readable format
RENAME_TIME = {
    "Vang_XES_018019s":      "Vang_XES_0.18s",
    "Vang_RioVTEST_018019s": "Vang_RioVTEST_0.18s",
    "Vang_XES_025026s":      "Vang_XES_0.25s",
    "Vang_RioVTEST_025026s": "Vang_RioVTEST_0.25s",
    "Vang_XES_030031s":      "Vang_XES_0.30s",
    "Vang_RioVTEST_030031s": "Vang_RioVTEST_0.30s",
    "Vang_XES_035036s":      "Vang_XES_0.35s",
    "Vang_RioVTEST_035036s": "Vang_RioVTEST_0.35s",
}

# Valid range for target classes
MIN_CLASS, MAX_CLASS = 0, 7


# ------------------------ Diagnostics function -------------------------------
def diagnosticos_colunas(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Runs diagnostics for the selected columns:
    - Checks for missing columns
    - Flags columns fully empty (100% NaN)
    - Counts NaN values per column
    - Prints sample rows
    """
    presentes = [c for c in cols if c in df.columns]
    ausentes = [c for c in cols if c not in df.columns]

    print("\n=== Column diagnostics ===")
    print(f"Desired: {len(cols)}  | Present: {len(presentes)}  | Missing: {len(ausentes)}")
    if ausentes:
        print("Missing columns:", ausentes)

    vazias = [c for c in presentes if df[c].isna().all()]
    if vazias:
        print("Empty columns (100% NaN):", vazias)

    print("\nNaN counts by column (present ones):")
    for c in presentes:
        n_nan = df[c].isna().sum()
        if n_nan > 0:
            print(f"  - {c}: {n_nan} NaNs")

    print("\nSample of 5 rows (present columns):")
    print(df[presentes].head(5))


# ----------------------------- Main function --------------------------------
def main() -> None:
    # -------------------------------------------------------------------------
    # 1) File path and Excel loading
    # -------------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir.parent / "data_raw" / "tabelaestabilidade_minUG.xlsx"

    print("Loading file:", file_path)
    # Important: decimal="," to handle comma as decimal separator
    df = pd.read_excel(file_path, decimal=",")

    # -------------------------------------------------------------------------
    # 2) Select relevant columns (target + features + Estabilidade if present)
    # -------------------------------------------------------------------------
    cols_interesse = [TARGET] + FEATURES
    diagnosticos_colunas(df, cols_interesse)

    cols_presentes = [c for c in cols_interesse if c in df.columns]
    if TARGET not in cols_presentes:
        raise ValueError(f"Target '{TARGET}' not found in file.")

    # Keep relevant columns (includes "Estabilidade" if available)
    df_red = df[cols_presentes + ["Estabilidade"]].copy() if "Estabilidade" in df.columns else df[cols_presentes].copy()

    # Rename columns to human-readable format
    ren_map = {k: v for k, v in RENAME_TIME.items() if k in df_red.columns}
    df_red.rename(columns=ren_map, inplace=True)

    # Identify feature columns (everything except target and Estabilidade)
    feat_cols = [c for c in df_red.columns if c not in [TARGET, "Estabilidade"]]

    # -------------------------------------------------------------------------
    # 3) Remove invalid target rows (NaN or outside range)
    # -------------------------------------------------------------------------
    total_before = len(df_red)
    df_red = df_red[df_red[TARGET].notna()].copy()
    df_red = df_red[(df_red[TARGET] >= MIN_CLASS) & (df_red[TARGET] <= MAX_CLASS)].copy()
    total_after = len(df_red)
    print(f"\nRows before target cleaning: {total_before}")
    print(f"Rows after target cleaning: {total_after}")

    # -------------------------------------------------------------------------
    # 4) Additional cleaning: inconsistent rows
    #    Special case: (Estabilidade="Instavel" and min_UGs=0)
    # -------------------------------------------------------------------------
    if "Estabilidade" in df_red.columns:
        cond_invalid = (df_red["Estabilidade"] == "Instavel") & (df_red[TARGET] == 0)
        qtd_invalid = cond_invalid.sum()
        if qtd_invalid > 0:
            print(f"\nRemoving {qtd_invalid} rows with Estabilidade='Instavel' and {TARGET}=0")
            df_red = df_red[~cond_invalid].copy()

    # -------------------------------------------------------------------------
    # 5) Pre-cleaning visualizations
    #    Generate scatterplots and histograms before removing negatives
    # -------------------------------------------------------------------------
    out_dir_scatter_pre = script_dir.parent / "feature_vs_target_pre_cleaning"
    out_dir_scatter_pre.mkdir(parents=True, exist_ok=True)

    out_dir_hist_pre = script_dir.parent / "histograms_pre_cleaning"
    out_dir_hist_pre.mkdir(parents=True, exist_ok=True)

    df_plot_pre = df_red.copy()
    df_plot_pre[TARGET] = df_plot_pre[TARGET].astype(int)

    # Color map for classes 0..7
    cmap = get_cmap("tab10")
    class_colors = {k: cmap(k) for k in range(MIN_CLASS, MAX_CLASS + 1)}

    for feat in feat_cols:
        plt.figure(figsize=(7, 4.5))
        for k in range(MIN_CLASS, MAX_CLASS + 1):
            dsub = df_plot_pre[df_plot_pre[TARGET] == k]
            if dsub.empty:
                continue
            plt.scatter(
                dsub[feat], dsub[TARGET],
                s=14, alpha=0.7, label=f"{k}",
                color=class_colors[k], edgecolors="none",
            )
        plt.xlabel(feat)
        plt.ylabel(TARGET)
        plt.yticks(range(MIN_CLASS, MAX_CLASS + 1))
        plt.title(f"{feat} vs {TARGET} (pre-cleaning)")
        plt.legend(title=f"Class ({TARGET})", ncol=4, fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir_scatter_pre / f"{feat}_vs_{TARGET}.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 6) Remove rows with negative feature values
    # -------------------------------------------------------------------------
    total_before_neg = len(df_red)
    num_feats = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_red[c])]
    df_red = df_red[(df_red[num_feats] >= 0).all(axis=1)].copy()
    total_after_neg = len(df_red)
    print(f"\nRemoved {total_before_neg - total_after_neg} rows with negative features")

    # -------------------------------------------------------------------------
    # 7) Post-cleaning visualizations
    # -------------------------------------------------------------------------
    out_dir_scatter_post = script_dir.parent / "feature_vs_target_pos_cleaning"
    out_dir_scatter_post.mkdir(parents=True, exist_ok=True)

    df_plot_post = df_red.copy()
    df_plot_post[TARGET] = df_plot_post[TARGET].astype(int)

    for feat in num_feats:
        plt.figure(figsize=(7, 4.5))
        for k in range(MIN_CLASS, MAX_CLASS + 1):
            dsub = df_plot_post[df_plot_post[TARGET] == k]
            if dsub.empty:
                continue
            plt.scatter(
                dsub[feat], dsub[TARGET],
                s=14, alpha=0.7, label=f"{k}",
                color=class_colors[k], edgecolors="none",
            )
        plt.xlabel(feat)
        plt.ylabel(TARGET)
        plt.yticks(range(MIN_CLASS, MAX_CLASS + 1))
        plt.title(f"{feat} vs {TARGET} (post-cleaning)")
        plt.legend(title=f"Class ({TARGET})", ncol=4, fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir_scatter_post / f"{feat}_vs_{TARGET}.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 8) Save cleaned dataset (CSV + Parquet)
    # -------------------------------------------------------------------------
    out_dir_clean = script_dir.parent / "data_cleaned"
    out_dir_clean.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir_clean / "tabelaestabilidade_minUG_cleaned.csv"
    parquet_path = out_dir_clean / "tabelaestabilidade_minUG_cleaned.parquet"

    df_red.to_csv(csv_path, index=False)
    try:
        df_red.to_parquet(parquet_path, index=False)
    except Exception as e:
        print("Warning: failed to save Parquet:", e)

    print("\n=== Files saved ===")
    print("CSV:", csv_path)
    print("Parquet:", parquet_path)


if __name__ == "__main__":
    main()
