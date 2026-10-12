#!/usr/bin/env python
# coding: utf-8

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# Settings
ifrac = 0.99
irun = 0                 # 0 = first run
nrun = 2

csv_file = "CoRE2019-N2M-77K.csv"
pkl_file = f"{ifrac}/5_FFig/Statistical_analysis/{nrun}_result.pkl"
out_dir = f"{ifrac}/5_FFig/{irun}/6_Latent_Structural_Correlation"

structural_cols = ["vf", "sa_tot_m2g", "sa_tot_m2cm3", "pld", "lcd"]
structural_labels = [
    r"AVF",
    "GSA\n(m$^2$/g)",
    "VSA\n(m$^2$/cm$^3$)",
    r"PLD ($\AA$)",
    r"LCD ($\AA$)",
]

os.makedirs(out_dir, exist_ok=True)

# Plot settings used in RUNM-17-08-2026.py
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = [11, 7]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50
plt.rcParams["font.size"] = 18

fontdict_x = {"fontsize": 22, "weight": "bold", "ha": "center"}
fontdict_y = {"fontsize": 22, "weight": "bold", "va": "baseline", "ha": "center"}


# Read the saved GLRM results
with open(pkl_file, "rb") as f:
    result = pickle.load(f)

if len(result) != nrun:
    raise ValueError(f"Expected {nrun} runs in {pkl_file}, found {len(result)}")
if not 0 <= irun < len(result):
    raise IndexError(f"irun={irun} is outside the available run indices 0 to {len(result)-1}")

run = result[irun]
RM = np.asarray(run["RM"], dtype=float)

# RM contains the material latent factors and the material bias in its last row.
# Remove the bias row; columns then correspond to materials.
MAct = RM[:-1, :].T
k = MAct.shape[1]

print(f"Ifrac: {ifrac}")
print(f"Run: {irun + 1}")
print(f"Material latent-factor matrix: {MAct.shape}")

# Read the structural descriptors. The material order must match the order used
# when the GLRM result was generated.
df = pd.read_csv(csv_file)
missing_cols = [col for col in structural_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing structural columns in {csv_file}: {missing_cols}")

structural_df = df[structural_cols].copy()
if MAct.shape[0] != len(structural_df):
    raise ValueError(
        f"Material count mismatch: GLRM={MAct.shape[0]}, CSV={len(structural_df)}"
    )

X_struct = structural_df.to_numpy(dtype=float)
print(f"Structural-property matrix: {X_struct.shape}")
print(f"Number of latent dimensions: {k}")


# Calculate Pearson and Spearman correlations, ignoring missing/non-finite values.
def get_correlations(latent_matrix, descriptor_matrix, method):
    corr = np.full((latent_matrix.shape[1], descriptor_matrix.shape[1]), np.nan)
    pval = np.full_like(corr, np.nan)

    for i in range(latent_matrix.shape[1]):
        for j in range(descriptor_matrix.shape[1]):
            x = latent_matrix[:, i]
            y = descriptor_matrix[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
                continue

            if method == "pearson":
                corr[i, j], pval[i, j] = pearsonr(x, y)
            else:
                corr[i, j], pval[i, j] = spearmanr(x, y)

    latent_names = [f"Latent {i + 1}" for i in range(latent_matrix.shape[1])]
    return (
        pd.DataFrame(corr, index=latent_names, columns=structural_cols),
        pd.DataFrame(pval, index=latent_names, columns=structural_cols),
    )


pearson_df, pearson_pval_df = get_correlations(MAct, X_struct, "pearson")
spearman_df, spearman_pval_df = get_correlations(MAct, X_struct, "spearman")

print("\nPearson correlation")
print(pearson_df.to_string(float_format=lambda x: f"{x:.3f}"))
print("\nSpearman correlation")
print(spearman_df.to_string(float_format=lambda x: f"{x:.3f}"))

# Save correlation tables
pearson_df.to_csv(f"{out_dir}/Pearson_ifrac_{ifrac}_run_{irun + 1}.csv")
pearson_pval_df.to_csv(f"{out_dir}/Pearson_pvalues_ifrac_{ifrac}_run_{irun + 1}.csv")
spearman_df.to_csv(f"{out_dir}/Spearman_ifrac_{ifrac}_run_{irun + 1}.csv")
spearman_pval_df.to_csv(f"{out_dir}/Spearman_pvalues_ifrac_{ifrac}_run_{irun + 1}.csv")


# Plot the correlation matrix using the same font and axis styling as the main script.
def plot_correlation(corr_df, method, filename):
    plot_df = corr_df.copy()
    plot_df.columns = structural_labels

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 15, "weight": "bold"},
        linewidths=1,
        linecolor="black",
        cbar_kws={"label": method},
    )

    ax.set_xlabel("Structural descriptor", fontsize=19, fontweight="bold", labelpad=10)
    ax.set_ylabel("Material latent dimension", fontsize=19, fontweight="bold", labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=14, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, fontweight="bold")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15, width=1, color="black")
    plt.setp(cbar.ax.get_yticklabels(), fontweight="bold")
    cbar.set_label(method, fontsize=18, fontweight="bold", labelpad=12)
    cbar.outline.set(linewidth=1.5, edgecolor="black")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}", format="pdf", bbox_inches="tight")
    plt.savefig(f"{out_dir}/{filename.replace('.pdf', '.png')}", dpi=300, bbox_inches="tight")
    plt.close(fig)


plot_correlation(pearson_df, "Pearson r", f"Latent_Structural_Pearson_ifrac_{ifrac}_run_{irun + 1}.pdf")
plot_correlation(spearman_df, "Spearman $\\rho$", f"Latent_Structural_Spearman_ifrac_{ifrac}_run_{irun + 1}.pdf")

# Summarize the strongest absolute Pearson association for each latent dimension.
summary = []
for latent in pearson_df.index:
    correlations = pearson_df.loc[latent].dropna()
    if correlations.empty:
        continue
    property_name = correlations.abs().idxmax()
    r = correlations[property_name]
    summary.append({
        "Latent_dimension": latent,
        "Structural_descriptor": property_name,
        "Pearson_r": r,
        "Abs_Pearson_r": abs(r),
        "p_value": pearson_pval_df.loc[latent, property_name],
    })

summary_df = pd.DataFrame(summary).sort_values("Abs_Pearson_r", ascending=False)
summary_df.to_csv(f"{out_dir}/Strongest_Pearson_association_ifrac_{ifrac}_run_{irun + 1}.csv", index=False)
print("\nStrongest Pearson association for each latent dimension")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Correlations among the structural descriptors provide context for interpreting
# latent-factor/descriptor correlations.
structural_corr = structural_df.corr(method="spearman")
structural_corr.columns = structural_labels
structural_corr.index = structural_labels
structural_corr.to_csv(f"{out_dir}/Structural_descriptor_Spearman_correlation.csv")

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    structural_corr,
    ax=ax,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 15, "weight": "bold"},
    linewidths=1,
    linecolor="black",
    square=True,
    cbar_kws={"label": "Spearman $\\rho$"},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14, fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, fontweight="bold")
ax.set_xlabel("Structural descriptor", fontdict=fontdict_x, labelpad=12)
ax.set_ylabel("Structural descriptor", fontdict=fontdict_y, labelpad=12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14, width=1, color="black")
plt.setp(cbar.ax.get_yticklabels(), fontweight="bold")
cbar.set_label("Spearman $\\rho$", fontsize=17, fontweight="bold", labelpad=12)
cbar.outline.set(linewidth=1.5, edgecolor="black")
plt.tight_layout()
plt.savefig(f"{out_dir}/Structural_descriptor_correlation.pdf", format="pdf", bbox_inches="tight")
plt.savefig(f"{out_dir}/Structural_descriptor_correlation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nAnalysis complete. Results saved in: {out_dir}")
