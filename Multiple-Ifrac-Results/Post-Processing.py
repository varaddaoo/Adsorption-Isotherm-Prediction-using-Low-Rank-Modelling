#!/usr/bin/env python
# coding: utf-8

# Importing essential modules

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

# Settting default plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = [11, 7]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50

fontdict_t = {'fontsize': 30, 'weight': 'bold', 'ha': 'center'}
fontdict_x = {'fontsize': 23, 'weight': 'bold', 'ha': 'center'}
fontdict_y = {'fontsize': 23, 'weight': 'bold', 'va': 'baseline', 'ha': 'center'}
fontdict_l = {'size': 23, 'weight': 'bold'}


#Ifrac = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#Ifrac = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
Ifrac = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]
nrun  = 10

result = {}

for ifrac in Ifrac:
    with open(f"{ifrac}/5_FFig/Statistical_analysis/{nrun}_result.pkl", "rb") as f:
        result[ifrac] = pickle.load(f)

    assert nrun == len(result[ifrac]), (
        f"Expected independent runs: {nrun}. "
        f"Got: {len(result[ifrac])} for ifrac = {ifrac}"
    )

# all_results is now a list of dictionaries
print(f"✅ Result loaded for all Ifrac each with {nrun} independant runs")

# Test 1 : Get result for Ifrac = 0.4 for all runs
ifrac = 0.4
print(f"\nStatistical summary for ifrac = {ifrac:.2f}")
print(f"Number of independent runs: {len(result[ifrac])}")
print("Contents of all runs:\n")
print(f"{result[ifrac]}")

# Test 2 : Get result for Ifrac = 0.9 for 3rd run
ifrac = 0.9
irun  = 1    # Python is 0-indexed, so 3rd run is index 2
print(f"\nResult for Ifrac = {ifrac:.2f}, Run {irun + 1}:\n")
print(result[ifrac][irun])

# Test 3 : Get the predicted matrix for ifrac = 0.75 for first run
ifrac = 0.75
irun  = 0
print(f"\nPredicted matrix for ifrac = {ifrac:.2f}, Run {irun + 1}:\n")
print(result[ifrac][0]['RMMat_pred'])

# Test4 : Get the best hyperparameters for 3rd run for all Ifrac
irun = 1 
print(f"\nBest hyperparameters for Run {irun + 1} across all Ifrac:\n")
for ifrac in Ifrac:
    best_hp = result[ifrac][irun]["Rbesthp"]
    print(f"Ifrac = {ifrac:.2f}: {best_hp}")

# -----------Testing is complete ------------------

# First use case : R2 vs Ifrac across all runs
# --------------------------
# Compute R² mean and sample std for all three models (for test set)
# --------------------------

R2_means_LB = []
R2_means_BO = []
R2_means_LO = []

R2_stds_LB = []
R2_stds_BO = []
R2_stds_LO = []

for ifrac in Ifrac:
    r2_LB = [run['RTestMat']['R²'] for run in result[ifrac]]       # Latent-bias
    r2_BO = [run['RTestMat_b']['R²'] for run in result[ifrac]]     # Bias-only
    r2_LO = [run['RTestMat_l']['R²'] for run in result[ifrac]]     # Latent-only
    
    R2_means_LB.append(np.mean(r2_LB))
    R2_means_BO.append(np.mean(r2_BO))
    R2_means_LO.append(np.mean(r2_LO))
    
    R2_stds_LB.append(np.std(r2_LB, ddof=1))
    R2_stds_BO.append(np.std(r2_BO, ddof=1))
    R2_stds_LO.append(np.std(r2_LO, ddof=1))

# Convert to numpy arrays for plotting
R2_means_LB = np.array(R2_means_LB)
R2_means_BO = np.array(R2_means_BO)
R2_means_LO = np.array(R2_means_LO)

R2_stds_LB = np.array(R2_stds_LB)
R2_stds_BO = np.array(R2_stds_BO)
R2_stds_LO = np.array(R2_stds_LO)

# --------------------------
# Plot R² vs Ifrac with error bars
# --------------------------

#plt.figure()
plt.figure(figsize=(13,7))

plt.errorbar(Ifrac, R2_means_LB, yerr=R2_stds_LB, fmt='o-', linewidth=3, color='royalblue', markersize=12, capsize=5, label='LB')

plt.errorbar(Ifrac, R2_means_LO, yerr=R2_stds_LO, fmt='^-', linewidth=3, color='forestgreen', markersize=12, capsize=5, label='LO')

plt.errorbar(Ifrac, R2_means_BO, yerr=R2_stds_BO, fmt='s-', linewidth=3, color='darkred', markersize=12, capsize=5, label='BO')

#plt.xlabel("Fraction of Known Entries (Ifrac)", fontdict=fontdict_x)
#plt.ylabel("R² on Test Set", fontdict=fontdict_y)

#plt.title("R² test vs Ifrac", fontdict=fontdict_t)

#plt.xticks(fontsize=18, weight='bold')

plt.xticks(ticks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], labels = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95'], fontdict={'fontsize': 18, 'fontweight': 'bold'})

#plt.xticks(ticks = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], labels = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95'], fontdict={'fontsize': 18, 'fontweight': 'bold'})

plt.yticks(fontsize=18, weight='bold')

plt.legend(fontsize=25, loc='best', frameon=True)

plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.0)

plt.tight_layout
plt.savefig(f"temp_figure/R²_vs_Ifrac.pdf", format='pdf')
plt.close()

#=================================================================================================================#

#=================================================================================================================#

# --------------------------
# Plot R² vs Ifrac for all models
# --------------------------
#plt.figure(figsize=(8,5))
plt.figure(figsize=(13,7))

# Latent-bias (LB)
plt.plot(Ifrac, R2_means_LB, 'o-', color='royalblue', linewidth=2, markersize=15, label='LB')
plt.fill_between(Ifrac, R2_means_LB-R2_stds_LB, R2_means_LB+R2_stds_LB, color='royalblue', alpha=0.2)

# Latent-only (LO)
plt.plot(Ifrac, R2_means_LO, '^-', color='forestgreen', linewidth=2, markersize=15, label='LO')
plt.fill_between(Ifrac, R2_means_LO-R2_stds_LO, R2_means_LO+R2_stds_LO, color='forestgreen', alpha=0.2)

# Bias-only (BO)
plt.plot(Ifrac, R2_means_BO, 's-', color='darkred', linewidth=2, markersize=15, label='BO')
plt.fill_between(Ifrac, R2_means_BO-R2_stds_BO, R2_means_BO+R2_stds_BO, color='darkred', alpha=0.2)

# Labels, title, grid, legend
#plt.xlabel("Fraction of Known Entries (Ifrac)", fontdict=fontdict_x)
#plt.ylabel("R² on Test Set", fontdict=fontdict_y)

#plt.xticks(fontsize=18, weight='bold')

plt.xticks(ticks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], labels = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95'], fontdict={'fontsize': 18, 'fontweight': 'bold'})

plt.yticks(fontsize=18, weight='bold')

#plt.yticks(ticks = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], labels = ['0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95'], fontdict={'fontsize': 18, 'fontweight': 'bold'})

#plt.title("R² test vs Ifrac", fontdict=fontdict_t)

plt.legend(fontsize=26, loc='best', frameon=True)

#plt.grid(True, linestyle='--', alpha=1.0, linewidth=1.0)

plt.tight_layout()
plt.savefig("temp_figure/R2_vs_Ifrac_Seaborn.pdf", format='pdf')
plt.close()

#=====================================================================================================================#

# ==========================================
# ✅ Extract and Plot Hyperparameters (dim & reg)
# ==========================================

dims = []
regs = []
ifracs_expanded = []  # keeps track of which value belongs to which Ifrac

for ifrac in Ifrac:
    assert ifrac in result, f"❌ Missing Ifrac={ifrac} in result dictionary!"
    runs = result[ifrac]

    # --- New assertion: ensure we have exactly nrun entries for this Ifrac ---
    assert len(runs) == nrun, (
        f"❌ For Ifrac={ifrac}, expected {nrun} runs, got {len(runs)}"
    )

    for run_idx, run in enumerate(runs, start=1):
        assert run is not None, f"❌ Run {run_idx} for Ifrac={ifrac} is None!"
        assert 'Rbesthp' in run, f"❌ Missing 'Rbesthp' in run {run_idx} for Ifrac={ifrac}"
        assert isinstance(run['Rbesthp'], dict), f"❌ Rbesthp must be a dict for Ifrac={ifrac}"
        assert 'dim' in run['Rbesthp'], f"❌ Missing 'dim' for Ifrac={ifrac}, run {run_idx}"
        assert 'reg' in run['Rbesthp'], f"❌ Missing 'reg' for Ifrac={ifrac}, run {run_idx}"

        dims.append(run['Rbesthp']['dim'])
        regs.append(run['Rbesthp']['reg'])
        ifracs_expanded.append(ifrac)

# --- Global validation across all Ifrac values ---
expected_total = len(Ifrac) * nrun
assert len(dims) == expected_total, f"❌ Expected {expected_total} dims, got {len(dims)}"
assert len(regs) == expected_total, f"❌ Expected {expected_total} regs, got {len(regs)}"
assert len(ifracs_expanded) == expected_total, "❌ Ifrac tracking mismatch!"

print("✅ All hyperparameters extracted successfully!")
print(f"Total extracted: dims={len(dims)}, regs={len(regs)}")

# --- Convert to numpy arrays for plotting ---
dims = np.array(dims)
regs = np.array(regs)
ifracs_expanded = np.array(ifracs_expanded)

#==================================================================================#

#==================================================================================#

# Create a DataFrame for easier plotting
data = pd.DataFrame({
    "Ifrac": ifracs_expanded,
    "Ifrac_str": [f"{x:.2f}" for x in ifracs_expanded],
    "Dim": dims,
    "Reg": regs
})

# Custom HLS palette with 12 distinct hues
n_colors = 12
colors = sns.color_palette("hls", n_colors)
cmap_hls = LinearSegmentedColormap.from_list("hls_continuous", colors, N=n_colors)

# ✅ Sample the colormap to create discrete color list
unique_ifrac = sorted(data["Ifrac_str"].unique(), key=float)
palette = [cmap_hls(i / (len(unique_ifrac) - 1)) for i in range(len(unique_ifrac))]

#sns.set(style="whitegrid", context="talk")
os.makedirs("temp_figure", exist_ok=True)

# ==========================================
# ✅ Plot 1: dim vs Ifrac
# ==========================================

plt.figure(figsize=(13,7))

ax = sns.swarmplot(data=data, x="Ifrac", y="Dim",hue="Ifrac_str", palette=palette,size=8, alpha=0.85, edgecolor='k', linewidth=1.0)

#plt.xlabel(r"Fraction of known entries ($I_{\mathrm{frac}}$)", fontdict=fontdict_x)
#plt.ylabel(r"Optimal latent dimension ($k_{\mathrm{opt}}$)", fontdict=fontdict_y)

ax.set(xlabel=None, ylabel=None)

plt.xticks(fontsize=25, weight='bold')
plt.yticks(fontsize=25, weight='bold')

#plt.yticks(ticks = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28], labels = ['19', '20', '21', '22', '23', '24', '25', '26', '27', '28'], fontdict={'fontsize': 24, 'fontweight': 'bold'})

#plt.grid(True, linestyle='--', alpha=0.8, linewidth=1.0)

plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left',
           fontsize=22, title=r"Ifrac", title_fontsize=22, frameon=True)

plt.tight_layout()
plt.savefig("temp_figure/dim_vs_Ifrac_swarm.pdf", format='pdf', bbox_inches='tight')
plt.close()

# ==========================================
# ✅ Plot 2: reg vs Ifrac
# ==========================================

plt.figure(figsize=(13,7))

ax = sns.swarmplot(data=data, x="Ifrac", y="Reg",hue="Ifrac_str", palette=palette,size=8, alpha=0.85, edgecolor='k', linewidth=1.0)

#plt.xlabel(r"Fraction of known entries ($I_{\mathrm{frac}}$)", fontdict=fontdict_x)
#plt.ylabel(r"Optimal regularization parameter ($\lambda_{\mathrm{opt}}$)", fontdict=fontdict_y)

ax.set(xlabel=None, ylabel=None)

plt.xticks(fontsize=25, weight='bold')
plt.yticks(fontsize=25, weight='bold')

plt.grid(True, linestyle='--', alpha=0.8, linewidth=1.0)

plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left',
           fontsize=22, title=r"Ifrac", title_fontsize=22, frameon=True)

plt.tight_layout()
plt.savefig("temp_figure/reg_vs_Ifrac_swarm.pdf", format='pdf', bbox_inches='tight')
plt.close()
