#!/usr/bin/env python
# coding: utf-8

# Importing essential modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Settting default plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = [11, 7]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50

fontdict_t = {'fontsize': 24, 'weight': 'bold', 'ha': 'center'}
fontdict_x = {'fontsize': 22, 'weight': 'bold', 'ha': 'center'}
fontdict_y = {'fontsize': 22, 'weight': 'bold', 'va': 'baseline', 'ha': 'center'}
fontdict_l = {'size': 20, 'weight': 'bold'}

def PlotHeatmap(df, Token, irank):

    """
    Description:
    
            Aim is to obtain Pearson correlation matrix.

            This tells us how the columns of material-pressure matrix i.e. uptake values at different pressures at a given temperature are corelated
    
            PlotHeatmap() gives us the values of corelation
    
    Input: 
            df     ==> DataFrame of shape (nmat, npress)

            Token  ==>

            irank  ==>

    Ouput: 
            Peason correlation plots in folder `2_Corr/PlotHeatmap`

    """

    os.makedirs(f'PlotHeatmap/', exist_ok=True)

    plt.figure(figsize=(11,7))

    corr = df.corr(method = 'pearson')
    # https://python-graph-gallery.com/91-customize-seaborn-heatmap/

    colors = sns.color_palette("hls", 4)   # 8 distinct hues
    cmap_hls = LinearSegmentedColormap.from_list("hls_continuous", colors, N=8)
    
    #sns.heatmap(corr, annot=True, annot_kws={"size": 7}, fmt=".2f", cmap=cmap_hls, vmin=-1, vmax=1, linewidths=1, linecolor='black')
    ax = sns.heatmap(corr, cmap=cmap_hls, linewidths=1, linecolor='black', \
        vmin=-1, vmax=1, xticklabels=False, yticklabels=False)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20, width=1, color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

    ax.collections[0].colorbar.outline.set(linewidth=2, edgecolor='black')

    plt.title(f"Pearson Correlation with rank = {irank}", fontdict=fontdict_t)
    plt.xlabel("Pressure (bar)", fontdict=fontdict_x)
    plt.ylabel("Pressure (bar)", fontdict=fontdict_y)

    plt.tight_layout()

    plt.savefig(f"PlotHeatmap/{Token}_{irank}.pdf", format='pdf')

    plt.close()

#plt.rcParams['figure.figsize'] = [16, 8]

X = np.loadtxt('SVD.csv', delimiter=',')

nmat, npress = X.shape

# ---- Plot ORIGINAL MATRIX ----
PlotHeatmap(pd.DataFrame(X), "ORI", "full")

print(f"The shape of X is: {X.shape}")

U, S, VT = np.linalg.svd(X, full_matrices = False)
S = np.diag(S)
tol = 1e-12
rank = np.sum(np.diag(S) > tol)
print(f"Rank of the Original Matrix is : {rank}")

j = 0
for r in range(1, npress + 1):
    # Construct an approximate image
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    print("----------------------------------")
    print(f"The shape of Xapprox is : {Xapprox.shape}")
    print("----------------------------------")
    PlotHeatmap(pd.DataFrame(Xapprox), "Approx", r)


plt.figure()
plt.semilogy(np.diag(S), marker='o', markersize=10, markerfacecolor='white', 
	markeredgecolor='black', linewidth=2)
plt.xlabel('Eigen values', fontdict=fontdict_x)
plt.ylabel('jth Eigen values', fontdict=fontdict_y)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.title('Singular Values', fontdict=fontdict_t)
#plt.grid()
plt.axhline(y=1e2, color='red', linestyle='--', linewidth=2, label=r'$10^2$ threshold')
plt.legend(fontsize=18)
plt.savefig(f'PlotHeatmap/1_Eigen.pdf', bbox_inches='tight', pad_inches=0)

plt.figure()
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)),marker='o', markersize=10, 
	markerfacecolor='white', markeredgecolor='black', linewidth=2)
plt.xlabel('Eigen values', fontdict=fontdict_x)
plt.ylabel('Commulative sum of eigne values', fontdict=fontdict_y)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.title('Singular Values: Cummulative Sum', fontdict=fontdict_t)
#plt.grid()
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label=r'95% Explained Variance')
plt.legend(fontsize=18)

plt.savefig(f'PlotHeatmap/2_Eigen.pdf', bbox_inches='tight', pad_inches=0) 