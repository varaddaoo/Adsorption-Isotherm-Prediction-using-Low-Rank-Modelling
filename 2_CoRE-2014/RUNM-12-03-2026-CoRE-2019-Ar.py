#!/usr/bin/env python
# coding: utf-8

# Importing essential modules

import numpy as np
import pandas as pd
import matplotlib
import os
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

from itertools import product
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pyglrm import glrm, QuadLoss, QuadReg, ZeroReg
from itertools import product
from tqdm import tqdm
from typing import List, Tuple, Union, Optional
from scipy.stats import spearmanr
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

# Settting default plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = [11, 7]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50

fontdict_t = {'fontsize': 20, 'weight': 'bold', 'ha': 'center'}
fontdict_x = {'fontsize': 22, 'weight': 'bold', 'ha': 'center'}
fontdict_y = {'fontsize': 22, 'weight': 'bold', 'va': 'baseline', 'ha': 'center'}
fontdict_l = {'size': 20, 'weight': 'bold'}

# Setting random seed for reproducibility

np.random.seed(76293)
random.seed(76293)
SEED = 76293

# pressure contains the pressure points on the adsorption isotherm which are to be considered to be included in this study

#pressure = [
#    "0.00000001bar",
#    "0.00000010bar",
#    "0.00000100bar",
#    "0.00001000bar",
#    "0.00010000bar",
#    "0.00100000bar",
#    "0.01000000bar",
#    "0.02000000bar",
#    "0.04000000bar",
#    "0.06000000bar",
#    "0.08000000bar",
#    "0.10000000bar",
#    "0.12000000bar",
#    "0.14000000bar",
#    "0.16000000bar",
#    "0.18000000bar",
#    "0.20000000bar",
#    "0.24000000bar",
#    "0.28000000bar",
#    "0.30000000bar",
#    "0.32000000bar",
#    "0.36000000bar",
#    "0.40000000bar",
#    "0.50000000bar",
#    "0.60000000bar",
#    "0.70000000bar",
#    "0.80000000bar",
#    "0.90000000bar",
#    "1.00000000bar"
#]

pressure = [
    "1e-8",
    "1e-7",
    "1e-6",
    "1e-5",
    "1e-4",
    "1e-3",
    "1e-2",
    "2e-2",
    "4e-2",
    "6e-2",
    "8e-2",
    "1e-1",
    "1.2e-1",
    "1.4e-1",
    "1.6e-1",
    "1.8e-1",
    "2e-1",
    "2.4e-1",
    "2.8e-1",
    "3e-1",
    "3.2e-1",
    "3.6e-1",
    "4e-1",
    "5e-1",
    "6e-1",
    "7e-1",
    "8e-1",
    "9e-1",
    "1e0"
]

def PlotPCABefore(df, Ipressure):

    """
    Description:

            Here the aim is to do PCA on material-pressure matrix and pressure-material matrix in their base form

                For material-pressure matrix case: 1.) Each material is expressed in a basis set formed by pressure values.
                                                   2.) PCA projections are done and they are visualized in a scatter with colormap indicating pressure value.

                For pressure-material matrix case: 1.) Each property is expressed in a basis set formed by material values.
                                                   2.) PCA projections are done and they are visualized in the form of arrows.

                If the basis sets are truely in the same subspace then the materials havin high property value in the scatter plot would congrugate in the direction of the arraw of the respective property in arrow plot

                `Check this`: Intuition tells us that given a material and its one property (say Ar uptake at 0.1 bar), we should be able to predict other property (say Ar uptake at 0.2 bar)

    Input:
            df     ==> DataFrame of shape (nmat, npress)

    Ouput:
            PCA plots in folder `1_PCA/1_Before`

    """

    os.makedirs('PCA_Before/', exist_ok=True)

    # Material Case

    [inmat, inpress] = df.shape

	# Proprety vector
	# Property Case
    dg = df.T

    # Material Case
    norm       = MinMaxScaler()
    Mat_norm   = norm.fit_transform(df) # after norm the datatype changes from dataframe to numpy array

    # Property Case
    norm_T     = MinMaxScaler()
    Press_norm = norm_T.fit_transform(dg) # after norm the datatype changes from dataframe to numpy array

    # Material Case
    pca        = PCA(n_components=2)
    Mat_pca    = pca.fit_transform(Mat_norm)

    # Property Case
    pca_T      = PCA(n_components=2)
    Press_pca  = pca_T.fit_transform(Press_norm)

    # Material Case
    cmap = get_cmap('rainbow', 7)

	# Property Case
    colors = plt.cm.get_cmap('hsv', inpress)

    # Property Case
    norm_pca = np.linalg.norm(Press_pca, axis=1)  # Find the magnitude of each vector
    normalized_pca = Press_pca / norm_pca[:, np.newaxis]  # Normalize each vector by its magnitude

    for i in range(inpress):

        plt.figure(figsize = (11,7))

        # Material Case

        sc = plt.scatter(Mat_pca[:,0], Mat_pca[:,1], s=30, c=Mat_norm[:,i], cmap = cmap, edgecolor='black', linewidth=1, alpha=0.75)

        # Plotting X and Y axis
        plt.axhline(0, color='black', linewidth=2)  # x-axis
        plt.axvline(0, color='black', linewidth=2)  # y-axis

        val = float(Ipressure[i])
        mantissa, exponent = f"{val:.1e}".split("e")
        exponent = int(exponent)

        cbar = plt.colorbar(sc)
        #cbar.set_label('\n' + rf'$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{exponent}\ bar}}$', fontsize=23)
        cbar.set_label('\n' + rf'$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ }}$',fontsize=23)
        cbar.ax.tick_params(labelsize=23, width=1, color='black')
        plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')
        #cbar.ax.tick_params(axis='both', which='major', labelweight='bold')

        cbar.set_ticks([0.000, 0.143, 0.286, 0.428, 0.571, 0.714, 0.857, 1.000])
        cbar.set_ticklabels(['0.00', '0.14', '0.28', '0.42', '0.57', '0.71', '0.86', '1.00'])

        plt.xlabel('PCA dimention 1', fontdict=fontdict_x)
        plt.ylabel('PCA dimention 2', fontdict=fontdict_y)
        #plt.title('PCA Material-Pressure Matrix (100% known)', fontdict=fontdict_t)

        plt.xticks(ticks = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], labels = ['-0.5', '0.0', '0.5', '1.0', '1.5', '2.0', '2.5'], fontdict={'fontsize': 23, 'fontweight': 'bold'})
        plt.yticks(ticks = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75], labels = ['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00', '1.25', '1.50', '1.75'], fontdict={'fontsize': 23, 'fontweight': 'bold'})

        #plt.legend(fontsize=16, loc='best', frameon = True)

        # Property Case

        # Plotting X and Y axis
        plt.axhline(0, color='black', linewidth=2)  # x-axis
        plt.axvline(0, color='black', linewidth=2)  # y-axis

        if normalized_pca[i,0] < 0.0:
        	normalized_pca[i,0] = normalized_pca[i,0] * 0.5
        	normalized_pca[i,1] = normalized_pca[i,1] * 0.5
        elif normalized_pca[i,0] > 0.0:
        	normalized_pca[i,0] = normalized_pca[i,0] * 2.5
        	normalized_pca[i,1] = normalized_pca[i,1] * 2.5

        plt.quiver(0.0, 0.0, normalized_pca[i,0], normalized_pca[i,1], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)

        plt.xlabel('PCA dimension 1', fontdict=fontdict_x)
        plt.ylabel('PCA dimension 2', fontdict=fontdict_y)

        #legend_patch = mpatches.Patch(color=colors(i), label=rf"$\mathbf{{Ar\ Uptake\ at\ Pressure\ {Ipressure[i]} bar}}$")

        legend_patch = mpatches.Patch(color=colors(i), label=rf"$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ bar}}$")
        
        plt.legend(handles=[legend_patch], fontsize=18, loc='best', frameon=True)
        #plt.title('PCA Pressure-Material Matrix (100% known)', fontdict=fontdict_t)

        plt.savefig(f"PCA_Before/{i}_Material_Pressure_Matix.pdf", format='pdf')
        plt.close()

def PlotHeatmap(df, Token, Ifrac, Iirun):

    """
    Description:
    
            Aim is to obtain Pearson correlation matrix.

            This tells us how the columns of material-pressure matrix i.e. uptake values at different pressures at a given temperature are corelated

            PlotHeatmap() gives us the values of corelation
    
    Input: 
            df     ==> DataFrame of shape (nmat, npress)

            Token  ==> Identifier for saving the plot (e.g., LB, LO, BO)

            Ifrac  ==> Fraction of known entries

            Iirun  ==> Run index

    Ouput: 
            Peason correlation plots in folder `2_Corr/PlotHeatmap/`

    """

    os.makedirs(f'2_Corr/PlotHeatmap/{Iirun}', exist_ok=True)

    plt.figure(figsize=(11,7))

    corr = df.corr(method = 'pearson')
    # https://python-graph-gallery.com/91-customize-seaborn-heatmap/

    mask = np.triu(np.ones_like(corr, dtype=bool))

    colors = sns.color_palette("hls", 4)   # 8 distinct hues
    cmap_hls = LinearSegmentedColormap.from_list("hls_continuous", colors, N=8)

    #sns.heatmap(corr, annot=True, annot_kws={"size": 7}, fmt=".2f", cmap=cmap_hls, vmin=-1, vmax=1, linewidths=1, linecolor='black')
    ax = sns.heatmap(corr, mask=mask, cmap=cmap_hls, linewidths=1, linecolor='black', \
        vmin=-1, vmax=1, xticklabels=False, yticklabels=False)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20, width=1, color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

    ax.collections[0].colorbar.outline.set(linewidth=2, edgecolor='black')

    #plt.title(f"Pearson Correlation ({Ifrac*100:.0f}% known)", fontdict=fontdict_t)
    plt.xlabel("Pressure index (pressure in bar)", fontdict=fontdict_x)
    plt.ylabel("Pressure index (pressure in bar)", fontdict=fontdict_y)
    
    plt.tight_layout()
    
    plt.savefig(f"2_Corr/PlotHeatmap/{Iirun}/{Token}.pdf", format='pdf')
    
    plt.close()

def PearsonPlot(Idf, iprop):

    """
    Description:

            Aim is to visualize the Pearson correlation amongust different columns of material-pressure matrix.   
            
            PlotHeatmap() gives us the values of corelation where as PearsonPlot() helps in visualization of these correlations.

    Input:

            Idf     ==>  Amp (material-pressure matrix) in dataframe format with a shape of (nmat, npress) 

            iprop   ==>  pressure (pressure headings) in the form of list with a size of (1, npress)

    Output:

            Pearson correlation plots in the folder 2_Corr/PearsonPlot

    """

    os.makedirs('2_Corr/PearsonPlot', exist_ok=True)

    [inmat, inpress] = Idf.shape

    for i in range(inpress): # range(inpress)
        for j in range(i, inpress):

            plt.figure(figsize=(11,7))

            X = Idf.iloc[:, i]
            Y = Idf.iloc[:, j]

            iCOR = X.corr(Y)

            #plt.scatter(X, Y, color='green', label=f"{iprop[i]} vs {iprop[j]}", alpha=0.6, edgecolor='black', s=80)
            plt.scatter(X, Y, color='blue', label=f"{iprop[i]} vs {iprop[j]}", alpha=0.6, s=20)
            
            plt.plot([min(min(X), min(Y)), max(max(X), max(Y))],[min(min(X), min(Y)), max(max(X), max(Y))],color='gray', linestyle='--', label='Perfect Prediction')

            plt.xticks(fontsize=26,fontweight='bold')
            plt.yticks(fontsize=26,fontweight='bold')

            plt.xlabel(f"Uptake of Ar at {iprop[i]}", fontdict=fontdict_x)
            plt.ylabel(f"Uptake of Ar at {iprop[j]}", fontdict=fontdict_y)
            plt.title(f"Pearson Correlation : {iCOR:.4f}", fontdict=fontdict_t)
            #plt.legend(loc='upper right', fontsize=20)
            plt.tight_layout()
            plt.savefig(f"2_Corr/PearsonPlot/{i+1}_{iprop[j]}_vs_{iprop[i]}.pdf", format='pdf')
            plt.close()

def PlotIsotherm(Idf, Imof, Iaxis):

    """
    Description:

            Aim is to visualize the adsorption isotherm for each material in the dataset.
            The semilog case clearly shows S-shaped isothem in low pressure regime

    Input: 
            Idf     ==>    DataFrame of shape (nmat, npress).   

            Imof    ==>   Names of all the materials whose isothem has to be plotted. 
                    It is a list of mof_names with a size of (nmat, 1).

            Iaxis   ==>  if Iaxis = SEMILOG, then X axis of isotherm is in log scale and Y axis in linear scale.
                         if Iaxis = LINIEAR, then both X and Y axis of the isotherm is in linear scale. 

    Output:
            Adsorption isothem plots in the folders `3_Isotherm/Semilog` for SEMILOG case and `3_Isotherm/Linear` for LINEAR case

    """

    if Iaxis=='SEMILOG':
        os.makedirs('3_Isotherm/Semilog', exist_ok=True)
    elif Iaxis=='LINEAR':
        os.makedirs('3_Isotherm/Linear', exist_ok=True)
    else:
        raise ValueError(f"\033[91mWrong value of axis\033[0m")

    [inmat, inpress] = Idf.shape

    for i in range(inmat): # range(inmat)
        
        plt.figure(figsize=(11,7))

        if Iaxis=='SEMILOG':
            plt.xscale('log')
        else:
            plt.xscale('linear')

        X = np.array([0.00000001,0.00000010,0.00000100,0.00001000,0.00010000,0.00100000,
            0.01000000,0.02000000,0.04000000,0.06000000,0.08000000,0.10000000,
            0.12000000,0.14000000,0.16000000,0.18000000,0.20000000,0.24000000,
            0.28000000,0.30000000,0.32000000,0.36000000,0.40000000,0.50000000,
            0.60000000,0.70000000,0.80000000,0.90000000,1.00000000])

        X_label = [f"{x:.0e}" for x in X]
        
        plt.scatter(X, Idf.iloc[i, :], s=100, marker='*', facecolors='none', edgecolors='blue', linewidth=1.5, alpha=1.0)
        
        plt.xlabel('Pressure (bar)', fontdict=fontdict_x)
        plt.ylabel('Argon uptake (mol/kg)', fontdict=fontdict_y)

        plt.title(f'Adsorption isotherm at 87 K for {Imof[i]}', fontdict=fontdict_t)
        
        #plt.xticks(ticks = X, labels = X_label, rotation=90, fontdict={'fontsize': 23, 'fontweight': 'bold'})
        plt.xticks(fontsize=20) # fontweight='bold'
        plt.yticks(fontsize=20) # fontweight='bold'

        plt.tight_layout()

        if Iaxis=='SEMILOG':
            plt.savefig(f"3_Isotherm/Semilog/{i}_{Imof[i]}_Iso.pdf", format='pdf')
        else:
            plt.savefig(f"3_Isotherm/Linear/{i}_{Imof[i]}_Iso.pdf", format='pdf')

        plt.close()

def standardize_matrix(IMat, Iaxis):

    """

    Description:

            Standardize the input matrix along the given axis, handling NaNs.
            This is required becasue we would be working with standardized matrix for model development and calculating prediction error.
    
    Input:

            IMat  ==> Input matrix to be standardized (may contain NaNs)

            Iaxis ==> Axis along which standardization is done

                      1 -> row-wise standardize
                      0 -> column-wise standardize

    Output:

            RMat  ==> Standardized matrix (NaNs preserved)

            Rmean ==> Mean vector before standardization

            Rstd  ==> std  vector before standardization

    """

    start_time = time.time()

    Rmean = np.nanmean(IMat, axis=Iaxis, keepdims=True)
    Rstd  = np.nanstd(IMat, axis=Iaxis, keepdims=True)
    
    Rstd[Rstd == 0] = 1.0  # avoid division by zero

    # Standardize, preserving NaNs
    RMat = (IMat - Rmean) / Rstd
    RMat[np.isnan(IMat)] = np.nan  # ensure NaNs remain

    # Post-checks
    mean_after = np.nanmean(RMat, axis=Iaxis)
    std_after  = np.nanstd(RMat, axis=Iaxis)

    # Checking where mean is non zero
    mean_mask   = ~np.isclose(mean_after, 0.0, atol=1e-7)
    std_mask    = ~np.isclose(std_after, 1.0, atol=1e-7)

    if np.any(mean_mask):
        bad_rows_mean  = np.where(mean_mask)[0]
        bad_idx_mean   = [(r, c) for r in bad_rows_mean for c in range(RMat.shape[1])]
        raise AssertionError(f"\033[91mMean not zero in rows {bad_rows_mean}, at index {bad_idx_mean}, with value of {mean_after[mean_mask]}\033[0m")

    if np.any(std_mask):
        bad_rows_std   = np.where(std_mask)[0]
        bad_idx_std    = [(r, c) for r in bad_rows_std for c in range(RMat.shape[1])]
        raise AssertionError(f"\033[91mSTD not unity in rows {bad_rows_std}, at index {bad_idx_std}, with value of {std_after[std_mask]}\033[0m")

    if not np.allclose(mean_after, 0, atol=1e-7):
        bad_rows = np.where(~np.isclose(mean_after, 0, atol=1e-7))[0]
        raise AssertionError(f"\033[91mMean not zero in rows/cols {bad_rows}, values: {mean_after[bad_rows]}\033[0m")
    if not np.allclose(std_after, 1, atol=1e-7):
        bad_rows = np.where(~np.isclose(std_after, 1, atol=1e-7))[0]
        raise AssertionError(f"\033[91mSTD not unity in rows/cols {bad_rows}, values: {std_after[bad_rows]}\033[0m")
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"\n\n\tTime taken to run standardize_matrix is : {elapsed_time:.4f} minutes")

    return Rmean, Rstd, RMat

def unstandardize_matrix(IMat, Imean, Istd):
    
    """
    WARNING !!!! 

    The logic in this function may be erroneous. Kindly check before use 

    Description:

        Reverse the standardization process and reconstruct the original matrix.

    Input:

        IMat  ==> Standardized matrix (NaNs may be present)
        Imean ==> Mean vector used for standardization (shape must be broadcastable to RMat)
        Istd  ==> Standard deviation vector used for standardization (same shape rule applies)

    Output:

        RMat  ==> Unstandardized/original-scale matrix (NaNs preserved)
    
    """

    # Sanity checks
    if IMat.shape != Imean.shape and Imean.size != 1 and not np.allclose(np.shape(Imean), np.shape(Istd)):
        raise ValueError("\033[91mShapes of IMat, Imean, and Istd are not broadcastable.\033[0m")

    # Reverse the standardization
    RMat = (IMat * Istd) + Imean

    # Preserve NaN positions
    RMat[np.isnan(IMat)] = np.nan

    return RMat

def PropMatMissing(Ifrac, IMat, Inmat, Inpress, Isample, Ifrac_col=None, Icol=None, Ifrac_row=None, Irow=None):

    """

    Description:

            Generate a Pressure-material matrix with missing entries for model training.

    Input:
            
            Ifrac       ==> Fraction of entries in the Pressure-material matrix that should be known (e.g., 0.4 means 40% known, 60% missing)
            
            IMat        ==> Fully observed Pressure-material matrix (unstandardized).Must be unstandardized so that missing entries can be removed safely; standardization is applied later using only known entries.
            
            Inmat       ==> The no of materials. Equal to the columns of Pressure-material matrix
            
            Inpress     ==> The no of pressure points on adsorption isotherm. Equal to rows of Pressure-material matrix
            
            Isample     ==> Sampling strategy ("Random" or "Column")
            
            Ifrac_col   ==> Fraction of known entries in the special column (required if Isample=="Column")
            
            Icol        ==> Index of the special column (required if Isample=="Column")

    Output:
            
            RMat        ==> Pressure-material matrix with missing entries
            
            idx_known   ==> Indices of the known entries in RMat
            
            idx_unknown ==> Indices of the missing entries in RMat

    """

    start_time = time.time()

    assert 0 < Ifrac <= 1, "\033[91mIfrac must be between 0 and 1\033[0m"

    assert IMat.shape == (Inpress, Inmat), \
            f"\033[91mWrong Pressure-material matrix is supplied to PropMatMissing. Expected shape was ({Inpress, Inmat}) got {IMat.shape}\033[0m"

    elem_known      = int(np.floor(Ifrac * Inpress * Inmat))
    elem_unknown    = int(np.floor((1 - Ifrac) * Inpress * Inmat))

    assert abs((elem_known + elem_unknown) - (Inpress * Inmat)) <= 1, \
            f"\033[91mMismatch in known and unknown entries of Pressure-material matrix. Expected ({Inpress * Inmat}) got ({elem_known + elem_unknown})\033[0m"

    assert elem_known >= 0 and elem_unknown >= 0, "\033[91mKnown or unknown entries cannot be negative\033[0m"

    all_idx = list(product(range(Inpress), range(Inmat)))

    if Isample == "Random":

        idx_known   = random.sample(all_idx, elem_known) # Randomly select ids of ${elem_known} number of entries of Pressure-material matrix

    elif Isample == "Column":

        assert Ifrac_col is not None and Icol is not None, \
            "\033[91mIfrac_col and Icol must be provided for column sampling\033[0m"

        assert 0 < Ifrac_col <= 1, "\033[91mIfrac_col must be between 0 and 1\033[0m"

        # Known entries in the target column
        elem_Icol_known = int(np.floor(Ifrac_col * Inpress))
        Icol_rows = list(range(Inpress))
        known_Icol_rows = random.sample(Icol_rows, elem_Icol_known) # This is going to give me row no only. For example [2,5,7]. I need column number also like [(2,3), (5,3), (7,3)]
        
        idx_known_Icol  = []
        for iidx in known_Icol_rows:
            idx_known_Icol.append((iidx, Icol)) # This will give me [(2,3), (5,3), (7,3)]

        # Remaining known entries to satisfy overall Ifrac
        remaining_known = elem_known - elem_Icol_known
        if remaining_known < 0:
            raise ValueError("\033[91mIfrac_column is too large for the given Ifrac\033[0m")

        # Exclude target column from remaining pool
        remaining_idx_pool = [idx for idx in all_idx if idx[1] != Icol]
        idx_known_rest = random.sample(remaining_idx_pool, remaining_known)

        # Combine known entries
        idx_known = idx_known_Icol + idx_known_rest

    elif Isample == "Row":

        assert Ifrac_row is not None and Irow is not None, \
            "\033[91mIfrac_row and Irow must be provided for row sampling\033[0m"
        assert 0 < Ifrac_row <= 1, "\033[91mIfrac_row must be between 0 and 1\033[0m"

        # Number of known entries in the target row
        elem_row_known = int(np.floor(Ifrac_row * Inmat))

        # Select known entries in the target row
        row_cols = list(range(Inmat))
        known_row_cols = random.sample(row_cols, elem_row_known)
        idx_known_row = [(Irow, c) for c in known_row_cols]

        # Remaining known entries to satisfy overall Ifrac
        remaining_known = elem_known - elem_row_known
        if remaining_known < 0:
            raise ValueError("\033[91mIfrac_row is too large for the given Ifrac\033[0m")

        # Exclude target row from remaining pool
        remaining_idx_pool = [idx for idx in all_idx if idx[0] != Irow]
        idx_known_rest = random.sample(remaining_idx_pool, remaining_known)

        # Combine known entries
        idx_known = idx_known_row + idx_known_rest

    else:
        raise ValueError(f"\033[91mUnknown sampling method: {Isample}\033[0m")

    # Unknown entries
    idx_unknown = [] # ids of remaining or missing entries of Pressure-material matrix

    for iidx in all_idx: # looping through all the ids of Pressure-material matrix
        if iidx not in idx_known: # Finding ids that are not in known category
            idx_unknown.append(iidx) # ids that are not in idx_known should be in idx_unknown
    assert len(idx_known) + len(idx_unknown) == len(all_idx),\
            f"\033[91mMismatch in known and unknown indices. Expected {len(all_idx)} got {len(idx_known) + len(idx_unknown)}\033[0m"

    # Initialize RMat with NaNs
    # All entries are missing initially
    RMat = np.full((Inpress, Inmat), np.nan)

    # Fill in the known entries with their true (unstandardized) value
    print(f"\tFilling the known entries with their true (unstandardized) value")
    
    for ipress, imat in idx_known:
        RMat[ipress, imat] = IMat[ipress, imat]
        assert np.isclose(RMat[ipress, imat], IMat[ipress, imat]),\
                f"\033[91mMismatch in value at index ({ipress}, {imat}): RMat={RMat[ipress, imat]}, IMat={IMat[ipress, imat]}\033[0m"

    # Cross-check to ensure Ifrac is not violated
    obs_Ifrac = np.count_nonzero(~np.isnan(RMat)) / RMat.size

    assert np.isclose(Ifrac, obs_Ifrac, atol = 1e-5),\
            f"\033[91mMismatch in Overall known fraction: expected {Ifrac}, got {obs_Ifrac}\033[0m"

    # Cross check to ensure Ifrac_col is not violated
    if Isample == "Column":
        Icol_known = np.count_nonzero(~np.isnan(RMat[:, Icol]))
        Icol_frac = Icol_known / Inpress
        
        assert np.isclose(Icol_frac, Ifrac_col, atol=0.05),\
                f"\033[91mMismatch in column fraction. Expected {Ifrac_col} got {Icol_frac:.3f}\033[0m"
    
    # Cross check to ensure all the entries in known indices are filled in RMat
    for ipress, imat in idx_known:
        assert not np.isnan(RMat[ipress, imat]), f"\033[91mMismatch in entries of known indices: Known entry ({ipress},{imat}) is NaN\033[0m"

    # Cross check to ensure all the entries in unknown indices are NaN
    for ipress, imat in idx_unknown:
        assert np.isnan(RMat[ipress, imat]), f"\033[91mMismatch in entries of unknown indices ({ipress},{imat}) is not NaN\033[0m"

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"\tTime taken to run PropMatMissing is : {elapsed_time:.4f} minutes")

    return RMat, idx_known, idx_unknown

def fit_lrm(IMat, Inpress, Inmat, Ihpara, Iidx_known, IP0=None, IM0=None, Ioffval=True):
    
    """

    Description:
    
            Fit a low-rank model (GLRM) to an incomplete Pressure-material matrix using pyglrm.
    
    Input:

            X_Mat (ndarray)      ==> Incomplete matrix of shape (n_p, n_m) with np.nan for missing entries
            
            Ihpara               ==> Contains dim (rank) and reg (regularization strength)
            
            Iidx_known (list)    ==> List of (p, m) tuples indicating observed entries
            
            P0 (ndarray or None) ==> Optional initial guess for P of shape (dim + 1, npress).
            
            M0 (ndarray or None) ==> Optional initial guess for M of shape (dim + 1, nmat).offval offset value

    Output:
            
            RP (ndarray)         ==> Latent Pressure vectors of shape (dim + 1, npress)
            
            RM (ndarray)         ==> Latent material vectors of shape (dim + 1, nmat)
            
            Rmodel               ==> Fitted GLRM model
            
            Rch                  ==> Convergence history

    """

    start_time = time.time()

    assert IMat.shape == (Inpress, Inmat), f"\033[91mIMat shape mismatch: expected ({Inpress}, {Inmat}), got {IMat.shape}\033[0m"

    if Ioffval == True:

        # Initialize if not provided
        if IP0 is None:
            IP0 = np.random.randn(Ihpara["dim"]+1, Inpress)
        if IM0 is None:
            IM0 = np.random.randn(Ihpara["dim"]+1, Inmat)

        Rmodel = glrm(
            losses=QuadLoss(),
            rx=QuadReg(Ihpara["reg"] / Inmat),
            ry=QuadReg(Ihpara["reg"] / Inpress),
            n_components=Ihpara["dim"] + 1,  # 1 is added to include the bias term
            X=IM0,
            Y=IP0,
            offset=Ioffval
            )
        
        print(f"\tOffset: {Rmodel.offset}")
        
        assert Rmodel.offset is True, f"\033[91mThe expected offset to the glrm model is True, but got {Rmodel.offset}\033[0m"

        Rmodel.set_hyperparams(obs=Iidx_known)

        print("\n\t============================================================================================\n")
        Rch = Rmodel.fit(inputs = IMat)

        RP = Rmodel.X  # (dim + 1, Inmat) material vectors
        RM = Rmodel.Y  # (dim + 1, Inpress) Pressure vectors

        #print(RP.shape)
        #print(RM.shape)

        assert RP.shape == (Ihpara['dim'] + 1, Inpress), f"\033[91mExpected shape of RP is {(Ihpara['dim'] + 1, Inpress)}, but got {RP.shape}\033[0m"
        assert RM.shape == (Ihpara['dim'] + 1, Inmat), f"\033[91mExpected RM shape {(Ihpara['dim'] + 1, Inmat)}, but got {RM.shape}\033[0m"

        # Internal consistency check: impute vs manual P.T @ M
        IMat_pred_model = Rmodel.impute()

        # Recreating the matrix IMat by using the decomposed matrix RM and RP. Note these contains bias in them
        IMat_pred_manual = RP.T @ RM # shape (Inpress * Ihpara.dim + 1) @ (Ihpara.dim + 1, Inmat) = (Inpress * Inmat)

        # Note that last row of RP has does not contain any material bias and is just row of 1
        P_act = RP[:-1, :] # because RP.shape = Ihpara.dim + 1 * Inpress but in the X = P.T + M P.shape = Ihpara.dim * Inpress that is we need to exclude the last row of RP

        # Note that the last row of RM contains the material bias
        M_act = RM[:-1, :] # because RM.shape = Ihpara.dim + 1 * Inpress but in the X = P.T + M M.shape = Ihpara.dim * Inmat that is we need to exclude the last row of RM

        mat_bias = RM[-1, :].reshape(-1, 1)  # shape (Inmat, 1) This is material bias
    
        #print(f"\n\t===========================================================")
        #print(f"\n\tLatent-vec-Pressure shape without bias term is : {P_act.shape}")
        #print(f"\tLatent-vec-material shape without bias term is   : {M_act.shape}")
        #print(f"\tMaterial bias vector shape                       :  {mat_bias.shape}")

        assert P_act.shape == (Ihpara['dim'], Inpress), f"\033[91mExpected P_act shape is : {(Ihparam['dim'], Inpress)}, but got {P_act.shape}\033[0m"
        assert M_act.shape == (Ihpara['dim'], Inmat), f"\033[91mExpected M_act shape is : {(Ihparam['dim'], Inmat)}, but got {M_act.shape}\033[0m"

        # Recreating the matrix IMat by using the decomposed matrix M_act and P_act. Note here we have bias separated from RP and RM
        IMat_act_manual = P_act.T @ M_act + np.ones((RP.shape[1], 1)) @ mat_bias.T

        #print(IMat_pred_model)
        #print("\n\n")
        #print(IMat_pred_manual)
        #print("\n\n")
        #print(IMat_act_manual)

        # Quantify error over full matrix
        abs_full_errors = np.abs((IMat_pred_manual - IMat_act_manual) / IMat_pred_manual)
        #print(f"\n\tImputation Consistency Check between IMat_pred_manual and IMat_act_manual")
        print(f"\n\tMax reconstruction error :   {np.max(abs_full_errors):.4f}. This should be close to 0.0000")
        #print(f"\tThis should be as close to 0.00 as possible\n")
        #print(f"Min reconstruction error:   {np.min(abs_full_errors):.6f}")
        #print(f"Mean reconstruction error:  {np.mean(abs_full_errors):.6f}")

        assert IMat_pred_model.shape == IMat_act_manual.shape, \
                f"\033[91mReconstruction shape mismatch: {IMat_pred_model.shape} vs {IMat_act_manual.shape}\033[0m"
        assert IMat_pred_manual.shape == IMat_act_manual.shape, \
                f"\033[91mReconstruction shape mismatch: {IMat_pred_manual.shape} vs {IMat_act_manual.shape}\033[0m"
        assert np.allclose(IMat_pred_model, IMat_act_manual, atol=1e-4), \
                f"\033[91mManual reconstruction model.impute()  and P_act.T @ M_act + np.ones((RP.shape[1], 1)) @ mat_bias.T does not match model's imputed output.\033[0m"

        assert np.allclose(IMat_pred_manual, IMat_act_manual, atol=1e-4), \
                f"\033[91mManual reconstruction (RM.T @ RP)  and P_act.T @ M_act + np.ones((RP.shape[1], 1)) @ mat_bias.T does not match model's imputed output.\033[0m"

        #raise ValueError('Testing is Going on !!!')

    elif Ioffval == False:

        # Initialize if not provided
        if IP0 is None:
            IP0 = np.random.randn(Ihpara["dim"], Inpress)
        if IM0 is None:
            IM0 = np.random.randn(Ihpara["dim"], Inmat)

        Rmodel = glrm(
            losses=QuadLoss(),
            rx=QuadReg(Ihpara["reg"] / Inmat),
            ry=QuadReg(Ihpara["reg"] / Inpress),
            n_components=Ihpara["dim"],
            X=IM0,
            Y=IP0,
            offset=Ioffval
            )

        print(f"\tOffset: {Rmodel.offset}")

        assert Rmodel.offset is False, f"\033[91mThe expected offset to the glrm model is False, but got {Rmodel.offset}\033[0m"

        Rmodel.set_hyperparams(obs=Iidx_known)

        print("\n\t============================================================================================\n")
        Rch = Rmodel.fit(inputs = IMat)

        RP = Rmodel.X  # (dim + 1, Inmat) material vectors
        RM = Rmodel.Y  # (dim + 1, Inpress) Pressure vectors

        #print(RP.shape)
        #print(RM.shape)

        assert RP.shape == (Ihpara['dim'], Inpress), f"\033[91mExpected shape of RP is {(Ihpara['dim'], Inpress)}, but got {RP.shape}\033[0m"
        assert RM.shape == (Ihpara['dim'], Inmat), f"\033[91mExpected RM shape {(Ihpara['dim'], Inmat)}, but got {RM.shape}\033[0m"

        # Internal consistency check: impute vs manual P.T @ M
        IMat_pred_model = Rmodel.impute()

        # Recreating the matrix IMat by using the decomposed matrix RM and RP. Note these contains bias in them
        IMat_pred_manual = RP.T @ RM # shape (Inpress * Ihpara.dim) @ (Ihpara.dim, Inmat) = (Inpress * Inmat)

        ## Note that last row of RP has does not contain any material bias and is just row of 1
        ##P_act = RP[:-1, :] # because RP.shape = Ihpara.dim + 1 * Inpress but in the X = P.T + M P.shape = Ihpara.dim * Inpress that is we need to exclude the last row of RP

        ## Note that the last row of RM contains the material bias
        ##M_act = RM[:-1, :] # because RM.shape = Ihpara.dim + 1 * Inpress but in the X = P.T + M M.shape = Ihpara.dim * Inmat that is we need to exclude the last row of RM

        ##mat_bias = RM[-1, :].reshape(-1, 1)  # shape (Inmat, 1) This is material bias
    
        ##print(f"\n\t===========================================================")
        ##print(f"\n\tLatent-vec-Pressure shape without bias term is : {P_act.shape}")
        ##print(f"\tLatent-vec-material shape without bias term is   : {M_act.shape}")
        ##print(f"\tMaterial bias vector shape                       :  {mat_bias.shape}")

        ##assert P_act.shape == (Ihpara['dim'], Inpress), f"\033[91mExpected P_act shape is : {(Ihparam['dim'], Inpress)}, but got {P_act.shape}\033[0m"
        ##assert M_act.shape == (Ihpara['dim'], Inmat), f"\033[91mExpected M_act shape is : {(Ihparam['dim'], Inmat)}, but got {M_act.shape}\033[0m"

        ## Recreating the matrix IMat by using the decomposed matrix M_act and P_act. Note here we have bias separated from RP and RM
        ##IMat_act_manual = P_act.T @ M_act + np.ones((RP.shape[1], 1)) @ mat_bias.T

        ##print(IMat_pred_model)
        ##print("\n\n")
        ##print(IMat_pred_manual)
        ##print("\n\n")
        ##print(IMat_act_manual)

        # Quantify error over full matrix
        abs_full_errors = np.abs((IMat_pred_manual - IMat_pred_model) / IMat_pred_manual)
        #print(f"\n\tImputation Consistency Check between IMat_pred_manual and IMat_pred_model")
        print(f"\n\tMax reconstruction error :   {np.max(abs_full_errors):.4f}. This should be close to 0.0000")
        #print(f"\tThis should be as close to 0.00 as possible\n")
        #print(f"Min reconstruction error:   {np.min(abs_full_errors):.6f}")
        #print(f"Mean reconstruction error:  {np.mean(abs_full_errors):.6f}")

        assert IMat_pred_manual.shape == IMat_pred_model.shape, \
                f"\033[91mReconstruction shape mismatch: {IMat_pred_manual.shape} vs {IMat_pred_model.shape}"
        #assert IMat_pred_manual.shape == IMat_act_manual.shape, \
        #        f"\033[91mReconstruction shape mismatch: {IMat_pred_manual.shape} vs {IMat_act_manual.shape}\033[0m"
        #assert np.allclose(IMat_pred_model, IMat_act_manual, atol=1e-4), \
        #        f"\033[91mManual reconstruction model.impute()  and P_act.T @ M_act + np.ones((RP.shape[1], 1)) @ mat_bias.T does not match model's imputed output.\033[0m"

        assert np.allclose(IMat_pred_manual, IMat_pred_model, atol=1e-4), \
                f"\033[91mManual reconstruction (RM.T @ RP)  and Rmodel.impute() does not match.\033[0m"

        #raise ValueError('Testing is Going on !!!')

    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"\n\tTime taken to run fit_lrm is : {elapsed_time:.4f} minutes")

    return RP, RM, Rmodel, Rch

def GLRMTest(Ioffval, IAmp_T, Inmat, Inpress, Isample):

    """

    Description: 

    GLRMTest(offset, Amp_T, nmat, npress, sample)

            Aim is to run several test cases on the fit_lrm() function to check whether function works or not.

    Input:

            Ioffval     ==> Offset value to be passed to the glrm model. If 'True' then offsets are added
            
            My question why to put True what if its false?

            IAmp_T      ==>  Pressure-material matrix Amp_T

            Inmat       ==>   # of materials or # rows of material-pressure matrix

            Inpress     ==> # of pressure or # columns of material-pressure matrix

            Isample     ==> # Sampling strategy for selecting missing entries of pressure-material matrix required for PropMatMissing

    Output: 

            Validation that fit_lrm() works fine
    """

    start_time = time.time()

    print("\n======================================================================================")
    print(f"Running GLRM fitting, imputation and consistency checks for Ioffval = {Ioffval}, frac = 0.9 and frac = 0.4 ...")
    print("========================================================================================\n")

    # Step 1: Creating hyperparameters
    hpara = {"dim": 3, "reg": 0.4}

    # Step 2: Custom initialization of matrix P0 and M0
    if Ioffval == True:
        P0 = np.random.randn(hpara["dim"] + 1, nmat)
        M0 = np.random.randn(hpara["dim"] + 1, npress)
    elif Ioffval == False:
        P0 = np.random.randn(hpara["dim"], nmat)
        M0 = np.random.randn(hpara["dim"], npress)

    # Step 3: Creating Missing Pressure-material matrix with frac = 0.7
    [Amp_T_incomplete_frac, idx_known_frac, idx_unknown_frac] = PropMatMissing(0.7, IAmp_T, Inmat, Inpress, Isample, None, None, None, None)

    # Step 4: Standardizing the incomplete Pressure-material matrix
    Amp_T_incomplete_frac_mean, Amp_T_incomplete_frac_std, Amp_T_incomplete_frac_stan = standardize_matrix(Amp_T_incomplete_frac, 1)

    # Step 5: Fiting the model for frac = 0.7
    P, M, model, ch = fit_lrm(Amp_T_incomplete_frac_stan, Inpress, Inmat, hpara, idx_known_frac, P0, M0, Ioffval)

    # Step 5 Fiting the model for frac = 0.4
    # Note here the inputs to fit_lrm are taken from [DEBUG 4: RANDOM] case to save time
    P, M, model, ch = fit_lrm(Amp_T_incomplete_stan, npress, nmat, hpara, idx_known, P0, M0, Ioffval)

    print("\n\t================================================================================")
    print(f"\t\033[92m GLRM model learned latent and bias vector for Ioffval = {Ioffval}, frac = 0.7 and frac = 0.4 and successfully passed all the tests !!!\033[0m")
    print("\t================================================================================\n")

    # Step 5: Bias-only model for frac = 0.9
    hpara_bias = {"dim": 0, "reg": 0.00}

    P0_bias = np.random.randn(hpara["dim"] + 1, nmat)
    M0_bias = np.random.randn(hpara["dim"] + 1, npress)

    P_b, M_b, model_b, ch_b = fit_lrm(Amp_T_incomplete_frac_stan, Inpress, Inmat, hpara_bias, idx_known_frac, P0, M0, True)

    X_full_bias_only = P_b.T @ M_b

    assert M_b.shape == (1, nmat), "\033[91mBias-only M shape incorrect\033[0m"
    assert P_b.shape == (1, npress), "\033[91mBias-only P shape incorrect\033[0m"
    assert np.allclose(model_b.impute(), X_full_bias_only, atol=1e-4), \
        "\033[91mBias-only imputation mismatch\033[0m"

    # Step 7: Bias-only model for frac = 0.4
    # Note here the inputs to fit_lrm are taken from [DEBUG 4: RANDOM] case to save time

    P_b, M_b, model_b, ch_b = fit_lrm(Amp_T_incomplete_stan, npress, nmat, hpara_bias, idx_known, P0, M0, True)
    
    X_full_bias_only = P_b.T @ M_b

    assert M_b.shape == (1, nmat), "\033[91mBias-only M shape incorrect\033[0m"
    assert P_b.shape == (1, npress), "\033[91mBias-only P shape incorrect\033[0m"
    assert np.allclose(model_b.impute(), X_full_bias_only, atol=1e-4), \
            "\033[91mBias-only imputation mismatch\033[0m"

    print("\n\t================================================================================")
    print("\t\033[92m GLRM model successfully passed all the tests for learning BIAS ONLY vector for frac = 0.9 and frac = 0.4\033[0m")
    print("\t================================================================================\n")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    print("\n================================================================================")
    print(f"\033[92m[DEBUG 10]: All GLRM tests successfully passed for Ioffval = {Ioffval}.\033[0m")
    print(f"\n\tTime taken to run GLRMTest is : {elapsed_time:.4f} minutes")
    print("================================================================================\n")

def PlotHPError(IRMSD, IR2, Ihpg, Ifrac, irun):

    """

    Description:

            The aim is to plot the error in validation set as we search through hyperparameter space

    Input:  

            IRMSD       ==> List of RMSD values on validation set

            IR2         ==> List of R2 values on validation set

            Ihpg        ==> Dictionary of hyper-parameter grid

            Ifrac       ==> Fraction of entries in the Pressure-material matrix that should be known (e.g., 0.4 means 40% known, 60% missing)

    Output:

            Visualization of error matrices as we search through hyper-parameter space.

    To be done:

             Instead of plotting for the entire hyper-parameter grid, please plot for only the top 10 or 20 hyper-paramters in ascending order

    """

    start_time = time.time()

    os.makedirs(f'4_HPTuning/{irun}', exist_ok=True)

    X = np.arange(1, len(Ihpg) + 1)

    # Converting list to numpy array
    iRMSD     = np.array(IRMSD) 

    # RMSD values of the HPs with lowest RMSD
    iRMSD_top = iRMSD[np.argsort(IRMSD)[:10]] 

    # Index of the HPs with lowest RMSD. It could be in any order say 13,76,34,...
    # But we need sorted values like 13, 34, 76, ...
    X_top     = np.array(X)[np.argsort(IRMSD)[:10]]

    # This makes sure that X_top = 1, 2, 3, ...
    sort_idx  = np.argsort(X_top)
    X_top_RMSD, iRMSD_top = X_top[sort_idx], iRMSD_top[sort_idx]
    
    XTicks     = np.arange(1, 11, 1)
    YTicksRMSD = np.arange(0.00, 0.40, 0.05)
    YTicksR2   = np.arange(0.88, 1.02, 0.02)

    # Build labels for the selected top-20 hyperparameters
    hp_labels_RMSD = [f"{Ihpg[idx-1]['dim']}\n{Ihpg[idx-1]['reg']:g}" for idx in X_top_RMSD]

    plt.figure(figsize=(11, 7))
    
    # Plotting RMSD

    plt.plot(XTicks, iRMSD_top, marker='*', linestyle='-', color='green', markersize=20, linewidth=1.5, alpha=0.6, label='RMSD')
    
    #plt.xticks(ticks=XTicks, labels=hp_labels_RMSD, fontsize=23)
    plt.xticks(ticks=XTicks, labels=hp_labels_RMSD, fontdict={'fontsize': 23, 'fontweight': 'bold'})
    
    plt.yticks(ticks=YTicksRMSD, fontsize=23, weight='bold') #fontsize=23
    
    #plt.title(f'Validation RMSD vs. Hyperparameter index ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    #plt.xlabel("Hyperparameter index", fontdict=fontdict_x)
    #plt.ylabel("RMSD", fontdict=fontdict_y)
    
    plt.legend(loc='upper right', fontsize=23) # fontsize=20
    
    plt.savefig(f"4_HPTuning/{irun}/RMSD_HP_{Ifrac}.pdf", format='pdf')

    # Converting list to numpy array
    iR2 = np.array(IR2)

    # R2 values of the HPs with highest R2
    iR2_top = iR2[np.argsort(IR2)[-10:]]

    # Index of the HPs with highest R2. It could be in any order say 13,76,34,...
    # But we need sorted values like 13, 34, 76, ...
    X_top_R2 = np.array(X)[np.argsort(IR2)[-10:]]

    # Conceptually lower RMSD should be equivalent to Higher R2
    # Hence, X_top ~ X_top_R2. At least there should be overlap.
    # Check overlap between the two sets
    overlap = np.intersect1d(X_top, X_top_R2)
    overlap_ratio = len(overlap) / 10

    if overlap_ratio < 0.9:
        print("\n")
        print(f"\033[93mWarning: Only {len(overlap)} of top 20 hyperparameters overlap between RMSD and R²!\033[0m")
        print("\033[93mThis may indicate inconsistency between error and fit quality metrics.\033[0m")
    else:
        print(f"\033[92m Consistency check passed for PlotHPError(): {len(overlap)}/20 overlap between RMSD and R².\033[0m")

    # This makes sure that X_top = 1, 2, 3, ...
    sort_idx  = np.argsort(X_top_R2)
    X_top_R2, iR2_top = X_top_R2[sort_idx], iR2_top[sort_idx]

    hp_labels_R2 = [f"{Ihpg[idx-1]['dim']}\n{Ihpg[idx-1]['reg']:g}" for idx in X_top_R2]

    plt.figure(figsize=(11, 7))

    # Plotting R2

    plt.plot(XTicks, iR2_top, marker='*', linestyle='-', color='green', markersize=20, linewidth=1.5, alpha=0.6, label='R2')
    
    #plt.xticks(ticks=XTicks, labels=hp_labels_R2, fontsize=23)
    plt.xticks(ticks=XTicks, labels=hp_labels_R2, fontdict={'fontsize': 23, 'fontweight': 'bold'})
 
    plt.yticks(ticks=YTicksR2, fontsize=23, rotation=0, weight='bold') # fontsize=28
    
    #plt.title(f'Validation R² vs. Hyperparameter index ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    #plt.xlabel("Hyperparameter index", fontdict=fontdict_x)
    #plt.ylabel("R²", fontdict=fontdict_y)
    
    plt.legend(loc='upper right', fontsize=23) # fontsize=20
    
    plt.savefig(f"4_HPTuning/{irun}/R2_HP_{Ifrac}.pdf", format='pdf')

    #--------------------------RMSD and R2 vs dim-------------------------------------#

    # Extract hyperparameters
    reg_vals = np.array([hp["reg"] for hp in Ihpg])
    dim_vals = np.array([hp["dim"] for hp in Ihpg])

    # Compute mean and std of RMSD across all reg for each dim

    mean_rmsd = []
    std_rmsd  = []

    for idim in np.unique(dim_vals):
        vals = iRMSD[dim_vals == idim]
        mean_rmsd.append(np.mean(vals))
        std_rmsd.append(np.std(vals))

    mean_rmsd = np.array(mean_rmsd)
    std_rmsd  = np.array(std_rmsd)
    
    plt.figure(figsize=(11, 7))

    # Plotting RMSD

    plt.errorbar(np.unique(dim_vals), mean_rmsd, yerr=std_rmsd, fmt='o-', linewidth=3, color='royalblue', markersize=12, capsize=5, label='RMSD')
    
    plt.xticks(ticks=dim_vals, fontsize=20, weight='bold', rotation=90) # weight='bold'
    #plt.xticks(ticks=dim_vals, fontdict={'fontsize': 20, 'fontweight': 'bold'}, rotation=90)
    
    plt.yticks(ticks=YTicksRMSD, fontsize=20, weight='bold')
    
    #plt.title(f'Mean validation RMSD vs. Latent Dimentions ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    plt.xlabel("Latent Dimentions", fontdict=fontdict_x)
    plt.ylabel("RMSD", fontdict=fontdict_y)
    
    plt.legend(loc='best', fontsize=20)
    
    plt.savefig(f"4_HPTuning/{irun}/RMSD_LatDim_{Ifrac}.pdf", format='pdf')

    # Converting list to numpy array
    iR2 = np.array(IR2)

    # Compute mean and std of RMSD across all reg for each dim

    mean_R2 = []
    std_R2  = []

    for idim in np.unique(dim_vals):
        vals = iR2[dim_vals == idim]
        mean_R2.append(np.mean(vals))
        std_R2.append(np.std(vals))

    mean_R2 = np.array(mean_R2)
    std_R2  = np.array(std_R2)

    plt.figure(figsize=(11, 7))

    # Plotting R2

    plt.errorbar(np.unique(dim_vals), mean_R2, yerr=std_R2, fmt='o-', linewidth=3, color='royalblue', markersize=12, capsize=5, label='R²')
    
    plt.xticks(ticks=dim_vals, fontsize=20, weight='bold', rotation=90) # weight='bold'
    
    plt.yticks(ticks=YTicksR2, fontsize=20, weight='bold')
    
    #plt.title(f'Validation R² vs. Latent Dimention ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    #plt.xlabel("Latent Dimention", fontdict=fontdict_x)
    #plt.ylabel("R²", fontdict=fontdict_y)
    
    plt.legend(loc='best', fontsize=20)
    
    plt.savefig(f"4_HPTuning/{irun}/R2_LatDim_{Ifrac}.pdf", format='pdf')

    #--------------------------RMSD and R2 vs reg-------------------------------------#

    # Compute mean and std of RMSD across all reg for each dim

    mean_rmsd = []
    std_rmsd  = []

    for ireg in np.unique(reg_vals):
        vals = iRMSD[reg_vals == ireg]
        mean_rmsd.append(np.mean(vals))
        std_rmsd.append(np.std(vals))

    mean_rmsd = np.array(mean_rmsd)
    std_rmsd  = np.array(std_rmsd)
    
    plt.figure(figsize=(11, 7))

    # Plotting RMSD

    plt.errorbar(np.arange(1, len(np.unique(reg_vals)) + 1), mean_rmsd, yerr=std_rmsd, fmt='o-', linewidth=3, color='royalblue', markersize=12, capsize=5, label='RMSD')
    
    plt.xticks(ticks = np.arange(1, len(np.unique(reg_vals)) + 1), labels = [f"{val:g}" for val in np.unique(reg_vals)], rotation=90,fontdict={'fontsize': 20, 'fontweight': 'bold'}) # 'fontweight': 'bold'
    
    plt.yticks(ticks=YTicksRMSD, fontsize=20, weight='bold') # weight='bold'
    
    #plt.title(f'Mean validation RMSD vs. Regularization parameter ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    #plt.xlabel("Regularization Paramter", fontdict=fontdict_x)
    #plt.ylabel("RMSD", fontdict=fontdict_y)
    
    plt.legend(loc='best', fontsize=20)
    
    plt.savefig(f"4_HPTuning/{irun}/RMSD_LatReg_{Ifrac}.pdf", format='pdf')

    # Converting list to numpy array
    iR2 = np.array(IR2)

    # Compute mean and std of RMSD across all reg for each dim

    mean_R2 = []
    std_R2  = []

    #for ireg in reg_vals:
    for ireg in np.unique(reg_vals):
        vals = iR2[reg_vals == ireg]
        mean_R2.append(np.mean(vals))
        std_R2.append(np.std(vals))

    mean_R2 = np.array(mean_R2)
    std_R2  = np.array(std_R2)

    plt.figure(figsize=(11, 7))

    # Plotting R2
    
    plt.errorbar(np.arange(1, len(np.unique(reg_vals)) + 1), mean_R2, yerr=std_R2, fmt='o-', linewidth=3, color='royalblue', markersize=12, capsize=5, label='R²')
    
    plt.xticks(ticks = np.arange(1, len(np.unique(reg_vals)) + 1), labels = [f"{val:g}" for val in np.unique(reg_vals)], rotation=90,fontdict={'fontsize': 20, 'fontweight': 'bold', 'fontweight': 'bold'}) # 'fontweight': 'bold'
    
    plt.yticks(ticks=YTicksR2, fontsize=20, weight='bold') # weight='bold'
    
    #plt.title(f'Validation R² vs. Regularization Parameter ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    #plt.xlabel("Regularization Parameter", fontdict=fontdict_x)
    #plt.ylabel("R²", fontdict=fontdict_y)
    
    plt.legend(loc='best', fontsize=19)
    
    plt.savefig(f"4_HPTuning/{irun}/R2_LatReg_{Ifrac}.pdf", format='pdf')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print(f"Time taken to run PlotHPError() is: {elapsed_time:.4f} minutes")

def PlotHPGrid(IRMSD, IR2, Ihpg, Ifrac, irun):

    """

    Description:

        Plot validation performance (RMSD and R²) across the hyperparameter grid using Seaborn.

    Inputs:

        IRMSD  : list of RMSD values
        
        IR2    : list of R² values
        
        Ihpg   : list of dictionaries, each with keys 'reg' and 'dim'
        
        Ifrac  : fraction of known entries in the matrix
        
        irun   : ith run out of nrun for the same Ifrac to ensure statistical accuracy

    """

    start_time = time.time()

    os.makedirs(f'4_HPTuning/{irun}', exist_ok=True)

    IRMSD = np.array(IRMSD)
    IR2   = np.array(IR2)

    # Extract hyperparameters
    reg_vals = np.array([hp["reg"] for hp in Ihpg])
    dim_vals = np.array([hp["dim"] for hp in Ihpg])

    reg_unique = np.unique(reg_vals)
    dim_unique = np.unique(dim_vals)

    # Initialize grids
    RMSD_grid = np.full((len(dim_unique), len(reg_unique)), np.nan)
    R2_grid   = np.full((len(dim_unique), len(reg_unique)), np.nan)

    # Fill grids
    for i, reg in enumerate(reg_unique):
        for j, dim in enumerate(dim_unique):
            mask = (reg_vals == reg) & (dim_vals == dim)
            if np.sum(mask) > 1:
                raise ValueError(f"\033[91mDuplicate entries found for reg={reg}, dim={dim}")
            elif np.sum(mask) == 0:
                raise ValueError(f"\033[91mNo entry found for reg={reg}, dim={dim}")
            else:
                RMSD_grid[j, i] = IRMSD[mask][0]
                R2_grid[j, i]   = IR2[mask][0]

    # Convert grids to DataFrame for Seaborn
    RMSD_df = pd.DataFrame(RMSD_grid, index=dim_unique, columns=reg_unique)
    R2_df   = pd.DataFrame(R2_grid, index=dim_unique, columns=reg_unique)

    colors = sns.color_palette("hls", 8)   # 8 distinct hues
    cmap_hls = LinearSegmentedColormap.from_list("hls_continuous", colors, N=256)

    # --- RMSD Heatmap ---
    plt.figure(figsize=(11,7))
    
    #ax = sns.heatmap(RMSD_df, cmap=cmap_hls, linewidths=1, linecolor='black', \
    #    cbar_kws={'label': 'Validation RMSD'}, xticklabels=True, yticklabels=True)

    ax = sns.heatmap(RMSD_df, cmap=cmap_hls, linewidths=1, linecolor='black', xticklabels=True, yticklabels=True)
    
    cbar = ax.collections[0].colorbar
    
    #cbar.set_label('Validation RMSD', fontsize=22, fontweight='bold', color='black')

    cbar.ax.tick_params(labelsize=20, width=1, color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')
    
    ax.collections[0].colorbar.outline.set(linewidth=2, edgecolor='black')
    
    #plt.xlabel("Regularization", fontdict=fontdict_x)
    #plt.ylabel("Latent Dimension", fontdict=fontdict_y)

    plt.xticks(fontsize=18, weight='bold', rotation=90)
    plt.yticks(fontsize=18, weight='bold', rotation=0) # rotation=90
    
    #plt.title(f'Validation RMSD Heatmap ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    plt.tight_layout()
    
    plt.savefig(f"4_HPTuning/{irun}/RMSD_HPGrid_{Ifrac}.pdf", format='pdf')
    
    plt.close()

    # --- R2 Heatmap ---
    plt.figure(figsize=(11,7))

    #ax = sns.heatmap(R2_df, cmap=cmap_hls, linewidths=1, linecolor='black', \
    #    cbar_kws={'label': 'Validation R²'}, xticklabels=True, yticklabels=True)
    
    ax = sns.heatmap(R2_df, cmap=cmap_hls, linewidths=1, linecolor='black', xticklabels=True, yticklabels=True)
    
    cbar = ax.collections[0].colorbar
    
    #cbar.set_label('Validation R²', fontsize=22, fontweight='bold', color='black')

    cbar.ax.tick_params(labelsize=20, width=1, color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')
    
    ax.collections[0].colorbar.outline.set(linewidth=2, edgecolor='black')
    
    #plt.xlabel("Regularization", fontdict=fontdict_x)
    #plt.ylabel("Latent Dimension", fontdict=fontdict_y)

    plt.xticks(fontsize=18, weight='bold', rotation=90)
    plt.yticks(fontsize=18, weight='bold', rotation=0) # rotation=90
    
    #plt.title(f'Validation R² Heatmap ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    plt.tight_layout()
    
    plt.savefig(f"4_HPTuning/{irun}/R2_HPGrid_{Ifrac}.pdf", format='pdf')
    
    plt.close()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print(f"Time taken to run PlotHPGrid() is: {elapsed_time:.4f} minutes")

def HPTune(IMat, Inpress, Inmat, Ihpg, Iidx_train, Iidx_test, Ifrac, Ioffval, irun):
    
    """

    Description:

        Aim is to do hyperparameter tunning over all possible values of hyperparameters and find the best hyperparameter.
        
        Note both Iidx_train and Iidx_test comes from the known elements of IMat (IMat = Amp_T_incomplete_stan)
        
        For each hyperparameter value the function does following tasks:
            
            1.) Fit the model on the elemets of IMat corresponding to ids - Iidx_train
            2.) Impute the full matrix 
            3.) Compute RMSD on Iidx_test
            4.) Keep track of the best hyperparameter i.e. hyperparameter with lowest RMSD.

    Input: 
        
        IMat        ==> Incomplete standardized Pressure-material matrix (i.e. Amp_T_incomplete_stan)

        Ihpg        ==> hyperparameter grid
        
        Inpress     ==> size of properties
        
        Inmat       ==> size of materials
        
        Iidx_train  ==> ids corresponding to RMat which contain known entries which would be utilized for model training. (subset of idx_known of Amp_T_incomplete)
        
        Iidx_test   ==> ids corresponding to RMat which contain known entries which would be utilized for model testing. (subset of idx_known of Amp_T_incomple)
        
        Ifrac       ==> Fraction used to computinh missing elements
        
        Ioffval     ==> True 

        irun        ==> ith run out of nrun for the same Ifrac to ensure statistical accuracy

    Output:

        Rhp         ==> value of best  hyperparameter
        
        Rrmsd_TE    ==> value of test  RMSD corresponding to the best hyperparameter
        
        Rrmsd_TR    ==> value of train RMSD corresponding to the best hyperparameter

        Rplot_RMSD  ==> 

        Rplot_R2    ==>

    """

    start_time = time.time()

    Rplot_RMSD = []
    Rplot_R2   = []

    # Cross check to ensure all the entries in known indices are filled in IMat
    for ip, im in Iidx_train:
        assert not np.isnan(IMat[ip, im]), f"\033[91mMismatch in entries of known indices: Known entry ({ip},{im}) is NaN\033[0m"

    # Cross check to ensure all the entries in known indices are filled in IMat
    for ip, im in Iidx_test:
        assert not np.isnan(IMat[ip, im]), f"\033[91mMismatch in entries of known indices : Known entry ({ip}, {im} is NaN)\033[0m"

    # Asserting on shape of IMat
    assert IMat.shape == (Inpress, Inmat), f"\033[91mShape mismatch in IMat. Expected ({Inpress, Inmat}). Got {IMat.shape}\033[0m"

    # Initializing best hyperparameter
    Rhp      = {"dim": 0, "reg": 0.00}
    Rrmsd_TE = float('inf')
    Rrmsd_TR = float('inf')
    RR2_TE   = float('inf')

    # Progress bar
    count = enumerate(Ihpg, start = 1)
    count = tqdm(count, total=len(Ihpg), desc="Hyperparameter tunning: ")

    for i, ihpg in count:

        print("\n\n\t============================================================================================")
        print(f"\n\tNow testing dim={ihpg['dim']}, reg={ihpg['reg']} for irun={irun}\n")
        print("\t============================================================================================\n")

        # Entries of IMat corresponding to test ids. These will be utilized to compute test RMSD
        IMat_test_true  = np.array([IMat[i, j] for (i, j) in Iidx_test])
        assert not np.isnan(IMat_test_true).any(), "\033[91mNaN values found in IMat_test_true !\033[0m"
        assert np.isfinite(IMat_test_true).all(), "\033[91mNon-finite values in IMat_test_true !\033[0m"

        # Entries of IMat corresponding to train ids. These will be utilized to compute train RMSD
        IMat_train_true = np.array([IMat[i, j] for (i, j) in Iidx_train]) 

        # Train the model on data in train set
        P, M, model, ch = fit_lrm(IMat, Inpress, Inmat, ihpg, Iidx_train, None, None, Ioffval)

        # Imputing the entire matrix
        IMat_pred = model.impute()
        assert not np.isnan(IMat_pred).any(), "\033[91mNaN values found in IMat_pred !\033[0m"
        assert np.isfinite(IMat_pred).all(), "\033[91mFound inf or NaN values in IMat_pred !\033[0m"

        # Entries of imputed matrix corresponding to test ids. These will be utilized to compute RMSD
        IMat_pred_test_pred = np.array([IMat_pred[i, j] for (i, j) in Iidx_test])
        assert not np.isnan(IMat_pred_test_pred).any(), "\033[91mNaN values found in IMat_pred_test_pred !\033[0m"
        assert np.isfinite(IMat_pred_test_pred).all(), "\033[91mNon-finite values in IMat_pred_test_pred !\033[0m"

        # Entries of imputed matrix corresponding to test ids. These will be utilized to compute RMSD
        IMat_pred_train_pred = np.array([IMat_pred[i, j] for (i, j) in Iidx_train])

        # Computing RMSD
        iRMSD = np.sqrt(mean_squared_error(IMat_test_true, IMat_pred_test_pred))
        iR2   = r2_score(IMat_test_true, IMat_pred_test_pred)  

        Rplot_RMSD.append(iRMSD)
        Rplot_R2.append(iR2)

        print(f"\n\tRMSD in validation set is: {iRMSD:.4f}. Should be close to 0.0000\n")
        print(f"\tR2 between actual and predicted values in validation set is: {iR2:.4f}. Should be close to 1.0000\n")

        if iRMSD < Rrmsd_TE:
            
            print(f"\t\tRrmsd_TE is updated from {Rrmsd_TE:.4f} to {iRMSD:.4f}")
            print(f"\t\tRR2_TE is updated from {RR2_TE:.4f} to {iR2:.4f}")
            
            Rrmsd_TE = iRMSD
            RR2_TE   = iR2

            Rrmsd_TR = r2_score(IMat_train_true, IMat_pred_train_pred)
            RR2_TR   = r2_score(IMat_train_true, IMat_pred_train_pred)
            
            print(f"\t\tRrmsd_TR is: {Rrmsd_TR:.4f}")
            print(f"\t\tRR2_TR is: {RR2_TR:.4f}\n")
            
            Rhp      = ihpg

    # Plotting the error in test set as we search through hyperparameter space

    PlotHPError(Rplot_RMSD, Rplot_R2, Ihpg, Ifrac, irun)
    PlotHPGrid(Rplot_RMSD, Rplot_R2, Ihpg, Ifrac, irun)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print("\n================================================================================")
    print(f"\033[92m Hyper-parameter optimization done sucessfullly for Ifrac = {Ifrac} and irun = {irun}!!!\033[0m")
    print(f"Time taken for Hyper-parameter tunning is: {elapsed_time:.4f} minutes")
    print(f"Best hyperparameters are: {Rhp} with RMSD on validation set = {Rrmsd_TE:.4f} and RMSD on train set = {Rrmsd_TR:.4f}")
    print(f"Best RMSD on validation set = {Rrmsd_TE:.4f} and on train set = {Rrmsd_TR:.4f}")
    print(f"Best R2   on validation set = {RR2_TE:.4f} and on train set = {RR2_TR:.4f}")
    print(f"================================================================================\n")

    return Rhp, Rrmsd_TE, Rrmsd_TR, RR2_TE, RR2_TR, Rplot_RMSD, Rplot_R2 

def compute_metrics(y_true, y_pred):

    """

    Description:

            Aim is to compute the error metrices onthe input data. This is useful in ParityPlots().

    Input:
            y_true      ==> True value

            y_pred      ==> Model predicted value

    Output:

            A dict with R2, MAE and RMSE on the incomming data

    """

    return {'R²': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
            }

def ParityPlot(ITe, ITr, IPTe, IPTr, Ibias, Ifrac, irun):

    """

    Description:

            Aim is to plot a parity plot to visualize the model performance on the final lrm model

    Input:

    ParityPlot(True_Test, True_Train, Pred_Test, Pred_Train, bias, Ifrac)

            ITe         ==> True test set values 

            ITr         ==> True train set values

            TPTe        ==> Predicted test set values

            IPTr        ==> Predicted train set values

            Ibias       ==> Whether 'latent-bias' case or 'bias-only' case

            Ifrac       ==> Fraction of entries in the Pressure-material matrix that should be known (e.g., 0.4 means 40% known, 60% missing)

    Output:

            Parity plots for both lateral-bias case and latent-only case and
            bias-only case. These acts a base line to gauge model performance

    """ 

    start_time = time.time()

    os.makedirs(f'5_FFig/{irun}/1_Parity_Plot', exist_ok=True)   

    plt.figure(figsize=(11, 7))

    plt.scatter(ITr, IPTr, color='green', label='Training Data', alpha=0.6, edgecolors='black', s=80)

    plt.scatter(ITe, IPTe, color='red', label='Test Data', alpha=0.6, edgecolors='black', s=80)

    #plt.plot([min(ITr.min(), IPTr.min()), max(ITr.max(), IPTr.max())],
    #        [min(ITr.min(), IPTr.min()), max(ITr.max(), IPTr.max())],
    #        color='gray', linestyle='--', label='Perfect Prediction')
    
    plt.plot([min(min(ITr), min(IPTr)), max(max(ITr), max(IPTr))],
         [min(min(ITr), min(IPTr)), max(max(ITr), max(IPTr))],
         color='gray', linestyle='--', label='Perfect Prediction')

    train_metrics = compute_metrics(ITr, IPTr)
    test_metrics  = compute_metrics(ITe, IPTe)
    
    # Writing Model performance metrics on plot at the bottom left
    
    x_pos = 0.94
    y_pos = 0.04
    
    text = (f"{'Metric'.ljust(10)}{'Train'.rjust(10)}{'Test'.rjust(10)}\n"
            f"{'R²'.ljust(10)}{train_metrics['R²']:>10.3f}{test_metrics['R²']:>10.3f}\n"
            f"{'MAE'.ljust(10)}{train_metrics['MAE']:>10.3f}{test_metrics['MAE']:>10.3f}\n"
            f"{'RMSE'.ljust(10)}{train_metrics['RMSE']:>10.3f}{test_metrics['RMSE']:>10.3f}"
            )

    plt.gca().text(
        x_pos, y_pos, text,
        transform=plt.gca().transAxes,
        fontsize=18, color='black', weight='bold',
        horizontalalignment='right', verticalalignment='bottom',
        fontname='Times',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.7)
        )

    # I think we should put custion ticks
    plt.xticks(fontsize=23, weight='bold')
    plt.yticks(fontsize=23, weight='bold')
    
    #plt.title(f'Predicted vs True values ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
    
    plt.xlabel("True values", fontdict=fontdict_x)
    plt.ylabel("Predicted values", fontdict=fontdict_y)
    
    plt.legend(loc='upper right', fontsize=20)
    
    plt.savefig(f"5_FFig/{irun}/1_Parity_Plot/Parity_plot_{Ibias}_{Ifrac}.pdf", format='pdf')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    return elapsed_time

def PlotIsothermParity(IMat, IMat_pred, IMat_l_pred, IMat_b_pred, Inpress, Inmat, Ifrac, irun, IIIidx_unknown_frac, IIIidx_known_frac, full, Iaxis):
    """

    Description:

            The aim of this function is to plot parity plots between true and 
            predicted isotherms for all the materials.

            mof_name        ==>     (global variable) Names of all the materials whose isothem has to be plotted. 
                                    It is a list of mof_names with a size of (nmat, 1).   


            This function performs following tasks:

                1.) Plot isothem using the true values
                2.) Plot isothem using the predicted values
                3.) Calculate R2 and RMSD between these two curves and write that on plot

    Input:

                IMat            ==>     The normalized or unnormalized completely known pressure-material matrix. (Inpress * Inmat)
                
                IMat_pred       ==>     The normalized or unnormalized latent-bias model predicted pressure-material matrix. (Inpress * Inmat)
                
                IMat_l_pred     ==>     The normalized or unnormalized latent only model predicted pressure-material matrix. (Inpress * Inmat)

                IMat_b_pred     ==>     The normalized or unnormalized bias only model predicted pressure-material matrix. (Inpress * Inmat)
                
                Inpress         ==>     Same as before 
                
                Inmat           ==>     Same as before 
                
                Ifrac           ==>     Same as before

                Iaxis           ==>     if Iaxis = SEMILOG, then X axis of isotherm is in log scale and Y axis in linear scale.
                                        if Iaxis = LINIEAR, then both X and Y axis of the isotherm is in linear scale. 

    Output:

                Parity plots for each material are plotted.

    """

    start_time = time.time()

    assert IMat.shape == (Inpress, Inmat), f"\033[91mIMat shape mismatch: expected ({Inpress}, {Inmat}), got {IMat.shape}\033[0m"

    assert IMat_pred.shape == (Inpress, Inmat), f"\033[91mIMat_pred shape mismatch: expected ({Inpress}, {Inmat}), got {IMat_pred.shape}\033[0m"

    assert IMat_l_pred.shape == (Inpress, Inmat), f"\033[91mIMat_l_pred shape mismatch: expected ({Inpress}, {Inmat}), got {IMat_l_pred.shape}\033[0m"

    assert IMat_b_pred.shape == (Inpress, Inmat), f"\033[91mIMat_b_pred shape mismatch: expected ({Inpress}, {Inmat}), got {IMat_b_pred.shape}\033[0m"

    if Iaxis == 'SEMILOG' and full==True:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/Semilog/True', exist_ok=True)
    elif Iaxis == 'SEMILOG' and full==False:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/Semilog/False', exist_ok=True)
    elif Iaxis == 'LINEAR' and full==True:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/Linear/True', exist_ok=True)
    elif Iaxis == 'LINEAR' and full==False:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/Linear/False', exist_ok=True)
    elif Iaxis == 'UN_SEMILOG' and full==True:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Semilog/True', exist_ok=True)
    elif Iaxis =='UN_SEMILOG' and full==False:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Semilog/False', exist_ok=True)
    elif Iaxis == 'UN_LINEAR' and full==True:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Linear/True', exist_ok=True)
    elif Iaxis == 'UN_LINEAR' and full==False:
        os.makedirs(f'5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Linear/False', exist_ok=True)
    else:
        raise ValueError(f"\033[91mWrong value of axis")

    X = np.array([0.00000001,0.00000010,0.00000100,0.00001000,0.00010000,0.00100000,
        0.01000000,0.02000000,0.04000000,0.06000000,0.08000000,0.10000000,
        0.12000000,0.14000000,0.16000000,0.18000000,0.20000000,0.24000000,
        0.28000000,0.30000000,0.32000000,0.36000000,0.40000000,0.50000000,
        0.60000000,0.70000000,0.80000000,0.90000000,1.00000000])

    X_label = [f"{x:.0e}" for x in X]
        
    for i in range(Inmat):

        plt.figure(figsize=(11,7))

        if Iaxis == 'SEMILOG' or Iaxis == 'UN_SEMILOG':
            plt.xscale('log')
        elif Iaxis == 'LINEAR' or Iaxis == 'UN_LINEAR':
            plt.xscale('linear')

        if full==True:

            # Filter indices to only get the 'missing' points for the CURRENT material (column i)
            current_mat_test_idx = [p for (p, m) in IIIidx_unknown_frac if m == i]
            
            True_Test_i   = IMat[current_mat_test_idx, i]
            Pred_Test_i   = IMat_pred[current_mat_test_idx, i]
            Pred_l_Test_i = IMat_l_pred[current_mat_test_idx, i]
            Pred_b_Test_i = IMat_b_pred[current_mat_test_idx, i]

            # Calculate metrics for THIS material only
            TestMat   = compute_metrics(True_Test_i, Pred_Test_i)
            TestMat_l = compute_metrics(True_Test_i, Pred_l_Test_i)
            TestMat_b = compute_metrics(True_Test_i, Pred_b_Test_i)

            # Extract row indices for this specific material
            mat_known_idx = [p for (p, m) in IIIidx_known_frac if m == i]
            mat_unknown_idx = [p for (p, m) in IIIidx_unknown_frac if m == i]

            # 1. Plot Known True Values (Points the model saw)
            if mat_known_idx:
                plt.scatter(X[mat_known_idx], IMat[mat_known_idx, i],s=100, marker='*', linestyle='-', facecolors='none', edgecolors='blue', linewidth=1.5, alpha=1.0, label="True values (Train)")

            # 2. Plot Unknown True Values (Points the model had to guess)
            if mat_unknown_idx:
                plt.scatter(X[mat_unknown_idx], IMat[mat_unknown_idx, i],s=100, marker='*', linestyle='-', facecolors='none', edgecolors='magenta', linewidth=1.5, alpha=1.0, label="True values (Test)")

            #True_Test    = [IMat[p, m] for (p, m) in IIIidx_unknown_frac]         # True normalized or unnormalized values present in test set
            #Pred_Test    = [IMat_pred[p, m] for (p, m) in IIIidx_unknown_frac]    # Latent-bias Model predicted normalized or unnormalized values present in test set 
            #Pred_l_Test  = [IMat_l_pred[p, m] for (p, m) in IIIidx_unknown_frac]  # latent only model predicted normalized or unnormalized values present in test set
            #Pred_b_Test  = [IMat_b_pred[p, m] for (p, m) in IIIidx_unknown_frac]  # Bias only model predicted normalized or unnormalized values present in test set

            ## Calculating RMSD and R2 between true and predicted values for test set for final latent-bias model
            #TestMat      = compute_metrics(True_Test, Pred_Test)

            ## Calculating RMSD and R2 between true and predicted values for test set for final latent only model
            #TestMat_l    = compute_metrics(True_Test, Pred_l_Test)

            ## Calculating RMSD and R2 between true and predicted values for test set for final bias only model
            #TestMat_b    = compute_metrics(True_Test, Pred_b_Test)

        # Compute metrics
        latent_bias_metrics = compute_metrics(IMat[:, i], IMat_pred[:, i])
        latent_only_metrics = compute_metrics(IMat[:, i], IMat_l_pred[:, i])
        bias_only_metrics   = compute_metrics(IMat[:, i], IMat_b_pred[:, i])

        # Writing Model performance metrics on plot at the bottom left

        x_pos = 0.94
        y_pos = 0.04

        if full == True:

            text = (f"{'Metric'.ljust(10)}{'LB'.rjust(10)}{'LO'.rjust(10)}{'BO'.rjust(10)}\n"
                f"{'R²'.ljust(10)}{TestMat['R²']:>10.3f}{TestMat_l['R²']:>10.3f}{TestMat_b ['R²']:>10.3f}\n"
                f"{'MAE'.ljust(10)}{TestMat['MAE']:>10.3f}{TestMat_l['MAE']:>10.3f}{TestMat_b ['MAE']:>10.3f}\n"
                f"{'RMSE'.ljust(10)}{TestMat['RMSE']:>10.3f}{TestMat_l['RMSE']:>10.3f}{TestMat_b ['RMSE']:>10.3f}"
                )
        else:

            text = (f"{'Metric'.ljust(10)}{'LB'.rjust(10)}{'LO'.rjust(10)}{'BO'.rjust(10)}\n"
                f"{'R²'.ljust(10)}{latent_bias_metrics['R²']:>10.3f}{latent_only_metrics['R²']:>10.3f}{bias_only_metrics['R²']:>10.3f}\n"
                f"{'MAE'.ljust(10)}{latent_bias_metrics['MAE']:>10.3f}{latent_only_metrics['MAE']:>10.3f}{bias_only_metrics['MAE']:>10.3f}\n"
                f"{'RMSE'.ljust(10)}{latent_bias_metrics['RMSE']:>10.3f}{latent_only_metrics['RMSE']:>10.3f}{bias_only_metrics['RMSE']:>10.3f}"
                )
            plt.scatter(X, IMat[:, i], s=100, marker='*', linestyle='-', facecolors='none', edgecolors='blue', linewidth=1.5, alpha=1.0, label="True values")

        plt.gca().text(
            x_pos, y_pos, text,
            transform=plt.gca().transAxes,
            fontsize=18, color='black', weight='bold',
            horizontalalignment='right', verticalalignment='bottom',
            fontname='Times',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.7)
            )

        #plt.scatter(X, IMat[:, i], s=100, marker='*', linestyle='-', facecolors='none', edgecolors='blue', linewidth=1.5, alpha=1.0, label="True values")
        plt.scatter(X, IMat_pred[:, i], s=100, marker='s', linestyle='-', facecolors='none', edgecolors='red', linewidth=1.5, alpha=1.0, label="Latent-bias")
        plt.scatter(X, IMat_l_pred[:, i], s=100, marker='^', linestyle='-', facecolors='none', edgecolors='black', linewidth=1.5, alpha=1.0, label="Latent-only")
        plt.scatter(X, IMat_b_pred[:, i], s=100, marker='o', linestyle='-', facecolors='none', edgecolors='green', linewidth=1.5, alpha=1.0, label="Bias-only")

        plt.xlabel('Pressure (bar)', fontdict=fontdict_x)
        #plt.ylabel('Argon uptake normalized values', fontdict=fontdict_y)

        if Iaxis == 'SEMILOG':
            plt.ylabel('Argon uptake normalized values', fontdict=fontdict_y)
        elif Iaxis == 'LINEAR':
            plt.ylabel('Argon uptake normalized values', fontdict=fontdict_y)
        elif Iaxis == 'UN_SEMILOG':
            plt.ylabel('Argon uptake (mol/kg)', fontdict=fontdict_y)
        elif Iaxis == 'UN_LINEAR':
            plt.ylabel('Argon uptake (mol/kg)', fontdict=fontdict_y)

        #plt.title(f'Adsorption isotherm at 87 K for {mof_name[i]} ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)

        #plt.xticks(ticks = X, labels = X_label, rotation=90, fontdict={'fontsize': 23, 'fontweight': 'bold'})
        plt.xticks(fontsize=20, fontweight='bold') # fontweight='bold'
        plt.yticks(fontsize=20, fontweight='bold') # fontweight='bold'

        plt.legend(loc='upper right', fontsize=20)

        plt.tight_layout()

        if Iaxis == 'SEMILOG' and full==True:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/Semilog/True/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'SEMILOG' and full==False:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/Semilog/False/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'LINEAR' and full==True:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/Linear/True/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'LINEAR' and full==False:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/Linear/False/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'UN_SEMILOG' and full==True:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Semilog/True/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'UN_SEMILOG' and full==False:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Semilog/False/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'UN_LINEAR' and full==True:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Linear/True/{i}_{mof_name[i]}_Iso.pdf", format='pdf')
        elif Iaxis == 'UN_LINEAR' and full==False:
            plt.savefig(f"5_FFig/{irun}/2_Isotherm_Parity_Plot/UN_Linear/False/{i}_{mof_name[i]}_Iso.pdf", format='pdf')

        plt.close()
        
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    return elapsed_time

def PCAAfter(IM, IP, Ibesthp, Inmat, Inpress, Ioffval):

    """

    Description:

    Input:

    Output:

    """

    # Stack M and P horizontally (excluding the last row from both)
    if Ioffval == True:
        
        MAct = IM[:-1, :]  # shape: (k, Inmat) # For deployable latent-bias model
        PAct = IP[:-1, :]  # shape: (k, Inpress) # For deployable latent-bias model
        
        assert MAct.shape == (Ibesthp['dim'], Inmat), \
                f"\033[91mMismatch in Mact.shape: Expected ({Ibesthp['dim']}, {Inmat}), got {MAct.shape}\033[0m"
        assert PAct.shape == (Ibesthp['dim'], Inpress), \
                f"\033[91mMismatch in Pact.shape: Expected ({Ibesthp['dim']}, {Inpress}, got {PAct.shape})\033[0m"
    
    elif Ioffval == False:

        MAct = IM # shape: (k, Inmat) # For deployable latent-bias model
        PAct = IP # shape: (k, Inpress) # For deployable latent-bias model

        assert MAct.shape == (Ibesthp['dim'], Inmat), \
                f"\033[91mMismatch in Mact.shape: Expected ({Ibesthp['dim']}, {Inmat}), got {MAct.shape}\033[0m"
        assert PAct.shape == (Ibesthp['dim'], Inpress), \
                f"\033[91mMismatch in Pact.shape: Expected ({Ibesthp['dim']}, {Inpress}, got {PAct.shape})\033[0m"

    AllLatVec = np.hstack((MAct, PAct))  
    
    assert AllLatVec.shape == (Ibesthp["dim"], Inpress + Inmat)

    pca = PCA(n_components = 2)

    AllLatVec_T = AllLatVec.T

    p_pca = pca.fit_transform(AllLatVec_T) # Python's PCA function expects samples as rows

    p_pca_T = p_pca.T

    m_vecs_2D = p_pca_T[:, :Inmat]
    p_vecs_2D = p_pca_T[:, Inmat:]

    assert m_vecs_2D.shape == (2, Inmat)
    assert p_vecs_2D.shape == (2, Inpress)

    return MAct, PAct, AllLatVec, m_vecs_2D, p_vecs_2D

def PlotPCAAfter(Ip_vecs_2D, Im_vecs_2D, IMat, Inmat, Inpress, Ipressure, Ifrac, irun, Ioffval):

    start_time = time.time()

    #os.makedirs(f'PCA_After/{Ifrac}/{irun}/{Ioffval}', exist_ok=True)
    os.makedirs(f'5_FFig/{irun}/PCA_After/{Ioffval}', exist_ok=True)

    # Property Case
    assert Ip_vecs_2D.shape == (2, Inpress), f"\033[91mExpected {(2, Inpress)}, got {dg_pca.shape}\033[0m"
    
    # Material Case
    assert Im_vecs_2D.shape == (2, Inmat), f"\033[91mExpected {(2, Inmat)}, got {dg_pca.shape}\033[0m"

	# Property Case
    colors = plt.cm.get_cmap('hsv', Inpress)
    
    # Material Case
    cmap = get_cmap('rainbow', 7)

    # Property Case
    norm_pca = np.linalg.norm(Ip_vecs_2D.T, axis=1)  # Find the magnitude of each vector
    normalized_pca = Ip_vecs_2D.T / norm_pca[:, np.newaxis]  # Normalize each vector by its magnitude

    # Material Case
    norm       = MinMaxScaler()
    Mat_norm   = norm.fit_transform(IMat.T) # This is done for colorbar in plt.scatter()

    # Plotting only pca after property
    plt.figure(figsize = (11,7))

    plt.axhline(0, color='black', linewidth=2)  # x-axis
    plt.axvline(0, color='black', linewidth=2)  # y-axis

    sc = plt.scatter(Im_vecs_2D.T[:,0], Im_vecs_2D.T[:,1], s=30, edgecolor='black', linewidth=1, alpha=0.75, facecolor='none')
    
    plt.xlabel('PCA dimention 1', fontdict=fontdict_x)
    plt.ylabel('PCA dimention 2', fontdict=fontdict_y)

    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')

    for i in range(Inpress):

        val = float(Ipressure[i])
        mantissa, exponent = f"{val:.1e}".split("e")
        exponent = int(exponent)

        if normalized_pca[i,0] < 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 0.3 # 1.0
            normalized_pca[i,1] = normalized_pca[i,1] * 0.3 # 1.0
        elif normalized_pca[i,0] > 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 0.3 # 1.0
            normalized_pca[i,1] = normalized_pca[i,1] * 0.3 # 1.0

        plt.quiver(0.0, 0.0, normalized_pca[i,0], normalized_pca[i,1], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)

    #plt.savefig(f"PCA_After/{Ifrac}/{irun}/{Ioffval}/Mat_Prop_space.pdf", format='pdf')
    plt.savefig(f"5_FFig/{irun}/PCA_After/{Ioffval}/Mat_Prop_space.pdf", format='pdf')
        
    plt.close()

    for i in range(Inpress): # Inpress

    	# Material Case
        
        plt.figure(figsize = (11,7))

        plt.axhline(0, color='black', linewidth=2)  # x-axis
        plt.axvline(0, color='black', linewidth=2)  # y-axis

        val = float(Ipressure[i])
        mantissa, exponent = f"{val:.1e}".split("e")
        exponent = int(exponent)

        sc = plt.scatter(Im_vecs_2D.T[:,0], Im_vecs_2D.T[:,1], s=30, c=Mat_norm[:,i], cmap = cmap, edgecolor='black', linewidth=1, alpha=0.75)

        cbar = plt.colorbar(sc)
        cbar.set_label('\n' + rf'$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ bar}}$',fontsize=23)
        cbar.ax.tick_params(labelsize=23, width=1, color='black')
        plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

        cbar.set_ticks([0.000, 0.143, 0.286, 0.428, 0.571, 0.714, 0.857, 1.000])
        cbar.set_ticklabels(['0.00', '0.14', '0.28', '0.42', '0.57', '0.71', '0.86', '1.00'])

        plt.xlabel('PCA dimention 1', fontdict=fontdict_x)
        plt.ylabel('PCA dimention 2', fontdict=fontdict_y)
        #plt.title(f'PCA Material-Property-Latent Matrix ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)

        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')

        xmax = np.max(np.abs(Im_vecs_2D.T[:, 0]))
        ymax = np.max(np.abs(Im_vecs_2D.T[:, 1]))

        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)

        ##plt.xticks(ticks = [-1.0, -0.5, 0.0, 0.5, 1.0], labels = ['-1.0', '-0.5', '0.0', '0.5', '1.0'], fontdict={'fontsize': 23, 'fontweight': 'bold'})
        ##plt.yticks(ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], labels = ['-1.5','-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'], fontdict={'fontsize': 23, 'fontweight': 'bold'})
        
        #plt.xticks(ticks = [-0.5, 0.0, 0.5, 1.0], labels = ['-0.5', '0.0', '0.5', '1.0'], fontdict={'fontsize': 23, 'fontweight': 'bold'})
        #plt.yticks(ticks = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], labels = ['-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'], fontdict={'fontsize': 23, 'fontweight': 'bold'})

        ##plt.xticks(ticks = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8], labels = ['-0.2', '0.0', '0.2', '0.8', '1.0'], fontdict={'fontsize': 23, 'fontweight': 'bold'})
        ##plt.yticks(ticks = [-1.0, -0.5, 0.0, 0.5, 1.0], labels = ['-1.0', '-0.5', '0.0', '0.5', '1.0'], fontdict={'fontsize': 23, 'fontweight': 'bold'})
        
        # Property Case

        if normalized_pca[i,0] < 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 1.0 # 0.3
            normalized_pca[i,1] = normalized_pca[i,1] * 1.0 # 0.3
        elif normalized_pca[i,0] > 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 1.0 # 0.3
            normalized_pca[i,1] = normalized_pca[i,1] * 1.0 # 0.3

        #plt.quiver(0.0, 0.0, Ip_vecs_2D[0,i], Ip_vecs_2D[1,i], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)
        plt.quiver(0.0, 0.0, normalized_pca[i,0], normalized_pca[i,1], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)

        legend_patch = mpatches.Patch(color=colors(i), label=rf"$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ bar}}$")
        
        plt.legend(handles=[legend_patch], fontsize=18, loc='best', frameon=True)

        #plt.savefig(f"PCA_After/{Ifrac}/{irun}/{Ioffval}/{i}_Entire_space.pdf", format='pdf')
        plt.savefig(f"5_FFig/{irun}/PCA_After/{Ioffval}/{i}_Entire_space.pdf", format='pdf')
        plt.close()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    return elapsed_time

def PlotResult(IIIMat, IIInpress, IIInmat, IIIfrac, IIIirun, IIbesthp, \
    IIP, IIM, IIP_l, IIM_l, ITrue_Test, IPred_Test, IPred_l_Test, IPred_b_Test, \
    ITrue_Train, IPred_Train, IPred_l_Train, IPred_b_Train, IIMMat_pred, \
    IIMMat_l_pred, IIMMat_b_pred, IIimat, \
    IIidx_unknown_frac, IIidx_known_frac,\
    IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, \
    IIIoffval):
    
    """

    Description:

        1.) This function generates all the important plots after a GLRM run

        2.) It then plots the heatmaps of model predictions, parity plots for
            latent-bias, latent-only, and bias-only cases, and performs PCA
            on the learned latent vectors followed by PCA visualization

        3.) The required latent matrices and their 2D PCA representations
            are then returned for further use

    """

    # Heatmap for normalizes deployable model predictions 
    # For correct comparision the IIIMat has to be standardize w.r.t train set mean and std
    # For hint please look at RUN() and how the MMat_stan is created
    PlotHeatmap(pd.DataFrame(IIMMat_pred.T), "LB", IIIfrac, IIIirun)
    PlotHeatmap(pd.DataFrame(IIMMat_l_pred.T), "LO", IIIfrac, IIIirun)
    PlotHeatmap(pd.DataFrame(IIMMat_b_pred.T), "BO", IIIfrac, IIIirun)

    # These are unnormalized versions of the model predicts
    # I think these can be erroneous. Use with caution
    PlotHeatmap(pd.DataFrame(IIUNMMat_pred.T), "UNLB", IIIfrac, IIIirun)
    PlotHeatmap(pd.DataFrame(IIUNMMat_l_pred.T), "UNLO", IIIfrac, IIIirun)
    PlotHeatmap(pd.DataFrame(IIUNMMat_b_pred.T), "UNBO", IIIfrac, IIIirun)

    bias = "latent-bias"
    ETime = ParityPlot(ITrue_Test, ITrue_Train, IPred_Test, IPred_Train, bias, IIIfrac, IIIirun)

    print("\n================================================================================")
    print("\033[92m[DEBUG XX:] Parity plots are plotted for latent-bias case.\033[0m")
    print(f"Time taken for ploting parity plots for latent-bias case is: {ETime:.4f} minutes")
    print("================================================================================\n")

    bias = "latent-only"
    ETime = ParityPlot(ITrue_Test, ITrue_Train, IPred_l_Test, IPred_l_Train, bias, IIIfrac, IIIirun)

    print("\n================================================================================")
    print("\033[92m[DEBUG XX:] Parity plots are plotted for latent-only case.\033[0m")
    print(f"Time taken for ploting parity plots for latent-only case is: {ETime:.4f} minutes")
    print("================================================================================\n")

    bias = "bias-only"
    ETime = ParityPlot(ITrue_Test, ITrue_Train, IPred_b_Test, IPred_b_Train, bias, IIIfrac, IIIirun)

    print("\n================================================================================")
    print("\033[92m[DEBUG XX:] Parity plots are plotted for bias-only case.\033[0m")
    print(f"Time taken for ploting parity plots for bias-only case is: {ETime:.4f} minutes")
    print("================================================================================\n")

    #------------------------------------------------------------------------------------------#
    
    # WARNING !!!!!  
    # Following lines of code takes large amount of time and memory to run
    # Use only if required

    if IIIirun == 0:
        #ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, Iaxis='SEMILOG') # Normalized plots
        ##ETime = PlotIsothermParity(IIimat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, Iaxis='UN_SEMILOG') # Unnormalized plots. This may be erroneous. Use with caution
        #ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, Iaxis='UN_SEMILOG') # Unnormalized plots. This may be erroneous. Use with caution
        #print("\n================================================================================")
        #print("\033[92m[DEBUG XX:] PlotIsothermParity() ran successfully for bias-only case for SEMILOG case\033[0m")
        #print(f"Time taken for ploting isotherm parity plots for bias-only case for SEMILOG case is: {ETime:.4f} minutes")
        #print("================================================================================\n")

        #ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, Iaxis='LINEAR') # Normalized plots
        ##ETime = PlotIsothermParity(IIimat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution
        #ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution

        ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=True, Iaxis='LINEAR') # Normalized plots
        
        ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=True, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution

        ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=False, Iaxis='LINEAR') # Normalized plots

        ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInpress, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=False, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution

        print("\n================================================================================")
        print("\033[92m[DEBUG XX:] PlotIsothermParity() ran successfully for latent-only case for LINEAR case\033[0m")
        print(f"Time taken for ploting isotherm parity plots for latent-only case for LINEAR case is: {ETime:.4f} minutes")
        print("================================================================================\n")

    #------------------------------------------------------------------------------------------#

    RMAct, RPAct, RAllLatVec, Rm_vecs_2D, Rp_vecs_2D = PCAAfter(IIM, IIP, IIbesthp, IIInmat, IIInpress, True)
    RMAct_l, RPAct_l, RAllLatVec_l, Rm_vecs_2D_l, Rp_vecs_2D_l = PCAAfter(IIM_l, IIP_l, IIbesthp, IIInmat, IIInpress, False)
    
    PlotPCAAfter(Rp_vecs_2D, Rm_vecs_2D, IIIMat, IIInmat, IIInpress, pressure, IIIfrac, IIIirun, True)
    
    #raise ValueError(f"DEBUGGING GOING !!!!")

    return RMAct, RPAct, RAllLatVec, Rm_vecs_2D, Rp_vecs_2D

def SaveResult(IIMat, IIfrac, IInpress, IInmat, Iirun, \
    Iidx_unknown_frac, Iidx_known_frac, Ibesthp, \
    IP, IM, IP_l, IM_l, IP_b, IM_b, \
    IMMat_pred, IMMat_l_pred, IMMat_b_pred, \
    IMMat_stan_mean, IMMat_stan_std, IMMat_stan, \
    IUNMMat_pred, IUNMMat_l_pred, IUNMMat_b_pred, \
    Iimat, Iplot_RMSD, Iplot_R2, IIoffval):

    """

    Description:

        This function collects all the important quantities from a single GLRM run,
        computes the required train and test metrics (R^2 and RMSD), calls PlotResults(), 
        and stores everything which includes model parameters,predictions, metrics, PCA vectors, 
        etc into a dictionary so that it can be saved as a pickle file for subsequent analysis

    """

    True_Test    = [Iimat[p, m] for (p, m) in Iidx_unknown_frac]         # True normalized values present in test set
    Pred_Test    = [IMMat_pred[p, m] for (p, m) in Iidx_unknown_frac]    # Latent-bias Model predicted normalized values present in test set 
    Pred_l_Test  = [IMMat_l_pred[p, m] for (p, m) in Iidx_unknown_frac]  # latent only model predicted normalized values present in test set
    Pred_b_Test  = [IMMat_b_pred[p, m] for (p, m) in Iidx_unknown_frac]  # Bias only model predicted normalized values present in test set

    True_Train   = [Iimat[p, m] for (p, m) in Iidx_known_frac]           # True normalized values present in train set
    Pred_Train   = [IMMat_pred[p, m] for (p, m) in Iidx_known_frac]      # Latent-bias Model predicted normalized values present in train set 
    Pred_l_Train = [IMMat_l_pred[p, m] for (p, m) in Iidx_known_frac]    # Latent only model predicted normalized values present in train set
    Pred_b_Train = [IMMat_b_pred[p, m] for (p, m) in Iidx_known_frac]    # Bias only model predicted normalized values present in train set

    # Calculating RMSD and R2 between true and predicted values for test set for final latent-bias model
    TestMat      = compute_metrics(True_Test, Pred_Test)

    # Calculating RMSD and R2 between true and predicted values for test set for final latent only model
    TestMat_l    = compute_metrics(True_Test, Pred_l_Test)

    # Calculating RMSD and R2 between true and predicted values for test set for final bias only model
    TestMat_b    = compute_metrics(True_Test, Pred_b_Test)

    # Calculating RMSD and R2 between true and predicted values for train set for final latent-bias model
    TrainMat     = compute_metrics(True_Train, Pred_Train)

    # Calculating RMSD and R2 between true and predicted values for train set for final latent only model
    TrainMat_l   = compute_metrics(True_Train, Pred_l_Train)

    # Calculating RMSD and R2 between true and predicted values for train set for final bias only model
    TrainMat_b   = compute_metrics(True_Train, Pred_b_Train)

    print("\n================================================================================")
    
    print(f"\033[92mFinal glrm model is successfully constructed and predictions are obtained for irun = {Iirun} and Ifrac = {IIfrac}\n\033[0m")

    print(f"R2_Train : {TrainMat['R²']:.4f} and RMSD_Train : {TrainMat['RMSE']:.4f} for latent-bias model")
    print(f"R2_Test  : {TestMat['R²']:.4f}  and RMSD_Test  : {TestMat['RMSE']:.4f}  for latent-bias model")

    print("\n")

    print(f"$R2_Train_l : {TrainMat_l['R²']:.4f} and RMSD_Train_l : {TrainMat_l['RMSE']:.4f} for latent-only model")
    print(f"R2_Test_l   : {TestMat_l['R²']:.4f}  and RMSD_Test_l  : {TestMat_l['RMSE']:.4f}  for latent-only model")

    print("\n")

    print(f"R2_Train_b : {TrainMat_b['R²']:.4f} and RMSD_Train_b : {TrainMat_b['RMSE']:.4f} for bias-only model")
    print(f"R2_Test_b  : {TestMat_b['R²']:.4f}  and RMSD_Test_b  : {TestMat_b['RMSE']:.4f}  for bias-only model")

    print("================================================================================\n")

    # Saving the key plots

    MAct, PAct, AllLatVec, m_vecs_2D, p_vecs_2D = PlotResult(IIMat, IInpress, \
        IInmat, IIfrac, Iirun, Ibesthp, IP, IM, IP_l, IM_l, True_Test, Pred_Test, \
        Pred_l_Test, Pred_b_Test, True_Train, Pred_Train, Pred_l_Train, \
        Pred_b_Train, IMMat_pred, IMMat_l_pred, IMMat_b_pred, Iimat, \
        Iidx_unknown_frac, Iidx_known_frac, IUNMMat_pred, IUNMMat_l_pred, IUNMMat_b_pred, IIoffval)

    Result = {
    "RIfrac": float(IIfrac),                      # Fraction of known entries
    "RInpress": int(IInpress),                    # # of Pressure points
    "RInmat": int(IInmat),                        # # of Materials
    "Rbesthp": dict(Ibesthp),                     # Best hyperparameter for Tfrac and irun
    "Rplot_RMSD": list(Iplot_RMSD),               # How RMSD validation is varying for each hyperparameter
    "Rplot_R2": list(Iplot_R2),                   # How R2 validation is varying for each hyperparameter
    "RP": np.array(IP),                           # Deployable glrm model for latent-bias case P matrix 
    "RM": np.array(IM),                           # Deployable glrm model for latent-bias case M matrix 
    "RP_b": np.array(IP_b),                       # Deployable glrm model for bias-only   case P matrix 
    "RM_b": np.array(IM_b),                       # Deployable glrm model for bias-only   case M matrix 
    "RP_l": np.array(IP_l),                       # Deployable glrm model for latent-only case P matrix 
    "RM_l": np.array(IM_l),                       # Deployable glrm model for latent-only case M matrix 
    "RMMat_pred": np.array(IMMat_pred),           # Predictions of deplyoable glrm model for latent-bias case on train + test case
    "RMMat_b_pred": np.array(IMMat_b_pred),       # Predictions of deplyoable glrm model for bias-only   case on train + test case
    "RMMat_l_pred": np.array(IMMat_l_pred),       # Predictions of deplyoable glrm model for latent-only   case on train + test case
    "RIimat": np.array(Iimat),                    # imat
    "RMMat_stan_mean": np.array(IMMat_stan_mean), # Mean matrix of incomplete matrix 
    "RMMat_stan_std": np.array(IMMat_stan_std),   # Std matrix of incomplete matrix
    "RMMat_stan": np.array(IMMat_stan),           # Standardized incomplete matrix
    "Ridx_known": list(Iidx_known_frac),          # Indices of known entries (Train set)
    "Ridx_unknown": list(Iidx_unknown_frac),      # Indices of missing entries (Test set)
    "RTrue_Test": np.array(True_Test),            # True normalized values present in test set    
    "RPred_Test": np.array(Pred_Test),            # Latent-bias Model predicted normalized values present in test set 
    "RPred_b_Test": np.array(Pred_b_Test),        # Bias only model predicted normalized values present in test set
    "RPred_l_Test": np.array(Pred_l_Test),        # Latent only model predicted normalized values present in test set
    "RTrue_Train": np.array(True_Train),          # True normalized values present in train set
    "RPred_Train": np.array(Pred_Train),          # Latent-bias Model predicted normalized values present in train set 
    "RPred_b_Train": np.array(Pred_b_Train),      # Bias only model predicted normalized values present in train set
    "RPred_l_Train": np.array(Pred_l_Train),      # Latent only model predicted normalized values present in train set
    "RAmp_T": np.array(IIMat),                    # Original Property-Material matrix
    "RTestMat": TestMat,                          # Prediction of deployable latent-bias glrm model on test set
    "RTestMat_b": TestMat_b,                      # Prediction of deployable bias-only glrm model on test set
    "RTestMat_l": TestMat_l,                      # Prediction of deployable latent-only glrm model on test set
    "RTrainMat": TrainMat,                        # Prediction of deployable latent-bias glrm model on train set
    "RTrainMat_b": TrainMat_b,                    # Prediction of deployable bias-only glrm model on train set
    "RTrainMat_l": TrainMat_l,                    # Prediction of deployable latent-only glrm model on train set
    "RMAct": MAct,                                # Removing material bias from M to obtain MAct
    "RPAct": PAct,                                # Removing column of 1's corresponding to material bias from P to obtain PAct
    "RAllLatVec": AllLatVec,                      # Horizontal staking of RMAct and RPAct
    "Rm_vecs_2D": m_vecs_2D,                      # PCA reduced 2D representation of MAct
    "Rp_vecs_2D": p_vecs_2D,                      # PCA reduced 2D representation of PAct
    
    }

    return Result

def RUN(Ifrac, IMat, Inpress, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat, irun):
    
    """

    Description:

            This function performs following tasks:

                1.) Create incomplete Pressure-material matrix using Ifrac
                
                2.) Standardize this incomplete Pressure-material matrix
                
                3.) Split the known entries into train and validation set
                
                4.) Do necessary cross checking using assertion
                
                5.) Call HPTune() to carry out hyper-parameter tunning and get best hyper-parameters
                
                6.) Build the model on the train set which is combination of train and validation sets defined earlier and gauge its performance on test set
                
                7.) Do same for bias only model
                
                8.) Compare the model performace on test set entries

                9.) Generates all the nessesary data useful for further analysis and calls SaveResult()
                    to save this data

    Input:
                Ifrac       ==>     Same as before (Ifrac = float eg Ifrac = 0.7)
                
                IMat        ==>     Input Pressure-material matrix. (IMat = Amp_T)
                
                Inpress     ==>     Same as before 
                
                Inmat       ==>     Same as before
                
                Ihpg        ==>     Hyper parameter grid
                
                Ioffval     ==>     Same as before
                
                Isample     ==>     Same as before
                
                Iori_mean   ==>     Mean of filled Pressure-material matrix before standardization (Iori_mean = Amp_T_mean)
                
                Iori_std    ==>     Std of filled Pressure-material matrix before standardization (Iori_std = Amp_T_std)
                
                Iori_mat    ==>     Pressure-material matrix before standardization (Iori_Mat = Amp_T)

                irun        ==>     ith run out of nrun for the same theta to ensure statistical accuracy

    Output:

                A pickel named Results that stores all the data

    """

    print("\n======================================================================================")
    print(f"\n\n\t\033[92m[DEBUG 13:] RUN() is running for Ifrac = {Ifrac} and irun = {irun}\033[0m\n")
    print("======================================================================================\n")

    # Step 1: Creating Missing Pressure-material matrix with frac = Ifrac
    [MMat, idx_known_frac, idx_unknown_frac] = PropMatMissing(Ifrac, IMat, Inmat, Inpress, Isample, None, None, None, None)

    # Step 2: Standardizing the missing Pressure material matrix
    MMat_stan_mean, MMat_stan_std, MMat_stan = standardize_matrix(MMat, 1)

    # Step 3 Cross checking to ensure Pressure-material matrix has NaN and non NaN values at appropriate places.
    for ip, im in idx_unknown_frac:
        assert np.isnan(MMat_stan[ip, im]), f"\033[91mMismatch in entries of unknown indices ({ip},{im}) is not NaN\033[0m"

    for ip, im in idx_known_frac:
        assert not np.isnan(MMat_stan[ip, im]), f"\033[91mMismatch in entries of known indices: Known entry ({ip},{im}) is NaN\033[0m"

    # Step 4: Splitting the data into train (80%) and validation (20%) 
    #idx_train_frac, idx_val_frac = train_test_split(idx_known_frac, test_size=0.2, shuffle=True, random_state=SEED)
    idx_train_frac, idx_val_frac = train_test_split(idx_known_frac, test_size=0.2, shuffle=True)

    # Step 5 Cross checking to ensure Pressure-material matrix has NaN and non NaN values at appriate places 
    for ip, im in idx_train_frac:
        assert not np.isnan(MMat_stan[ip, im]), f"\033[91mMismatch in entries of known indices: Known entry ({ip},{im}) is NaN\033[0m"

    for ip, im in idx_val_frac:
        assert not np.isnan(MMat_stan[ip, im]), f"\033[91mMismatch in entries of known indices: Known entry ({ip},{im}) is NaN\033[0m"

    # Step 6: Hyperparameter tunning on train-validation set
    # Finding best hyper-parameter on train-validation set for latent-bias case
    
    print("\n======================================================================================")
    print(f"\n\n\tHyper-Parameter tunning started for latent-bias case for Ifrac = {Ifrac} and irun = {irun}\n")
    print("======================================================================================\n")

    besthp, RMSD_val_hpg, RMSD_train_hpg, R2_val_hpg, R2_train_hpg, plot_RMSD, plot_R2 = HPTune(MMat_stan, Inpress, Inmat, Ihpg, idx_train_frac, idx_val_frac, Ifrac, True, irun)

    # Step 7: Train the final latent-bias model on train + validation set datapoint and check its performance on test set
    
    print("\n======================================================================================")
    print(f"\n\n\tTraining final latent-bias model on new training set for Ifrac = {Ifrac}\n\n")
    print("======================================================================================\n")
    
    P, M, model, ch = fit_lrm(MMat_stan, Inpress, Inmat, besthp, idx_known_frac, None, None, True)

    # Step 8: Use the same best hyperparamter as that of latent-bias case and  
    # train the final latent-only model on train + validation set datapoint and 
    #check its performance on test set

    print("\n======================================================================================")
    print(f"\n\n\tTraining final latent-only model on new training set using the best hyperparameters obtained for latent-bias case for Ifrac = {Ifrac}\n\n")
    print("======================================================================================\n")

    P_l, M_l, model_l, ch_l = fit_lrm(MMat_stan, Inpress, Inmat, besthp, idx_known_frac, None, None, False)

    hpara_bias = {"dim": 0, "reg": 0.00}

    print("\n======================================================================================")
    print(f"\n\n\tTraining bias only model on new training set for Ifrac = {Ifrac}\n\n")
    print("======================================================================================\n")

    P_b, M_b, model_b, ch_b = fit_lrm(MMat_stan, Inpress, Inmat, hpara_bias, idx_known_frac, None, None, True)

    # Impute the entire matrix based on latent-bias, latent-only and bias-only model
    MMat_pred      = model.impute()
    MMat_l_pred    = model_l.impute()
    MMat_b_pred    = model_b.impute()
    
    # Since the Pressure-material matrix is the same, its mean and std should remain the same
    # The imputed matrix will be standardized hence, to check how good are the imputations with respect to original matrix we need to standardize the original matrix also.
    imean, istd, imat = standardize_matrix(IMat, 1)

    assert np.allclose(Iori_mean, imean, atol = 1e-6),\
            f"\033[91mMismatch in Mean vector of original matrix\033[0m"

    assert np.allclose(Iori_std, istd, atol = 1e-6),\
            f"\033[91mMismatch in std vector of original matrix\033[0m"

    assert np.allclose(Iori_mat, imat, atol = 1e-6),\
            f"\033[91mMismatch in the original standardized matrix\033[0m"

    UNIMat = unstandardize_matrix(imat, imean, istd)
    #MMat_stan_mean, MMat_stan_std, MMat_stan = standardize_matrix(MMat, 1)

    assert np.allclose(IMat, UNIMat, atol = 1e-6),\
            f"\033[91mMismatch in the reconstructed unnormalized matrix\033[0m"

    UNMMat_pred   = unstandardize_matrix(MMat_pred, MMat_stan_mean, MMat_stan_std)
    UNMMat_l_pred = unstandardize_matrix(MMat_l_pred, MMat_stan_mean, MMat_stan_std)
    UNMMat_b_pred = unstandardize_matrix(MMat_b_pred, MMat_stan_mean, MMat_stan_std)

    Result = SaveResult(IMat, Ifrac, Inpress, Inmat, irun, \
        idx_unknown_frac, idx_known_frac, besthp, \
        P, M, P_l, M_l, P_b, M_b, \
        MMat_pred, MMat_l_pred, MMat_b_pred, \
        MMat_stan_mean, MMat_stan_std, MMat_stan, \
        UNMMat_pred, UNMMat_l_pred, UNMMat_b_pred, \
        imat, plot_RMSD, plot_R2, Ioffval)

    return Result

def RUNM(Ifrac, IMat, Inpress, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat, nrun):
    
    """

    Description:
    
            Aim is to run RUNM multiple times for the same Ifrac to account 
            for randomness and save all results in a single pickle.
    
    Inputs:

        Ifrac, IMat, Inpress, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat : as in RUN()
        
        nrun : Number of independent runs to perform
    
    Output:

        Saves a single pickle file containing a list of dictionaries, one per run
    """
    
    AllResults = []

    for irun in range(nrun):
        print(f"\n===== Run {irun + 1}/{nrun} for Ifrac = {Ifrac} =====")
        
        # Call your original RUN function
        iresult = RUN(Ifrac, IMat, Inpress, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat, irun)
        
        AllResults.append(iresult)
        
        # Ensure folder exists
        os.makedirs(f'5_FFig/Statistical_analysis', exist_ok=True)
    
        # Save all runs in a single pickle
        with open(f"5_FFig/Statistical_analysis/{nrun}_result.pkl", "wb") as f:
            pickle.dump(AllResults, f)
    
        print(f"\n\033[92m The {irun + 1} run for Ifrac = {Ifrac} saved successfully.\033[0m")
    
    print(f"\n\033[92m All {nrun} runs for Ifrac = {Ifrac} saved successfully.\033[0m")

##--------------------------------------------------------------------------------------------##

# Reading the data file and initial checks

#Isotherm.csv file contains ==> mof_name,loading1,loading2,....

#df_data = pd.read_csv("sorted-core-2019-ar.csv")
df_data = pd.read_csv("Isotherm.csv")

mof_name = df_data["mof_name"].tolist() 
df_data  = df_data[pressure]

[nmat, npress] = df_data.shape

assert [len(mof_name), len(pressure)] == [nmat, npress], \
        f"\033[91mMismatch in df_data.shape: expected({len(mof_name)}, {len(pressure)}), got ({nmat}, npress)\033[0m" 

print("\n======================================================================================")
print("\033[92m[DEBUG 1:] df_data.shape Test passed.\033[0m")
print("======================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Calculating pearson correlation matrix

start_time = time.time()
#PlotHeatmap(df_data, "ORI", 1.0, "1_NO_irun")
end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 01:] PlotHeatmap ran successfully.\033[0m")
print(f"\tTime taken to run PlotHeatmap is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Visualizing Pearson correlation calculated in PlotHeatmap()

start_time = time.time()
#PearsonPlot(df_data.select_dtypes(include='number'), pressure)
end_time = time.time()
elapsed_time = (end_time - start_time) / 60


print("\n======================================================================================")
print("\033[92m[DEBUG 02:] PearsonPlot ran successfully.\033[0m")
print(f"\tTime taken to run PearsonPlot is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

##raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

# Visualizing adsorption isothem in semilog axis for all the materials under consideration
# WARNING !!!! This line of code may take lot of time and memory

start_time = time.time()
#PlotIsotherm(df_data.select_dtypes(include='number'), mof_name, "SEMILOG")
end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 03:] PlotIsotherm ran successfully for SEMILOG case.\033[0m")
print(f"\tTime taken to run PlotIsotherm for SEMILOG case is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

##raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

# Visualizing adsorption isothem in semilog axis for all the materials under consideration.

start_time = time.time()
#PlotIsotherm(df_data.select_dtypes(include='number'), mof_name, "LINEAR")
end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 04:]PlotIsotherm ran successfully for LINEAR case.\033[0m")
print(f"\tTime taken to run PlotIsotherm for LINEAR case is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

##raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

# PCA plots before fitting GLRM

start_time = time.time()
PlotPCABefore(df_data.select_dtypes(include='number'), pressure)
end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 05:] PlotPCABefore ran successfully.\033[0m")
print(f"\tTime taken to run PlotPCABefore is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

#raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

# Creating material-Pressure matrix
Amp = df_data.to_numpy() # shape: (nmat,npress)

assert Amp.shape == (nmat, npress), \
        f"\033[91mMismatch in Amp.shape: expected ({nmat}, {npress}), got {Amp.shape}\033[0m"

# Taking transpose of material-pressure matrix, becasue bias can only be added as columns in LowRanksModel()

# Since Julia implimentation of pyglrm requires the offsets to be added to the columns hence, we will be working with the transpose of the material Pressure matrix (Amp) which is Pressure-material matrix (Amp_T)

# Remember when we were trying to do PCA we learned that we have to remove mean from the columns of original matrix. Here the original matrix is Amp so we must remove mean from columns of Amp i.e. we must remove mean from rows of Amp.T i.e. Amp_T

Amp_T = Amp.T # shape: (npress,nmat)

assert Amp_T.shape == (npress, nmat), \
        f"\033[91mMismatch in Amp_T.shape: expected ({npress}, {nmat}), got {Amp_T.shape}\033[0m"
print("\n======================================================================================")
print(f"\033[92m[DEBUG 06:] Material Pressure matrix (Amp) with size {Amp.shape} and Pressure material matrix (Amp_T) with size {Amp_T.shape} successfully created.\033[0m")
print("======================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Standardize the given matrix. This case handles the case where elements of matrix are missing or NaN

start_time = time.time()

Amp_mean, Amp_std, Amp_stan       = standardize_matrix(Amp, 0)
Amp_T_mean, Amp_T_std, Amp_T_stan = standardize_matrix(Amp_T, 1)

assert np.allclose(Amp_stan.T, Amp_T_stan, atol = 1e-6),\
        f"\033[91mMismatch in Amp_stan.T and Amp_T_stan\033[0m"

end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print(f"\033[92m[DEBUG 07:] Standardization of Amp and Amp_T is complete.\033[0m")
print(f"\tTime taken to run Standardization of Amp and Amp_T is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

#raise ValueError(f"DEBUGGING GOING !!!!")

sample = "Random"

[Amp_T_incomplete_rand, idx_known_rand, idx_unknown_rand] = PropMatMissing(0.4, Amp_T, nmat, npress, sample, None, None, None, None)

# Should be commented out while testing
[Amp_T_incomplete_rand_1, idx_known_rand_1, idx_unknown_rand_1] = PropMatMissing(0.4, Amp_T, nmat, npress, sample, None, None, None, None)

# Should be commented out while testing
mask  = np.isnan(Amp_T_incomplete_rand)
mask1 = np.isnan(Amp_T_incomplete_rand_1)

# Should be commented out while testing
assert not np.array_equal(mask, mask1),\
        f"\033[91mSampling pattern did not change.\033[0m"

print("\n======================================================================================")
print("\033[92m[DEBUG 08: RANDOM] PropMatMissing ran for RANDOM case and incomplete Pressure-material matrix is generated successfully.\033[0m")
print("======================================================================================\n")

# Should be commented out while testing
np.savetxt("Amp_T_incomplete_rand.csv", Amp_T_incomplete_rand, delimiter=",")
np.savetxt("Amp_T_incomplete_rand_1.csv", Amp_T_incomplete_rand_1, delimiter=",")

Amp_T_incomplete = Amp_T_incomplete_rand
idx_known        = idx_known_rand
idx_unknown      = idx_unknown_rand

# Should be commented out while testing
del Amp_T_incomplete_rand_1, idx_known_rand_1, idx_unknown_rand_1

# Should be commented out while testing
del mask, mask1

del Amp_T_incomplete_rand, idx_known_rand, idx_unknown_rand

###############################################################################################

start_time = time.time()

Amp_T_incomplete_mean, Amp_T_incomplete_std, Amp_T_incomplete_stan           = standardize_matrix(Amp_T_incomplete, 1)

Amp_T_incomplete_T_mean, Amp_T_incomplete_T_std, Amp_T_incomplete_stan_T     = standardize_matrix(Amp_T_incomplete.T, 0)

Temp = unstandardize_matrix(Amp_T_incomplete_stan, Amp_T_incomplete_mean, Amp_T_incomplete_std)

assert np.allclose(Amp_T_incomplete, Temp, equal_nan=True), "\033[91mMismatch found between Amp_T_incomplete and Temp\033[0m"

del Temp

#raise ValueError(f"Testing going on")

#assert np.allclose(Amp_T_incomplete_stan.T, Amp_T_incomplete_stan_T, atol = 1e-6),\
#        f"\033[91mMismatch in Amp_stan.T and Amp_T_stan\033[0m"

end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 09:] Standardization of incomplete Pressure-material matrix is done and Amp_T_incomplete_stan is successfully created.\033[0m")
print("======================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Checking whether fit_lrm() works or not. 

offset = True
#offset = False

GLRMTest(False, Amp_T, nmat, npress, sample)
GLRMTest(True, Amp_T, nmat, npress, sample)

##--------------------------------------------------------------------------------------------##

# Creating hyper-parameter grid

# Generate the reg values: 10^1.0 to 10^3.0 in 25 log-spaced steps
reg = np.logspace(1.0, 3.0, num = 25)  # 10^1 to 10^3 num = 25
reg = np.round(reg).astype(int)
dim = range(1, npress)  # 1 to npress - 1 range(1, npress)

# Creating hyper-parameter grid
hpg = []

for idim in dim:
    for ireg in reg:
        hpg.append({"dim": idim, "reg": ireg})

for ihpg in hpg:
    print(f"Dimention: {ihpg['dim']}, Regularization: {ihpg['reg']}")

print(f"Printing specific element of hpg : {hpg[0]['dim']} and {hpg[0]['reg']}")

#assert len(hpg) == (npress - 1) * len(reg), \
#        f"\033[91mMistmatch in hyperparameter grid size. Expected: {(npress - 1) * len(reg)}. Got : {len(hpg)}\033[0m"

print("\n================================================================================")
print("\033[92m [DEBUG 11:] Hyper-parameter grid constructed sucessfullly!!!\033[0m")
print("=================================================================================\n")

print("\n================================================================================")
print(f"\033[92m[DEBUG 12:] RUNM() running for Ifrac = 0.5 !!!\033[0m")
print("=================================================================================\n")

RUNM(0.5, Amp_T, npress, nmat, hpg, offset, sample, Amp_T_mean, Amp_T_std, Amp_T_stan, 10)

print("\n================================================================================")
print("\033[92m[DEBUG 8:] RUNM() ran successfully for Ifrac = 0.5 !!!\033[0m")
print("=================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Deplyoable model is created
print(f"DONE")




