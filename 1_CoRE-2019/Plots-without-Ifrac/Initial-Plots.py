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

            valX = float(iprop[i])
            mantissaX, exponentX = f"{valX:.1e}".split("e")
            exponentX = int(exponentX)

            valY = float(iprop[j])
            mantissaY, exponentY = f"{valY:.1e}".split("e")
            exponentY = int(exponentY)

            iCOR = X.corr(Y)

            #plt.scatter(X, Y, color='green', label=f"{iprop[i]} vs {iprop[j]}", alpha=0.6, edgecolor='black', s=80)
            plt.scatter(X, Y, color='blue', label=f"{iprop[i]} vs {iprop[j]}", alpha=0.6, s=20)
            
            plt.plot([min(min(X), min(Y)), max(max(X), max(Y))],[min(min(X), min(Y)), max(max(X), max(Y))],color='gray', linestyle='--', label='Perfect Prediction')

            plt.xticks(fontsize=26,fontweight='bold')
            plt.yticks(fontsize=26,fontweight='bold')

            #plt.xlabel(f"Uptake of Ar at {iprop[i]}", fontdict=fontdict_x)
            #plt.ylabel(f"Uptake of Ar at {iprop[j]}", fontdict=fontdict_y)
            
            plt.xlabel(rf"$\mathbf{{Uptake\ of\ Ar\ at\ {mantissaX}\times 10^{{{exponentX}}}\ bar}}$",fontdict=fontdict_x)

            plt.ylabel(rf"$\mathbf{{Uptake\ of\ Ar\ at\ {mantissaY}\times 10^{{{exponentY}}}\ bar}}$",fontdict=fontdict_y)

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

##--------------------------------------------------------------------------------------------##

# Reading the data file and initial checks

#Isotherm.csv file contains ==> mof_name,loading1,loading2,....

df_data = pd.read_csv("sorted-core-2019-ar.csv")
#df_data = pd.read_csv("Isotherm.csv")

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
PlotHeatmap(df_data, "ORI", 1.0, "1_NO_irun")
end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 01:] PlotHeatmap ran successfully.\033[0m")
print(f"\tTime taken to run PlotHeatmap is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Visualizing Pearson correlation calculated in PlotHeatmap()

start_time = time.time()
PearsonPlot(df_data.select_dtypes(include='number'), pressure)
end_time = time.time()
elapsed_time = (end_time - start_time) / 60


print("\n======================================================================================")
print("\033[92m[DEBUG 02:] PearsonPlot ran successfully.\033[0m")
print(f"\tTime taken to run PearsonPlot is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

#raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

# Visualizing adsorption isothem in semilog axis for all the materials under consideration
# WARNING !!!! This line of code may take lot of time and memory

start_time = time.time()
PlotIsotherm(df_data.select_dtypes(include='number'), mof_name, "SEMILOG")
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
PlotIsotherm(df_data.select_dtypes(include='number'), mof_name, "LINEAR")
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

raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

