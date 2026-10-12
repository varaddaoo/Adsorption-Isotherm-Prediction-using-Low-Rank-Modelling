#!/usr/bin/env python
# coding: utf-8

# Importing essential modules

import argparse
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
import re

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
from sklearn.preprocessing import StandardScaler
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

def GetPropertyLabels(df_data):

    """
    Description:

        Here the aim is to process the column names of the data file.

        For adsorption properties, the column name is expected to follow:

            Adsorbate_uptake_(mol/kg)_Pressurebar_TemperatureK

        For example:

            N2_uptake_(mol/kg)_6bar_77K
            CH4_uptake_(mol/kg)_65bar_298K

        The adsorbate, pressure and temperature are extracted from the
        column name. Pressure is converted from bar to Pa and represented
        in scientific notation for plotting.

        For structural properties, predefined labels are assigned.

    Input:
        df_data ==> DataFrame containing mof_name and all properties

    Output:
        cname          ==> Original property column names
        property_label ==> Labels to be used in PCA plots

    """

    # ------------------------------------
    # Column names
    # ------------------------------------

    cname = df_data.columns.drop("mof_name").tolist()

    # ------------------------------------
    # Structural property labels
    # ------------------------------------

    structural_labels = {
        "vf": r'$\mathbf{AVF}$',
        "sa_tot_m2g": r'$\mathbf{GSA\ (m^2/g)}$',
        "sa_tot_m2cm3": r'$\mathbf{VSA\ (m^2/cm^3)}$',
        "pld": r'$\mathbf{PLD\ (Å)}$',
        "lcd": r'$\mathbf{LCD\ (Å)}$'
    }

    # ------------------------------------
    # Property labels
    # ------------------------------------

    property_label = []

    for column in cname:

        # ------------------------------------
        # Adsorption property
        # ------------------------------------

        if "_uptake_" in column:

            parts = column.split("_")

            assert len(parts) == 5, (f"\033[91mInvalid adsorption column name: {column}. Expected format: Adsorbate_uptake_(mol/kg)_Pressurebar_TemperatureK\033[0m")

            Adsorbate = parts[0]
            Pressure = float(parts[3].replace("bar", ""))
            Temperature = int(parts[4].replace("K", ""))

            # Convert pressure from bar to Pa
            Pressure = Pressure * 1.0e5

            # Convert pressure to scientific notation
            mantissa, exponent = f"{Pressure:.1e}".split("e")
            exponent = int(exponent)

            property_label.append(rf'$\mathbf{{{Adsorbate}\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ Pa\ ({Temperature}K)}}$')

        # ------------------------------------
        # Structural property
        # ------------------------------------

        else:

            assert column in structural_labels, (f"\033[91mUnknown property column: {column}. Please add this property to structural_labels.\033[0m")

            property_label.append(structural_labels[column])

    return cname, property_label

def TestGetPropertyLabels():

    """
    Description:

        Test GetPropertyLabels() using predefined column names.

        The test checks:
            1.) Multiple adsorbates and conditions without structural properties.
            2.) Structural properties without adsorption properties.
            3.) Single adsorbate at a single temperature and multiple pressures without structural properties.
            4.) Single adsorbate at a single temperature and multiple pressures with structural properties.
            5.) Malformed adsorption column name is rejected.

    """

    # ------------------------------------
    # Test Case 1:
    # Multiple adsorbates and conditions
    # without structural properties
    # ------------------------------------

    df_test = pd.DataFrame(columns=[
        "mof_name",
        "N2_uptake_(mol/kg)_6bar_77K",
        "N2_uptake_(mol/kg)_8bar_77K",
        "CH4_uptake_(mol/kg)_65bar_298K",
        "H2_uptake_(mol/kg)_100bar_77K",
        "Xe_uptake_(mol/kg)_1bar_298K"
    ])

    cname, property_label = GetPropertyLabels(df_test)

    expected_cname = [
        "N2_uptake_(mol/kg)_6bar_77K",
        "N2_uptake_(mol/kg)_8bar_77K",
        "CH4_uptake_(mol/kg)_65bar_298K",
        "H2_uptake_(mol/kg)_100bar_77K",
        "Xe_uptake_(mol/kg)_1bar_298K"
    ]

    expected_property_label = [
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 6.0 \times 10^{5}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 8.0 \times 10^{5}\ Pa\ (77K)}$',
        r'$\mathbf{CH4\ Uptake\ at\ Pressure\ 6.5 \times 10^{6}\ Pa\ (298K)}$',
        r'$\mathbf{H2\ Uptake\ at\ Pressure\ 1.0 \times 10^{7}\ Pa\ (77K)}$',
        r'$\mathbf{Xe\ Uptake\ at\ Pressure\ 1.0 \times 10^{5}\ Pa\ (298K)}$'
    ]

    assert cname == expected_cname, f"Test 1 Failed: cname mismatch"
    assert property_label == expected_property_label, f"Test 1 Failed: property_label mismatch"


    # ------------------------------------
    # Test Case 2:
    # Only structural properties
    # ------------------------------------

    df_test = pd.DataFrame(columns=[
        "mof_name",
        "vf",
        "sa_tot_m2g",
        "sa_tot_m2cm3",
        "pld",
        "lcd"
    ])

    cname, property_label = GetPropertyLabels(df_test)

    expected_cname = [
        "vf",
        "sa_tot_m2g",
        "sa_tot_m2cm3",
        "pld",
        "lcd"
    ]

    expected_property_label = [
        r'$\mathbf{AVF}$',
        r'$\mathbf{GSA\ (m^2/g)}$',
        r'$\mathbf{VSA\ (m^2/cm^3)}$',
        r'$\mathbf{PLD\ (Å)}$',
        r'$\mathbf{LCD\ (Å)}$'
    ]

    assert cname == expected_cname, f"Test 2 Failed: cname mismatch. Ecpected {expected_cname} but got {cname}"
    assert property_label == expected_property_label, f"Test 2 Failed: property_label mismatch. Expected {expected_property_label} got {property_label}"


    # ------------------------------------
    # Test Case 3:
    # Single adsorbate, single temperature,
    # multiple pressures without structural properties
    # ------------------------------------

    df_test = pd.DataFrame(columns=[
        "mof_name",
        "N2_uptake_(mol/kg)_0.001bar_77K",
        "N2_uptake_(mol/kg)_0.01bar_77K",
        "N2_uptake_(mol/kg)_0.1bar_77K",
        "N2_uptake_(mol/kg)_1bar_77K",
        "N2_uptake_(mol/kg)_10bar_77K"
    ])

    cname, property_label = GetPropertyLabels(df_test)

    expected_cname = [
        "N2_uptake_(mol/kg)_0.001bar_77K",
        "N2_uptake_(mol/kg)_0.01bar_77K",
        "N2_uptake_(mol/kg)_0.1bar_77K",
        "N2_uptake_(mol/kg)_1bar_77K",
        "N2_uptake_(mol/kg)_10bar_77K"
    ]

    expected_property_label = [
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{2}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{3}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{4}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{5}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{6}\ Pa\ (77K)}$'
    ]

    assert cname == expected_cname, f"Test 3 Failed: cname mismatch"
    assert property_label == expected_property_label, f"Test 3 Failed: property_label mismatch"


    # ------------------------------------
    # Test Case 4:
    # Single adsorbate, single temperature,
    # multiple pressures with structural properties
    # ------------------------------------

    df_test = pd.DataFrame(columns=[
        "mof_name",
        "vf",
        "sa_tot_m2g",
        "pld",
        "lcd",
        "N2_uptake_(mol/kg)_0.001bar_77K",
        "N2_uptake_(mol/kg)_0.01bar_77K",
        "N2_uptake_(mol/kg)_0.1bar_77K",
        "N2_uptake_(mol/kg)_1bar_77K",
        "N2_uptake_(mol/kg)_10bar_77K"
    ])

    cname, property_label = GetPropertyLabels(df_test)

    expected_cname = [
        "vf",
        "sa_tot_m2g",
        "pld",
        "lcd",
        "N2_uptake_(mol/kg)_0.001bar_77K",
        "N2_uptake_(mol/kg)_0.01bar_77K",
        "N2_uptake_(mol/kg)_0.1bar_77K",
        "N2_uptake_(mol/kg)_1bar_77K",
        "N2_uptake_(mol/kg)_10bar_77K"
    ]

    expected_property_label = [
        r'$\mathbf{AVF}$',
        r'$\mathbf{GSA\ (m^2/g)}$',
        r'$\mathbf{PLD\ (Å)}$',
        r'$\mathbf{LCD\ (Å)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{2}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{3}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{4}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{5}\ Pa\ (77K)}$',
        r'$\mathbf{N2\ Uptake\ at\ Pressure\ 1.0 \times 10^{6}\ Pa\ (77K)}$'
    ]

    assert cname == expected_cname, f"Test 4 Failed: cname mismatch"
    assert property_label == expected_property_label, f"Test 4 Failed: property_label mismatch"


    # ------------------------------------
    # Test Case 5:
    # Malformed adsorption column
    # ------------------------------------

    df_test = pd.DataFrame(columns=[
        "mof_name",
        "N2_uptake_(mol/kg)_6bar"
    ])

    try:

        cname, property_label = GetPropertyLabels(df_test)

        assert False, "Test 5 Failed: Malformed adsorption column was accepted"

    except AssertionError as error:

        if "Invalid adsorption column name" not in str(error):

            raise error


    print("\n======================================================================================")
    print("\033[92m[TEST:] GetPropertyLabels() test passed for all predefined cases.\033[0m")
    print("======================================================================================\n")

def PlotPCABefore(df):

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
            df     ==> DataFrame of shape (nmat, nprop)

    Ouput:
            PCA plots in folder `1_PCA/1_Before`

    """

    os.makedirs('PCA_Before/', exist_ok=True)

    # Material Case

    [inmat, inprop] = df.shape

	# Proprety vector
	# Property Case
    dg = df.T

    # Material Case
    norm       = MinMaxScaler()
    Mat_norm   = norm.fit_transform(df) # after norm the datatype changes from dataframe to numpy array

    # Property Case
    norm_T     = MinMaxScaler()
    Prop_norm = norm_T.fit_transform(dg) # after norm the datatype changes from dataframe to numpy array

    # Material Case
    pca        = PCA(n_components=2)
    Mat_pca    = pca.fit_transform(Mat_norm)

    # Property Case
    pca_T      = PCA(n_components=2)
    Prop_pca  = pca_T.fit_transform(Prop_norm)

    # Material Case
    cmap = get_cmap('rainbow', 7)

	# Property Case
    colors = plt.cm.get_cmap('hsv', inprop)

    # Property Case
    norm_pca = np.linalg.norm(Prop_pca, axis=1)  # Find the magnitude of each vector
    normalized_pca = Prop_pca / norm_pca[:, np.newaxis]  # Normalize each vector by its magnitude

    for i in range(inprop):

        plt.figure(figsize = (11,7))

        # Material Case

        sc = plt.scatter(Mat_pca[:,0], Mat_pca[:,1], s=30, c=Mat_norm[:,i], cmap = cmap, edgecolor='black', linewidth=1, alpha=0.75)

        # Plotting X and Y axis
        plt.axhline(0, color='black', linewidth=2)  # x-axis
        plt.axvline(0, color='black', linewidth=2)  # y-axis

        #val = float(Icname[i])
        #mantissa, exponent = f"{val:.1e}".split("e")
        #exponent = int(exponent)

        cbar = plt.colorbar(sc)
        ##cbar.set_label('\n' + rf'$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{exponent}\ bar}}$', fontsize=23)
        #cbar.set_label('\n' + rf'$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ }}$',fontsize=23)
        cbar.set_label('\n' + property_label[i], fontsize=20)
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

        ##legend_patch = mpatches.Patch(color=colors(i), label=rf"$\mathbf{{Ar\ Uptake\ at\ Pressure\ {Icname[i]} bar}}$")

        #legend_patch = mpatches.Patch(color=colors(i), label=rf"$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ bar}}$")
        legend_patch = mpatches.Patch(color=colors(i), label=property_label[i])
        
        plt.legend(handles=[legend_patch], fontsize=14, loc='best', frameon=True)
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
            df     ==> DataFrame of shape (nmat, nprop)

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
    plt.xlabel("Property index", fontdict=fontdict_x)
    plt.ylabel("Property index", fontdict=fontdict_y)
    
    plt.tight_layout()
    
    plt.savefig(f"2_Corr/PlotHeatmap/{Iirun}/{Token}.pdf", format='pdf')
    
    plt.close()

def PearsonPlot(Idf, iprop):

    """
    Description:

            Aim is to visualize the Pearson correlation amongust different columns of material-pressure matrix.   
            
            PlotHeatmap() gives us the values of corelation where as PearsonPlot() helps in visualization of these correlations.

    Input:

            Idf     ==>  Amp (material-pressure matrix) in dataframe format with a shape of (nmat, nprop) 

            iprop   ==> property labels in the form of list with a size of (1, nprop)

    Output:

            Pearson correlation plots in the folder 2_Corr/PearsonPlot

    """

    os.makedirs('2_Corr/PearsonPlot', exist_ok=True)

    [inmat, inprop] = Idf.shape

    for i in range(inprop): # range(inprop)
        for j in range(i, inprop):

            plt.figure(figsize=(11,7))

            X = Idf.iloc[:, i]
            Y = Idf.iloc[:, j]

            iCOR = X.corr(Y)

            #plt.scatter(X, Y, color='green', label=f"{iprop[i]} vs {iprop[j]}", alpha=0.6, edgecolor='black', s=80)
            plt.scatter(X, Y, color='blue', label=f"{iprop[i]} vs {iprop[j]}", alpha=0.6, s=20)
            
            plt.plot([min(min(X), min(Y)), max(max(X), max(Y))],[min(min(X), min(Y)), max(max(X), max(Y))],color='gray', linestyle='--', label='Perfect Prediction')

            plt.xticks(fontsize=26,fontweight='bold')
            plt.yticks(fontsize=26,fontweight='bold')

            #plt.xlabel(f"Uptake of Ar at {iprop[i]}", fontdict=fontdict_x)
            #plt.ylabel(f"Uptake of Ar at {iprop[j]}", fontdict=fontdict_y)
            
            plt.xlabel(f"{iprop[i]}", fontdict=fontdict_x)
            plt.ylabel(f"{iprop[j]}", fontdict=fontdict_y)
            
            plt.title(f"Pearson Correlation : {iCOR:.4f}", fontdict=fontdict_t)
            #plt.legend(loc='upper right', fontsize=20)
            plt.tight_layout()
            #plt.savefig(f"2_Corr/PearsonPlot/{i+1}_{iprop[j]}_vs_{iprop[i]}.pdf", format='pdf')
            plt.savefig(f"2_Corr/PearsonPlot/{i+1}_{j+1}_vs_{i+1}.pdf", format='pdf')
            plt.close()

def PlotIsotherm(Idf, Imof, Iaxis):

    """
    Description:

            Aim is to visualize the adsorption isotherm for each material in the dataset.
            The semilog case clearly shows S-shaped isothem in low pressure regime

    Input: 
            Idf     ==>    DataFrame of shape (nmat, nprop).   

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

    #[inmat, inprop] = Idf.shape

    adsorption_columns = [column for column in Idf.columns if "_uptake_" in column]

    Idf = Idf[adsorption_columns]

    [inmat, inprop] = Idf.shape

    for i in range(inmat): # range(inmat)
        
        plt.figure(figsize=(11,7))

        if Iaxis=='SEMILOG':
            plt.xscale('log')
        else:
            plt.xscale('linear')

        ##X = np.array([1,2,3,4,5,6,7,8,9,10,11])

        #X = np.array([0.00000001,0.00000010,0.00000100,0.00001000,0.00010000,0.00100000,
        #    0.01000000,0.02000000,0.04000000,0.06000000,0.08000000,0.10000000,
        #    0.12000000,0.14000000,0.16000000,0.18000000,0.20000000,0.24000000,
        #    0.28000000,0.30000000,0.32000000,0.36000000,0.40000000,0.50000000,
        #    0.60000000,0.70000000,0.80000000,0.90000000,1.00000000])

        X = np.array([float(column.split("_")[3].replace("bar", "")) for column in adsorption_columns])

        X_label = [f"{x:.0e}" for x in X]

        Adsorbate = adsorption_columns[0].split("_")[0]
        Temperature = int(adsorption_columns[0].split("_")[4].replace("K", ""))
        
        plt.scatter(X, Idf.iloc[i, :], s=100, marker='*', facecolors='none', edgecolors='blue', linewidth=1.5, alpha=1.0)
        
        plt.xlabel('Pressure (bar)', fontdict=fontdict_x)
        #plt.ylabel('Argon uptake (mol/kg)', fontdict=fontdict_y)
        plt.ylabel(f'{Adsorbate} uptake (mol/kg)', fontdict=fontdict_y)

        #plt.title(f'Adsorption isotherm at 87 K for {Imof[i]}', fontdict=fontdict_t)
        plt.title(f'Adsorption isotherm at {Temperature} K for {Imof[i]}', fontdict=fontdict_t)
        
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
    #std_mask    = ~np.isclose(std_after, 1.0, atol=1e-7)

    # Check STD only for rows/columns with at least two known entries.
    # With only one known entry, STD is mathematically zero and cannot be normalized to unity.
    known_count = np.sum(~np.isnan(IMat), axis=Iaxis)
    std_mask = (known_count >= 2) & ~np.isclose(std_after, 1.0, atol=1e-7)

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
    
    # Not required since it is captured in this std_mask = (known_count >= 2) & ~np.isclose(std_after, 1.0, atol=1e-7)
    #if not np.allclose(std_after, 1, atol=1e-7):
    #    bad_rows = np.where(~np.isclose(std_after, 1, atol=1e-7))[0]
    #    raise AssertionError(f"\033[91mSTD not unity in rows/cols {bad_rows}, values: {std_after[bad_rows]}\033[0m")
   
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
        raise ValueError("\033[91mIMat, Imean, and Istd must have compatible sizes (same shape or scalar values).\033[0m")

    # Reverse the standardization
    RMat = (IMat * Istd) + Imean

    # Preserve NaN positions
    RMat[np.isnan(IMat)] = np.nan

    return RMat

def PropMatMissingOld(Ifrac, IMat, Inmat, Inprop, Isample, Ifrac_col=None, Icol=None, Ifrac_row=None, Irow=None):

    """

    Description:

            Generate a Pressure-material matrix with missing entries for model training.

    Input:
            
            Ifrac       ==> Fraction of entries in the Pressure-material matrix that should be known (e.g., 0.4 means 40% known, 60% missing)
            
            IMat        ==> Fully observed Pressure-material matrix (unstandardized).Must be unstandardized so that missing entries can be removed safely; standardization is applied later using only known entries.
            
            Inmat       ==> The no of materials. Equal to the columns of Pressure-material matrix
            
            Inprop     ==> The no of pressure points on adsorption isotherm. Equal to rows of Pressure-material matrix
            
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

    assert IMat.shape == (Inprop, Inmat), \
            f"\033[91mWrong Pressure-material matrix is supplied to PropMatMissing. Expected shape was ({Inprop, Inmat}) got {IMat.shape}\033[0m"

    elem_known      = int(np.floor(Ifrac * Inprop * Inmat))
    elem_unknown    = int(np.floor((1 - Ifrac) * Inprop * Inmat))

    assert abs((elem_known + elem_unknown) - (Inprop * Inmat)) <= 1, \
            f"\033[91mMismatch in known and unknown entries of Pressure-material matrix. Expected ({Inprop * Inmat}) got ({elem_known + elem_unknown})\033[0m"

    assert elem_known >= 0 and elem_unknown >= 0, "\033[91mKnown or unknown entries cannot be negative\033[0m"

    all_idx = list(product(range(Inprop), range(Inmat)))

    if Isample == "RANDOM":

        idx_known   = random.sample(all_idx, elem_known) # Randomly select ids of ${elem_known} number of entries of Pressure-material matrix

    elif Isample == "COL":

        assert Ifrac_col is not None and Icol is not None, \
            "\033[91mIfrac_col and Icol must be provided for column sampling\033[0m"

        assert 0 < Ifrac_col <= 1, "\033[91mIfrac_col must be between 0 and 1\033[0m"

        # Known entries in the target column
        elem_Icol_known = int(np.floor(Ifrac_col * Inprop))
        Icol_rows = list(range(Inprop))
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

    elif Isample == "ROW":

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

    '''

    This part of the code was slower since it was of the order O(N^2) becasue 
    iidx not in idx_known takes O(N) time because Python must scan the entire list, 
    and I was doing this for every element of the matrix, hence the complexity becomes O(N^2)

    # Unknown entries
    idx_unknown = [] # ids of remaining or missing entries of Pressure-material matrix

    for iidx in all_idx: # looping through all the ids of Pressure-material matrix
        if iidx not in idx_known: # Finding ids that are not in known category
            idx_unknown.append(iidx) # ids that are not in idx_known should be in idx_unknown
    assert len(idx_known) + len(idx_unknown) == len(all_idx),\
            f"\033[91mMismatch in known and unknown indices. Expected {len(all_idx)} got {len(idx_known) + len(idx_unknown)}\033[0m"

    '''

    # Here I am using Sets since it is of the order O(N)

    idx_known_set = set(idx_known)
    all_idx_set   = set(all_idx)

    idx_unknown = list(all_idx_set - idx_known_set)

    assert len(idx_known_set) + len(idx_unknown) == len(all_idx_set), \
        f"\033[91mMismatch in known and unknown indices. Expected {len(all_idx_set)} got {len(idx_known_set) + len(idx_unknown)}\033[0m"

    # No overlap between known and unknown entries i.e. known intersection unknown = null set
    assert idx_known_set.isdisjoint(idx_unknown),"\033[91mKnown and unknown indices overlap.\033[0m"

    # Together known and unknown set cover entire matrix i.e. known union unknown = full matrix
    assert idx_known_set.union(idx_unknown) == all_idx_set, "\033[91mKnown and unknown indices do not reconstruct the full matrix.\033[0m"

    # Initialize RMat with NaNs
    # All entries are missing initially
    RMat = np.full((Inprop, Inmat), np.nan)

    # Fill in the known entries with their true (unstandardized) value
    print(f"\tFilling the known entries with their true (unstandardized) value")

    '''

    # This was slower

    for ipress, imat in idx_known:
        RMat[ipress, imat] = IMat[ipress, imat]
        assert np.isclose(RMat[ipress, imat], IMat[ipress, imat]),\
                f"\033[91mMismatch in value at index ({ipress}, {imat}): RMat={RMat[ipress, imat]}, IMat={IMat[ipress, imat]}\033[0m"

    '''

    # This is relatively faster
    rows, cols = zip(*idx_known)
    RMat[rows, cols] = IMat[rows, cols]

    assert np.allclose(RMat[rows, cols], IMat[rows, cols]), f"\033[91mMismatch found while filling known entries in RMat.\033[0m"

    # Cross-check to ensure Ifrac is not violated
    obs_Ifrac = np.count_nonzero(~np.isnan(RMat)) / RMat.size

    assert np.isclose(Ifrac, obs_Ifrac, atol = 1e-5),\
            f"\033[91mMismatch in Overall known fraction: expected {Ifrac}, got {obs_Ifrac}\033[0m"

    # Cross check to ensure Ifrac_col is not violated
    if Isample == "COL":
        Icol_known = np.count_nonzero(~np.isnan(RMat[:, Icol]))
        Icol_frac = Icol_known / Inprop
        
        assert np.isclose(Icol_frac, Ifrac_col, atol=0.05),\
                f"\033[91mMismatch in column fraction. Expected {Ifrac_col} got {Icol_frac:.3f}\033[0m"
    
    # Cross check to ensure all the entries in known indices are filled in RMat
    for ipress, imat in idx_known:
        assert not np.isnan(RMat[ipress, imat]), f"\033[91mMismatch in entries of known indices: Known entry ({ipress},{imat}) is NaN\033[0m"

    # Cross check to ensure all the entries in unknown indices are NaN
    for ipress, imat in idx_unknown:
        assert np.isnan(RMat[ipress, imat]), f"\033[91mMismatch in entries of unknown indices ({ipress},{imat}) is not NaN\033[0m"

    # Cross check to ensure no row or column is completely empty
    empty_rows = np.where(np.all(np.isnan(RMat), axis=1))[0]
    empty_cols = np.where(np.all(np.isnan(RMat), axis=0))[0]

    assert len(empty_rows) == 0, f"\033[91mEmpty rows at indices: {empty_rows}\033[0m"
    assert len(empty_cols) == 0, f"\033[91mEmpty columns at indices: {empty_cols}\033[0m"

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"\tTime taken to run PropMatMissing is : {elapsed_time:.4f} minutes")

    return RMat, idx_known, idx_unknown

def PropMatMissing(Ifrac, IMat, Inmat, Inprop, Isample, Ifrac_col=None, Icol=None, Ifrac_row=None, Irow=None, Irow_sub=None, Icol_sub=None):

    """

    Description:

            Generate a Pressure-material matrix with missing entries for model training.

    Input:
            
            Ifrac       ==> Fraction of entries in the Pressure-material matrix that should be known (e.g., 0.4 means 40% known, 60% missing)
            
            IMat        ==> Fully observed Pressure-material matrix (unstandardized).Must be unstandardized so that missing entries can be removed safely; standardization is applied later using only known entries.
            
            Inmat       ==> The no of materials. Equal to the columns of Pressure-material matrix
            
            Inprop     ==> The no of pressure points on adsorption isotherm. Equal to rows of Pressure-material matrix
            
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

    assert IMat.shape == (Inprop, Inmat), \
            f"\033[91mWrong Pressure-material matrix is supplied to PropMatMissing. Expected shape was ({Inprop, Inmat}) got {IMat.shape}\033[0m"

    all_idx = list(product(range(Inprop), range(Inmat)))

    if Isample not in ["SUBROW", "SUBCOL"]:

        elem_known      = int(np.floor(Ifrac * Inprop * Inmat))
        elem_unknown    = int(np.floor((1 - Ifrac) * Inprop * Inmat))

        assert abs((elem_known + elem_unknown) - (Inprop * Inmat)) <= 1, \
                f"\033[91mMismatch in known and unknown entries of Pressure-material matrix. Expected ({Inprop * Inmat}) got ({elem_known + elem_unknown})\033[0m"

        assert elem_known >= 0 and elem_unknown >= 0, "\033[91mKnown or unknown entries cannot be negative\033[0m"

        # -----------------------------------------------------------------------------
        # Makng sure that there is at least one known entry per row and column
        # -----------------------------------------------------------------------------

        min_entries = max(Inprop, Inmat)

        if elem_known < min_entries:
            raise ValueError(
                f"\033[91mIfrac too small. Minimum required fraction is "
                f"{min_entries/(Inprop*Inmat):.4f} to ensure at least one entry per row and column.\033[0m"
            )

        min_idx_known = set() # using set because it is faster than list

        # Ensuring that  each row of the resulting missing matrix has at least one entry
        for r in range(Inprop):
            c = random.randint(0, Inmat - 1)
            min_idx_known.add((r, c))

        # Ensuring that each column of the resulting missing matrix has at least one entry
        for c in range(Inmat):
            r = random.randint(0, Inprop - 1)
            min_idx_known.add((r, c))

        # Convert to list. I used sets because it is faster.
        min_idx_known = list(min_idx_known)

        # Remaining entries to sample
        min_elem_known = len(min_idx_known)
        remaining_elem_known = elem_known - min_elem_known

        if remaining_elem_known < 0:
            raise ValueError("\033[91mIfrac too small after enforcing row/column coverage.\033[0m")    

    if Isample == "RANDOM":
        
        remaining_pool = [idx for idx in all_idx if idx not in min_idx_known]
        
        assert remaining_elem_known <= len(remaining_pool), \
            "\033[91mNot enough entries in remaining_pool for sampling.\033[0m"

        rest_idx_known = random.sample(remaining_pool, remaining_elem_known)

        assert remaining_elem_known <= len(remaining_pool), \
            "\033[91mNot enough entries in remaining_pool for sampling.\033[0m"

        idx_known = min_idx_known + rest_idx_known

    elif Isample == "COL":

        assert Ifrac_col is not None and Icol is not None, \
            "\033[91mIfrac_col and Icol must be provided for column sampling\033[0m"

        assert 0 < Ifrac_col <= 1, "\033[91mIfrac_col must be between 0 and 1\033[0m"
        
        # Already covered entries in this column
        covered_in_col = [idx for idx in min_idx_known if idx[1] == Icol]
        no_covered_in_col = len(covered_in_col)

        elem_Icol_known = int(np.floor(Ifrac_col * Inprop))

        remaining_Icol_needed = elem_Icol_known - no_covered_in_col

        if remaining_Icol_needed < 0:
            remaining_Icol_needed = 0

        Icol_rows = list(range(Inprop))

        # Avoid already covered rows
        available_rows = [r for r in Icol_rows if (r, Icol) not in min_idx_known]

        assert remaining_Icol_needed <= len(available_rows), \
            "\033[91mNot enough rows available for column sampling.\033[0m"

        known_Icol_rows = random.sample(available_rows, remaining_Icol_needed)

        idx_known_Icol = [(r, Icol) for r in known_Icol_rows]

        remaining_known = remaining_elem_known - len(idx_known_Icol)
        
        assert set(idx_known_Icol).isdisjoint(min_idx_known), \
            "\033[91mOverlap between min_idx_known and idx_known_Icol.\033[0m"

        if remaining_known < 0:
            raise ValueError("\033[91mIfrac_column is too large for the given Ifrac\033[0m")

        # Exclude target column from remaining pool
        remaining_idx_pool = [
            idx for idx in all_idx 
            if idx[1] != Icol and idx not in min_idx_known and idx not in idx_known_Icol
        ]        
        
        assert remaining_known <= len(remaining_idx_pool), \
            "\033[91mNot enough entries in remaining_idx_pool (Column case).\033[0m"        

        rest_idx_known = random.sample(remaining_idx_pool, remaining_known)

        idx_known = min_idx_known + idx_known_Icol + rest_idx_known

    elif Isample == "ROW":

        assert Ifrac_row is not None and Irow is not None, \
            "\033[91mIfrac_row and Irow must be provided for row sampling\033[0m"
        assert 0 < Ifrac_row <= 1, "\033[91mIfrac_row must be between 0 and 1\033[0m"

        # Number of known entries in the target row
 
        covered_in_row = [idx for idx in min_idx_known if idx[0] == Irow]
        n_covered_in_row = len(covered_in_row)

        elem_row_known = int(np.floor(Ifrac_row * Inmat))

        remaining_row_needed = elem_row_known - n_covered_in_row

        if remaining_row_needed < 0:
            remaining_row_needed = 0        

        # Select known entries in the target row

        row_cols = list(range(Inmat))
        
        available_cols = [c for c in row_cols if (Irow, c) not in min_idx_known]

        assert remaining_row_needed <= len(available_cols), \
            "\033[91mNot enough columns available for row sampling.\033[0m"

        known_row_cols = random.sample(available_cols, remaining_row_needed)

        idx_known_row = [(Irow, c) for c in known_row_cols]

        # Remaining known entries to satisfy overall Ifrac
        remaining_known = remaining_elem_known - len(idx_known_row)
        
        if remaining_known < 0:
            raise ValueError("\033[91mIfrac_row is too large for the given Ifrac\033[0m")

        # Exclude target row from remaining pool
        remaining_idx_pool = [
            idx for idx in all_idx 
            if idx[0] != Irow and idx not in min_idx_known and idx not in idx_known_row
        ]

        assert remaining_known <= len(remaining_idx_pool), \
            "\033[91mNot enough entries in remaining_idx_pool (Row case).\033[0m"        

        rest_idx_known = random.sample(remaining_idx_pool, remaining_known)

        # Combine known entries
        idx_known = min_idx_known + idx_known_row + rest_idx_known

    elif Isample == "SUBROW":

        assert Irow_sub is not None, \
                "\033[91mIrow_sub must be provided for SUBROW sampling\033[0m"

        assert len(Irow_sub) > 0, \
                "\033[91mIrow_sub cannot be empty for SUBROW sampling\033[0m"

        assert all(0 <= r < Inprop for r in Irow_sub), \
                f"\033[91mSUBROW row indices exceed pressure dimension ({Inprop})\033[0m"

        sub_size = len(Irow_sub) * Inmat

        elem_known_sub   = int(np.floor(Ifrac * sub_size))
        elem_unknown_sub = int(np.floor((1 - Ifrac) * sub_size))

        assert abs((elem_known_sub + elem_unknown_sub) - sub_size) <= 1, \
                f"\033[91mMismatch in known and unknown entries of Pressure-material sub-matrix. Expected ({sub_size}) got ({elem_known_sub + elem_unknown_sub})\033[0m"

        assert elem_known_sub >= 0 and elem_unknown_sub >= 0, \
                "\033[91mKnown or unknown entries cannot be negative\033[0m"

        sub_idx = list(product(Irow_sub, range(Inmat)))

        assert len(sub_idx) == sub_size, \
                f"\033[91mSubmatrix size mismatch: expected {sub_size}, got {len(sub_idx)}\033[0m"

        idx_known_sub = random.sample(sub_idx, elem_known_sub)

        idx_known = list(idx_known_sub)

        idx_known_set = set(idx_known) # Converting known indices to sets for fatser calculation
        empty_rows = [r for r in range(Inprop) if not any(i == r for i, j in idx_known_set)] # Finding completely empty rows
        # For every empty row add one random entry
        for r in empty_rows:
            c = random.randint(0, Inmat - 1)
            idx_known_set.add((r, c))
        empty_cols = [c for c in range(Inmat) if not any(j == c for i, j in idx_known_set)] # Finding completely empty column
        # For every empty column add one random entry
        for c in empty_cols:
            r = random.randint(0, Inprop - 1)
            idx_known_set.add((r, c))

        idx_known = list(idx_known_set)

    elif Isample == "SUBCOL":

        assert Icol_sub is not None, \
                "\033[91mIcol_sub must be provided for SUBCOL sampling\033[0m"

        assert len(Icol_sub) > 0, \
                "\033[91mIcol_sub cannot be empty for SUBCOL sampling\033[0m"

        assert all(0 <= c < Inmat for c in Icol_sub), \
                f"\033[91mSUBCOL column indices exceed material dimension ({Inmat})\033[0m"

        sub_size = Inprop * len(Icol_sub)

        elem_known_sub   = int(np.floor(Ifrac * sub_size))
        elem_unknown_sub = int(np.floor((1 - Ifrac) * sub_size))

        assert abs((elem_known_sub + elem_unknown_sub) - sub_size) <= 1, \
                f"\033[91mMismatch in known and unknown entries of Pressure-material sub-matrix. Expected ({sub_size}) got ({elem_known_sub + elem_unknown_sub})\033[0m"

        assert elem_known_sub >= 0 and elem_unknown_sub >= 0, \
                "\033[91mKnown or unknown entries cannot be negative\033[0m"

        sub_idx = list(product(range(Inprop), Icol_sub))

        assert len(sub_idx) == sub_size, \
                f"\033[91mSubmatrix size mismatch: expected {sub_size}, got {len(sub_idx)}\033[0m"

        idx_known_sub = random.sample(sub_idx, elem_known_sub)

        # Ensuring that every row and column has at least one known entry
        idx_known_set = set(idx_known_sub)

        empty_rows = [r for r in range(Inprop)
                      if not any(i == r for i, j in idx_known_set)]

        # For every empty row add one random entry
        for r in empty_rows:
            c = random.randint(0, Inmat - 1)
            idx_known_set.add((r, c))

        empty_cols = [c for c in range(Inmat)
                      if not any(j == c for i, j in idx_known_set)]

        # For every empty column add one random entry
        for c in empty_cols:
            r = random.randint(0, Inprop - 1)
            idx_known_set.add((r, c))

        idx_known = list(idx_known_set)

        obs_frac = len(idx_known) / (Inprop * Inmat)

    else:
        raise ValueError(f"\033[91mUnknown sampling method: {Isample}\033[0m")

    '''

    This part of the code was slower since it was of the order O(N^2) becasue 
    iidx not in idx_known takes O(N) time because Python must scan the entire list, 
    and I was doing this for every element of the matrix, hence the complexity becomes O(N^2)

    # Unknown entries
    idx_unknown = [] # ids of remaining or missing entries of Pressure-material matrix

    for iidx in all_idx: # looping through all the ids of Pressure-material matrix
        if iidx not in idx_known: # Finding ids that are not in known category
            idx_unknown.append(iidx) # ids that are not in idx_known should be in idx_unknown
    assert len(idx_known) + len(idx_unknown) == len(all_idx),\
            f"\033[91mMismatch in known and unknown indices. Expected {len(all_idx)} got {len(idx_known) + len(idx_unknown)}\033[0m"

    '''

    assert len(idx_known) == len(set(idx_known)), \
        "\033[91mDuplicate entries detected in idx_known.\033[0m" 
    
    if Isample not in ["SUBROW", "SUBCOL"]:
        assert len(idx_known) == elem_known, \
            f"\033[91mMismatch in known entries. Expected {elem_known}, got {len(idx_known)}\033[0m" 

    if Isample in ["SUBROW", "SUBCOL"]:
        assert len(idx_known_sub) == elem_known_sub, \
                f"\033[91mMismatch in SUB known entries. Expected {elem_known_sub}, got {len(idx_known_sub)}\033[0m"
        
    rows_final = {r for r, _ in idx_known}
    cols_final = {c for _, c in idx_known}

    assert len(rows_final) == Inprop, \
        "\033[91mRow coverage lost after sampling.\033[0m"

    assert len(cols_final) == Inmat, \
        "\033[91mColumn coverage lost after sampling.\033[0m"  

    # Here I am using Sets since it is of the order O(N)

    idx_known_set = set(idx_known)
    all_idx_set   = set(all_idx)

    idx_unknown = list(all_idx_set - idx_known_set)

    assert len(idx_known_set) + len(idx_unknown) == len(all_idx_set), \
        f"\033[91mMismatch in known and unknown indices. Expected {len(all_idx_set)} got {len(idx_known_set) + len(idx_unknown)}\033[0m"

    # No overlap between known and unknown entries i.e. known intersection unknown = null set
    assert idx_known_set.isdisjoint(idx_unknown),"\033[91mKnown and unknown indices overlap.\033[0m"

    # Together known and unknown set cover entire matrix i.e. known union unknown = full matrix
    assert idx_known_set.union(idx_unknown) == all_idx_set, "\033[91mKnown and unknown indices do not reconstruct the full matrix.\033[0m"

    # Initialize RMat with NaNs
    # All entries are missing initially
    RMat = np.full((Inprop, Inmat), np.nan)

    # Fill in the known entries with their true (unstandardized) value
    print(f"\tFilling the known entries with their true (unstandardized) value")

    '''

    # This was slower

    for ipress, imat in idx_known:
        RMat[ipress, imat] = IMat[ipress, imat]
        assert np.isclose(RMat[ipress, imat], IMat[ipress, imat]),\
                f"\033[91mMismatch in value at index ({ipress}, {imat}): RMat={RMat[ipress, imat]}, IMat={IMat[ipress, imat]}\033[0m"

    '''

    # This is relatively faster
    rows, cols = zip(*idx_known)
    RMat[rows, cols] = IMat[rows, cols]

    assert np.allclose(RMat[rows, cols], IMat[rows, cols]), f"\033[91mMismatch found while filling known entries in RMat.\033[0m"

    # Cross-check to ensure Ifrac is not violated
    obs_Ifrac = np.count_nonzero(~np.isnan(RMat)) / RMat.size

    if Isample not in ["SUBROW", "SUBCOL"]:

        assert np.isclose(Ifrac, obs_Ifrac, atol = 1e-5),\
                f"\033[91mMismatch in Overall known fraction: expected {Ifrac}, got {obs_Ifrac}\033[0m"

    elif Isample in ["SUBROW", "SUBCOL"]:

        #obs_Ifrac_sub = len([idx for idx in idx_known if idx in sub_idx]) / sub_size
        obs_Ifrac_sub = len(idx_known_sub) / sub_size

        expected_Ifrac_sub = elem_known_sub / sub_size

        assert np.isclose(expected_Ifrac_sub, obs_Ifrac_sub, atol=1e-12), \
                f"\033[91mMismatch in {Isample} known fraction. Expected {expected_Ifrac_sub}, got {obs_Ifrac_sub}\033[0m"

        #assert np.isclose(Ifrac, obs_Ifrac_sub, atol = 1e-4), \
        #        f"\033[91mMismatch in SUB known fraction: expected {Ifrac}, got {obs_Ifrac_sub}\033[0m"

        print(f"For SUB matrix Ifrac of {Ifrac}, the overall Ifrac is {obs_Ifrac}")

    # Cross check to ensure Ifrac_col is not violated
    if Isample == "COL":
        Icol_known = np.count_nonzero(~np.isnan(RMat[:, Icol]))
        Icol_frac = Icol_known / Inprop
        
        assert np.isclose(Icol_frac, Ifrac_col, atol=0.05),\
                f"\033[91mMismatch in column fraction. Expected {Ifrac_col} got {Icol_frac:.3f}\033[0m"
    
    # Cross check to ensure all the entries in known indices are filled in RMat
    for ipress, imat in idx_known:
        assert not np.isnan(RMat[ipress, imat]), f"\033[91mMismatch in entries of known indices: Known entry ({ipress},{imat}) is NaN\033[0m"

    # Cross check to ensure all the entries in unknown indices are NaN
    for ipress, imat in idx_unknown:
        assert np.isnan(RMat[ipress, imat]), f"\033[91mMismatch in entries of unknown indices ({ipress},{imat}) is not NaN\033[0m"

    # Cross check to ensure no row or column is completely empty
    empty_rows = np.where(np.all(np.isnan(RMat), axis=1))[0]
    empty_cols = np.where(np.all(np.isnan(RMat), axis=0))[0]

    assert len(empty_rows) == 0, f"\033[91mEmpty rows at indices: {empty_rows}\033[0m"
    assert len(empty_cols) == 0, f"\033[91mEmpty columns at indices: {empty_cols}\033[0m"

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"\tTime taken to run PropMatMissing is : {elapsed_time:.4f} minutes")

    return RMat, idx_known, idx_unknown

def PropMatExisting(IMat, Inmat, Inprop):

    """
    Description:

        Process an already incomplete Pressure-material matrix.

        Unlike PropMatMissing(), this function does not generate
        missing entries. The NaN pattern already present in IMat
        is retained exactly as supplied.

    Input:

        IMat   ==> Already incomplete Pressure-material matrix.
                   Known entries contain their values and missing
                   entries are represented by NaN.

        Inmat  ==> Number of materials (columns).

        Inprop ==> Number of properties / pressure points (rows).

    Output:

        RMat       ==> Original incomplete Pressure-material matrix.
        idx_known  ==> Indices of the known entries.
        idx_unknown ==> Indices of the missing entries.
        Ifrac      ==> Actual fraction of known entries in IMat.

    """

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Basic checks
    # -------------------------------------------------------------------------

    assert IMat.shape == (Inprop, Inmat), \
        f"\033[91mWrong Pressure-material matrix supplied to PropMatExisting. Expected shape was ({Inprop}, {Inmat}), got {IMat.shape}\033[0m"

    # -------------------------------------------------------------------------
    # Keep the original missing-data pattern exactly as supplied
    # -------------------------------------------------------------------------

    RMat = IMat.copy()

    # -------------------------------------------------------------------------
    # Identify known and unknown entries
    # -------------------------------------------------------------------------

    idx_known = list(zip(*np.where(~np.isnan(RMat))))
    idx_unknown = list(zip(*np.where(np.isnan(RMat))))

    # -------------------------------------------------------------------------
    # Calculate actual known fraction
    # -------------------------------------------------------------------------

    Ifrac = len(idx_known) / RMat.size

    print(f"\nOriginal matrix known fraction   = {Ifrac:.6f}")
    print(f"Original matrix unknown fraction = {1.0 - Ifrac:.6f}")

    # -------------------------------------------------------------------------
    # Check that every property/pressure row has at least one known entry
    # -------------------------------------------------------------------------

    empty_rows = np.where(np.all(np.isnan(RMat), axis=1))[0]

    assert len(empty_rows) == 0, \
        f"\033[91mORIGINAL matrix contains completely empty row(s): " \
        f"{empty_rows.tolist()}\033[0m"

    # -------------------------------------------------------------------------
    # Check that every material column has at least one known entry
    # -------------------------------------------------------------------------

    empty_cols = np.where(np.all(np.isnan(RMat), axis=0))[0]

    assert len(empty_cols) == 0, \
        f"\033[91mORIGINAL matrix contains completely empty column(s): " \
        f"{empty_cols.tolist()}\033[0m"

    # -------------------------------------------------------------------------
    # Check known + unknown indices
    # -------------------------------------------------------------------------

    all_idx = set(product(range(Inprop), range(Inmat)))

    idx_known_set = set(idx_known)
    idx_unknown_set = set(idx_unknown)

    assert len(idx_known_set) + len(idx_unknown_set) == len(all_idx), \
        "\033[91mMismatch in known and unknown indices.\033[0m"

    assert idx_known_set.isdisjoint(idx_unknown_set), \
        "\033[91mKnown and unknown indices overlap.\033[0m"

    assert idx_known_set.union(idx_unknown_set) == all_idx, \
        "\033[91mKnown and unknown indices do not reconstruct the full matrix.\033[0m"

    # -------------------------------------------------------------------------
    # Final check: known entries are not NaN and unknown entries are NaN
    # -------------------------------------------------------------------------

    for iprop, imat in idx_known:
        assert not np.isnan(RMat[iprop, imat]), \
            f"\033[91mKnown entry ({iprop},{imat}) is NaN.\033[0m"

    for iprop, imat in idx_unknown:
        assert np.isnan(RMat[iprop, imat]), \
            f"\033[91mUnknown entry ({iprop},{imat}) is not NaN.\033[0m"

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    elapsed_time = (time.time() - start_time) / 60

    print(f"\tTime taken to run PropMatOriginal is : {elapsed_time:.4f} minutes")

    return RMat, idx_known, idx_unknown, Ifrac

def fit_lrm(IMat, Inprop, Inmat, Ihpara, Iidx_known, IP0=None, IM0=None, Ioffval=True):
    
    """

    Description:
    
            Fit a low-rank model (GLRM) to an incomplete Pressure-material matrix using pyglrm.
    
    Input:

            X_Mat (ndarray)      ==> Incomplete matrix of shape (n_p, n_m) with np.nan for missing entries
            
            Ihpara               ==> Contains dim (rank) and reg (regularization strength)
            
            Iidx_known (list)    ==> List of (p, m) tuples indicating observed entries
            
            P0 (ndarray or None) ==> Optional initial guess for P of shape (dim + 1, nprop).
            
            M0 (ndarray or None) ==> Optional initial guess for M of shape (dim + 1, nmat).offval offset value

    Output:
            
            RP (ndarray)         ==> Latent Pressure vectors of shape (dim + 1, nprop)
            
            RM (ndarray)         ==> Latent material vectors of shape (dim + 1, nmat)
            
            Rmodel               ==> Fitted GLRM model
            
            Rch                  ==> Convergence history

    """

    start_time = time.time()

    assert IMat.shape == (Inprop, Inmat), f"\033[91mIMat shape mismatch: expected ({Inprop}, {Inmat}), got {IMat.shape}\033[0m"

    if Ioffval == True:

        # Initialize if not provided
        if IP0 is None:
            IP0 = np.random.randn(Ihpara["dim"]+1, Inprop)
        if IM0 is None:
            IM0 = np.random.randn(Ihpara["dim"]+1, Inmat)

        Rmodel = glrm(
            losses=QuadLoss(),
            rx=QuadReg(Ihpara["reg"] / Inmat),
            ry=QuadReg(Ihpara["reg"] / Inprop),
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
        RM = Rmodel.Y  # (dim + 1, Inprop) Pressure vectors

        #print(RP.shape)
        #print(RM.shape)

        assert RP.shape == (Ihpara['dim'] + 1, Inprop), f"\033[91mExpected shape of RP is {(Ihpara['dim'] + 1, Inprop)}, but got {RP.shape}\033[0m"
        assert RM.shape == (Ihpara['dim'] + 1, Inmat), f"\033[91mExpected RM shape {(Ihpara['dim'] + 1, Inmat)}, but got {RM.shape}\033[0m"

        # Internal consistency check: impute vs manual P.T @ M
        IMat_pred_model = Rmodel.impute()

        # Recreating the matrix IMat by using the decomposed matrix RM and RP. Note these contains bias in them
        IMat_pred_manual = RP.T @ RM # shape (Inprop * Ihpara.dim + 1) @ (Ihpara.dim + 1, Inmat) = (Inprop * Inmat)

        # Note that last row of RP has does not contain any material bias and is just row of 1
        P_act = RP[:-1, :] # because RP.shape = Ihpara.dim + 1 * Inprop but in the X = P.T + M P.shape = Ihpara.dim * Inprop that is we need to exclude the last row of RP

        # Note that the last row of RM contains the material bias
        M_act = RM[:-1, :] # because RM.shape = Ihpara.dim + 1 * Inprop but in the X = P.T + M M.shape = Ihpara.dim * Inmat that is we need to exclude the last row of RM

        mat_bias = RM[-1, :].reshape(-1, 1)  # shape (Inmat, 1) This is material bias
    
        #print(f"\n\t===========================================================")
        #print(f"\n\tLatent-vec-Pressure shape without bias term is : {P_act.shape}")
        #print(f"\tLatent-vec-material shape without bias term is   : {M_act.shape}")
        #print(f"\tMaterial bias vector shape                       :  {mat_bias.shape}")

        assert P_act.shape == (Ihpara['dim'], Inprop), f"\033[91mExpected P_act shape is : {(Ihparam['dim'], Inprop)}, but got {P_act.shape}\033[0m"
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
            IP0 = np.random.randn(Ihpara["dim"], Inprop)
        if IM0 is None:
            IM0 = np.random.randn(Ihpara["dim"], Inmat)

        Rmodel = glrm(
            losses=QuadLoss(),
            rx=QuadReg(Ihpara["reg"] / Inmat),
            ry=QuadReg(Ihpara["reg"] / Inprop),
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
        RM = Rmodel.Y  # (dim + 1, Inprop) Pressure vectors

        #print(RP.shape)
        #print(RM.shape)

        assert RP.shape == (Ihpara['dim'], Inprop), f"\033[91mExpected shape of RP is {(Ihpara['dim'], Inprop)}, but got {RP.shape}\033[0m"
        assert RM.shape == (Ihpara['dim'], Inmat), f"\033[91mExpected RM shape {(Ihpara['dim'], Inmat)}, but got {RM.shape}\033[0m"

        # Internal consistency check: impute vs manual P.T @ M
        IMat_pred_model = Rmodel.impute()

        # Recreating the matrix IMat by using the decomposed matrix RM and RP. Note these contains bias in them
        IMat_pred_manual = RP.T @ RM # shape (Inprop * Ihpara.dim) @ (Ihpara.dim, Inmat) = (Inprop * Inmat)

        ## Note that last row of RP has does not contain any material bias and is just row of 1
        ##P_act = RP[:-1, :] # because RP.shape = Ihpara.dim + 1 * Inprop but in the X = P.T + M P.shape = Ihpara.dim * Inprop that is we need to exclude the last row of RP

        ## Note that the last row of RM contains the material bias
        ##M_act = RM[:-1, :] # because RM.shape = Ihpara.dim + 1 * Inprop but in the X = P.T + M M.shape = Ihpara.dim * Inmat that is we need to exclude the last row of RM

        ##mat_bias = RM[-1, :].reshape(-1, 1)  # shape (Inmat, 1) This is material bias
    
        ##print(f"\n\t===========================================================")
        ##print(f"\n\tLatent-vec-Pressure shape without bias term is : {P_act.shape}")
        ##print(f"\tLatent-vec-material shape without bias term is   : {M_act.shape}")
        ##print(f"\tMaterial bias vector shape                       :  {mat_bias.shape}")

        ##assert P_act.shape == (Ihpara['dim'], Inprop), f"\033[91mExpected P_act shape is : {(Ihparam['dim'], Inprop)}, but got {P_act.shape}\033[0m"
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

def GLRMTest(Ioffval, IAmp_T, Inmat, Inprop, Isample):

    """

    Description: 

    GLRMTest(offset, Amp_T, nmat, nprop, sample)

            Aim is to run several test cases on the fit_lrm() function to check whether function works or not.

    Input:

            Ioffval     ==> Offset value to be passed to the glrm model. If 'True' then offsets are added
            
            My question why to put True what if its false?

            IAmp_T      ==>  Pressure-material matrix Amp_T

            Inmat       ==>   # of materials or # rows of material-pressure matrix

            Inprop     ==> # of pressure or # columns of material-pressure matrix

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
        M0 = np.random.randn(hpara["dim"] + 1, nprop)
    elif Ioffval == False:
        P0 = np.random.randn(hpara["dim"], nmat)
        M0 = np.random.randn(hpara["dim"], nprop)

    # Step 3: Creating Missing Pressure-material matrix with frac = 0.7
    if Isample == "SUBROW":
        [Amp_T_incomplete_frac, idx_known_frac, idx_unknown_frac] = PropMatMissing(0.7, IAmp_T, Inmat, Inprop, Isample, None, None, None, None, Irow_sub, None)
    elif Isample == "SUBCOL":
        [Amp_T_incomplete_frac, idx_known_frac, idx_unknown_frac] = PropMatMissing(0.7, IAmp_T, Inmat, Inprop, Isample, None, None, None, None, None, Icol_sub)
    elif args.existing:
        [Amp_T_incomplete_frac, idx_known_frac, idx_unknown_frac, Ifrac] = PropMatExisting(IAmp_T, Inmat, Inprop)
    else:
        [Amp_T_incomplete_frac, idx_known_frac, idx_unknown_frac] = PropMatMissing(0.7, IAmp_T, Inmat, Inprop, Isample, None, None, None, None, None, None)

    # Step 4: Standardizing the incomplete Pressure-material matrix
    Amp_T_incomplete_frac_mean, Amp_T_incomplete_frac_std, Amp_T_incomplete_frac_stan = standardize_matrix(Amp_T_incomplete_frac, 1)

    # Step 5: Fiting the model for frac = 0.7
    P, M, model, ch = fit_lrm(Amp_T_incomplete_frac_stan, Inprop, Inmat, hpara, idx_known_frac, P0, M0, Ioffval)

    # Step 5 Fiting the model for frac = 0.4
    # Note here the inputs to fit_lrm are taken from [DEBUG 4: RANDOM] case to save time
    P, M, model, ch = fit_lrm(Amp_T_incomplete_stan, nprop, nmat, hpara, idx_known, P0, M0, Ioffval)

    print("\n\t================================================================================")
    print(f"\t\033[92m GLRM model learned latent and bias vector for Ioffval = {Ioffval}, frac = 0.7 and frac = 0.4 and successfully passed all the tests !!!\033[0m")
    print("\t================================================================================\n")

    # Step 5: Bias-only model for frac = 0.9
    hpara_bias = {"dim": 0, "reg": 0.00}

    P0_bias = np.random.randn(hpara["dim"] + 1, nmat)
    M0_bias = np.random.randn(hpara["dim"] + 1, nprop)

    P_b, M_b, model_b, ch_b = fit_lrm(Amp_T_incomplete_frac_stan, Inprop, Inmat, hpara_bias, idx_known_frac, P0, M0, True)

    X_full_bias_only = P_b.T @ M_b

    assert M_b.shape == (1, nmat), "\033[91mBias-only M shape incorrect\033[0m"
    assert P_b.shape == (1, nprop), "\033[91mBias-only P shape incorrect\033[0m"
    assert np.allclose(model_b.impute(), X_full_bias_only, atol=1e-4), \
        "\033[91mBias-only imputation mismatch\033[0m"

    # Step 7: Bias-only model for frac = 0.4
    # Note here the inputs to fit_lrm are taken from [DEBUG 4: RANDOM] case to save time

    P_b, M_b, model_b, ch_b = fit_lrm(Amp_T_incomplete_stan, nprop, nmat, hpara_bias, idx_known, P0, M0, True)
    
    X_full_bias_only = P_b.T @ M_b

    assert M_b.shape == (1, nmat), "\033[91mBias-only M shape incorrect\033[0m"
    assert P_b.shape == (1, nprop), "\033[91mBias-only P shape incorrect\033[0m"
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

def PlotHPError(IRMSD, IR2, IFrobNorm, Ihpg, Ifrac, irun):

    """

    Description:

            The aim is to plot the error in validation set as we search through hyperparameter space

    Input:  

            IRMSD       ==> List of RMSD values on validation set

            IR2         ==> List of R2 values on validation set

            Ihpg        ==> Dictionary of hyper-parameter grid

            Ifrac       ==> Fraction of entries in the Pressure-material matrix that should be known (e.g., 0.4 means 40% known, 60% missing)

            IFrobNorm   ==>

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

    plt.close()

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

    plt.close()

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

    plt.close()

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

    plt.close()

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

    plt.close()

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

    plt.close()


    #--------------------------Frobenius norm plots----------------------------# 

    # Top-10 Frobenius plot

    iFrob = np.array(IFrobNorm)

    iFrob_top = iFrob[np.argsort(iFrob)[:10]]
    X_top_Frob = np.array(X)[np.argsort(iFrob)[:10]]

    sort_idx = np.argsort(X_top_Frob)
    X_top_Frob, iFrob_top = X_top_Frob[sort_idx], iFrob_top[sort_idx]

    hp_labels_Frob = [f"{Ihpg[idx-1]['dim']}\n{Ihpg[idx-1]['reg']:g}" for idx in X_top_Frob]

    plt.figure(figsize=(11, 7))

    plt.plot(XTicks, iFrob_top, marker='*', linestyle='-', color='green',
        markersize=20, linewidth=1.5, alpha=0.6, label='Frobenius')

    plt.xticks(ticks=XTicks, labels=hp_labels_Frob,
        fontdict={'fontsize': 23, 'fontweight': 'bold'})

    plt.yticks(fontsize=23, weight='bold')

    plt.legend(loc='upper right', fontsize=23)

    plt.savefig(f"4_HPTuning/{irun}/Frob_HP_{Ifrac}.pdf", format='pdf')
    
    plt.close()

    # Frobenius vs dimension

    mean_frob = []
    std_frob  = []

    for idim in np.unique(dim_vals):
        vals = iFrob[dim_vals == idim]
        mean_frob.append(np.mean(vals))
        std_frob.append(np.std(vals))

    mean_frob = np.array(mean_frob)
    std_frob  = np.array(std_frob)

    plt.figure(figsize=(11, 7))

    plt.errorbar(np.unique(dim_vals), mean_frob, yerr=std_frob,
                 fmt='o-', linewidth=3, color='royalblue',
                 markersize=12, capsize=5, label='Frobenius')

    plt.xticks(ticks=dim_vals, fontsize=20, weight='bold', rotation=90)
    plt.yticks(fontsize=20, weight='bold')

    plt.legend(loc='best', fontsize=20)

    plt.savefig(f"4_HPTuning/{irun}/Frob_LatDim_{Ifrac}.pdf", format='pdf')
    
    plt.close()

    # Frobenius vs regularization

    mean_frob = []
    std_frob  = []

    for ireg in np.unique(reg_vals):
        vals = iFrob[reg_vals == ireg]
        mean_frob.append(np.mean(vals))
        std_frob.append(np.std(vals))

    mean_frob = np.array(mean_frob)
    std_frob  = np.array(std_frob)

    plt.figure(figsize=(11, 7))

    plt.errorbar(np.arange(1, len(np.unique(reg_vals)) + 1),
                 mean_frob, yerr=std_frob,
                 fmt='o-', linewidth=3, color='royalblue',
                 markersize=12, capsize=5, label='Frobenius')

    plt.xticks(
        ticks=np.arange(1, len(np.unique(reg_vals)) + 1),
        labels=[f"{val:g}" for val in np.unique(reg_vals)],
        rotation=90,
        fontdict={'fontsize': 20, 'fontweight': 'bold'}
    )

    plt.yticks(fontsize=20, weight='bold')

    plt.legend(loc='best', fontsize=20)

    plt.savefig(f"4_HPTuning/{irun}/Frob_LatReg_{Ifrac}.pdf", format='pdf')  

    plt.close()  

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print(f"Time taken to run PlotHPError() is: {elapsed_time:.4f} minutes")

def PlotHPGrid(IRMSD, IR2, IFrobNorm, Ihpg, Ifrac, irun):

    """

    Description:

        Plot validation performance (RMSD and R²) across the hyperparameter grid using Seaborn.

    Inputs:

        IRMSD       ==> list of RMSD values
        
        IR2         ==> list of R² values
        
        Ihpg        ==> list of dictionaries, each with keys 'reg' and 'dim'
        
        Ifrac       ==> fraction of known entries in the matrix
        
        irun        ==> ith run out of nrun for the same Ifrac to ensure statistical accuracy

        IFrobNorm   ==> 

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

    # --- Frobenius norm Heatmap ---

    IFrobNorm = np.array(IFrobNorm)

    Frob_grid = np.full((len(dim_unique), len(reg_unique)), np.nan)

    Frob_grid[j, i] = IFrobNorm[mask][0]

    Frob_df = pd.DataFrame(Frob_grid, index=dim_unique, columns=reg_unique)

    plt.figure(figsize=(11,7))

    ax = sns.heatmap(Frob_df, cmap=cmap_hls, linewidths=1,
                     linecolor='black', xticklabels=True, yticklabels=True)

    #ax = sns.heatmap(Frob_df, cmap=cmap_hls,vmin=np.min(Frob_df.values),vmax=np.max(Frob_df.values))

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20, width=1, color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

    ax.collections[0].colorbar.outline.set(linewidth=2, edgecolor='black')

    plt.xticks(fontsize=18, weight='bold', rotation=90)
    plt.yticks(fontsize=18, weight='bold', rotation=0)

    plt.tight_layout()

    plt.savefig(f"4_HPTuning/{irun}/Frob_HPGrid_{Ifrac}.pdf", format='pdf')

    plt.close()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print(f"Time taken to run PlotHPGrid() is: {elapsed_time:.4f} minutes")

def HPTune(IMat, Inprop, Inmat, Ihpg, Iidx_train, Iidx_test, Ifrac, Ioffval, irun):
    
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
        
        IMat            ==> Incomplete standardized Pressure-material matrix (i.e. Amp_T_incomplete_stan)

        Ihpg            ==> hyperparameter grid
        
        Inprop         ==> size of properties
        
        Inmat           ==> size of materials
        
        Iidx_train      ==> ids corresponding to RMat which contain known entries which would be utilized for model training. (subset of idx_known of Amp_T_incomplete)
        
        Iidx_test       ==> ids corresponding to RMat which contain known entries which would be utilized for model testing. (subset of idx_known of Amp_T_incomple)
        
        Ifrac           ==> Fraction used to computinh missing elements
        
        Ioffval         ==> True 

        irun            ==> ith run out of nrun for the same Ifrac to ensure statistical accuracy

    Output:

        Rhp             ==> value of best  hyperparameter
        
        Rrmsd_TE        ==> value of test  RMSD corresponding to the best hyperparameter
        
        Rrmsd_TR        ==> value of train RMSD corresponding to the best hyperparameter

        Rplot_RMSD      ==> 

        Rplot_R2        ==>

        Rplot_FrobNorm  ==> 

    """

    start_time = time.time()

    Rplot_RMSD      = []
    Rplot_R2        = []
    Rplot_FrobNorm  = []

    # Cross check to ensure all the entries in known indices are filled in IMat
    for ip, im in Iidx_train:
        assert not np.isnan(IMat[ip, im]), f"\033[91mMismatch in entries of known indices: Known entry ({ip},{im}) is NaN\033[0m"

    # Cross check to ensure all the entries in unknown indices are not filled in IMat
    for ip, im in Iidx_test:
        assert not np.isnan(IMat[ip, im]), f"\033[91mMismatch in entries of unknown indices : Known entry ({ip}, {im} is NaN)\033[0m"

    # Asserting on shape of IMat
    assert IMat.shape == (Inprop, Inmat), f"\033[91mShape mismatch in IMat. Expected ({Inprop, Inmat}). Got {IMat.shape}\033[0m"

    # Initializing best hyperparameter
    Rhp          = {"dim": 0, "reg": 0.00}
    
    Rrmsd_TE     = float('inf')
    Rrmsd_TR     = float('inf')
    
    RR2_TE       = float('inf')

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
        P, M, model, ch = fit_lrm(IMat, Inprop, Inmat, ihpg, Iidx_train, None, None, Ioffval)

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
        iRMSD     = np.sqrt(mean_squared_error(IMat_test_true, IMat_pred_test_pred))
        iR2       = r2_score(IMat_test_true, IMat_pred_test_pred) 
        iFrobNorm = np.linalg.norm(IMat_test_true - IMat_pred_test_pred) / np.linalg.norm(IMat_test_true) 

        Rplot_RMSD.append(iRMSD)
        Rplot_R2.append(iR2)
        Rplot_FrobNorm.append(iFrobNorm)

        print(f"\n\tRMSD in validation set is: {iRMSD:.4f}. Should be close to 0.0000\n")
        print(f"\tR2 between actual and predicted values in validation set is: {iR2:.4f}. Should be close to 1.0000\n")
        print(f"\tNormalized Frobenius norm is : {iFrobNorm:.4f}. Should be close to 0.0000\n")

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

    PlotHPError(Rplot_RMSD, Rplot_R2, Rplot_FrobNorm, Ihpg, Ifrac, irun)
    PlotHPGrid(Rplot_RMSD, Rplot_R2, Rplot_FrobNorm, Ihpg, Ifrac, irun)

    #raise ValueError(f"DEBUGGING GOING !!!!")

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

def PlotIsothermParity(IMat, IMat_pred, IMat_l_pred, IMat_b_pred, Inprop, Inmat, Ifrac, irun, IIIidx_unknown_frac, IIIidx_known_frac, full, Iaxis):
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

                IMat            ==>     The normalized or unnormalized completely known pressure-material matrix. (Inprop * Inmat)
                
                IMat_pred       ==>     The normalized or unnormalized latent-bias model predicted pressure-material matrix. (Inprop * Inmat)
                
                IMat_l_pred     ==>     The normalized or unnormalized latent only model predicted pressure-material matrix. (Inprop * Inmat)

                IMat_b_pred     ==>     The normalized or unnormalized bias only model predicted pressure-material matrix. (Inprop * Inmat)
                
                Inprop         ==>     Same as before 
                
                Inmat           ==>     Same as before 
                
                Ifrac           ==>     Same as before

                Iaxis           ==>     if Iaxis = SEMILOG, then X axis of isotherm is in log scale and Y axis in linear scale.
                                        if Iaxis = LINIEAR, then both X and Y axis of the isotherm is in linear scale. 

    Output:

                Parity plots for each material are plotted.

    """

    start_time = time.time()

    assert IMat.shape == (Inprop, Inmat), f"\033[91mIMat shape mismatch: expected ({Inprop}, {Inmat}), got {IMat.shape}\033[0m"

    assert IMat_pred.shape == (Inprop, Inmat), f"\033[91mIMat_pred shape mismatch: expected ({Inprop}, {Inmat}), got {IMat_pred.shape}\033[0m"

    assert IMat_l_pred.shape == (Inprop, Inmat), f"\033[91mIMat_l_pred shape mismatch: expected ({Inprop}, {Inmat}), got {IMat_l_pred.shape}\033[0m"

    assert IMat_b_pred.shape == (Inprop, Inmat), f"\033[91mIMat_b_pred shape mismatch: expected ({Inprop}, {Inmat}), got {IMat_b_pred.shape}\033[0m"

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

def PlotBar(IMat, IMat_pred, IMat_l_pred, IMat_b_pred, Inprop, Inmat, Ifrac, irun, IIIidx_unknown_frac, IIIidx_known_frac):
    
    """

    Description:

    Input:

                IMat                ==>     The normalized or unnormalized completely known pressure-material matrix. (Inprop * Inmat)
                
                IMat_pred           ==>     The normalized or unnormalized latent-bias model predicted pressure-material matrix. (Inprop * Inmat)
                
                IMat_l_pred         ==>     The normalized or unnormalized latent only model predicted pressure-material matrix. (Inprop * Inmat)

                IMat_b_pred         ==>     The normalized or unnormalized bias only model predicted pressure-material matrix. (Inprop * Inmat)
                
                Inprop             ==>     Same as before 
                
                Inmat               ==>     Same as before 
                
                Ifrac               ==>     Same as before

                irun                ==> 
                
                IIIidx_unknown_frac ==> 
                
                IIIidx_known_frac   ==> 

    Output:

                Bar plots for each property are plotted.

    """

    start_time = time.time()

    assert IMat.shape == (Inprop, Inmat), f"\033[91mIMat shape mismatch: expected ({Inprop}, {Inmat}), got {IMat.shape}\033[0m"

    assert IMat_pred.shape == (Inprop, Inmat), f"\033[91mIMat_pred shape mismatch: expected ({Inprop}, {Inmat}), got {IMat_pred.shape}\033[0m"

    assert IMat_l_pred.shape == (Inprop, Inmat), f"\033[91mIMat_l_pred shape mismatch: expected ({Inprop}, {Inmat}), got {IMat_l_pred.shape}\033[0m"

    assert IMat_b_pred.shape == (Inprop, Inmat), f"\033[91mIMat_b_pred shape mismatch: expected ({Inprop}, {Inmat}), got {IMat_b_pred.shape}\033[0m"

    assert len(property_label) == Inprop, f"\033[91mMismatch in property labels: expected {Inprop}, got {len(property_label)}\033[0m"

    cfrac = []

    os.makedirs(f'5_FFig/{irun}/3_Bar_Plot/', exist_ok=True)

    r2_lb = []
    r2_lo = []
    r2_bo = []

    plt.figure(figsize=(11,7))

    for i in range(Inprop):

        # Filter indices to only get the 'missing' points for the CURRENT property (row i)
        current_press_test_idx = [m for (p, m) in IIIidx_unknown_frac if p == i]

        # Skip this property if the original matrix contains no missing entries.
        # During testing of the Existing-data case, this was first observed for the
        # second property (i = 1), where icfrac = 0.0 and both True_Test_i and
        # Pred_Test_i were empty. Since no entries are missing for this property,
        # there are no test points available for evaluating the GLRM predictions
        # and R²/RMSD cannot be calculated.
        if len(current_press_test_idx) == 0:
            r2_lb.append(np.nan)
            r2_lo.append(np.nan)
            r2_bo.append(np.nan)
            cfrac.append(0.0)

            print(f"\nFor {i}th property, no missing entries are available for testing. Skipping R² calculation.\n")

            continue

        # Ensuring we are extracting columns and not rows
        #assert all(isinstance(m, int) for m in current_press_test_idx), f"\033[91mFiltered indices must be material indices\033[0m"
        assert all(0 <= m < Inmat for m in current_press_test_idx), f"\033[91mFiltered indices contain invalid material indices.\033[0m"
            
        if args.existing:
            True_Test_i = Amp_true_T_stan[i, current_press_test_idx]
        else:
            True_Test_i   = IMat[i, current_press_test_idx]
        
        Pred_Test_i   = IMat_pred[i, current_press_test_idx]
        Pred_l_Test_i = IMat_l_pred[i, current_press_test_idx]
        Pred_b_Test_i = IMat_b_pred[i, current_press_test_idx]

        # Checking consistency on test set sizes
        # Note that the size of True_Test_i is not equal to Inmat but is < Inmat, since it contains test set points
        # Same logic goes for Pred_Test_i, Pred_l_Test_i, Pred_b_Test_i 
        assert len(current_press_test_idx) <= Inmat, f"\033[91mMismatch in selecting the test set. Test set size was expected to be less than {Inmat} but got {len(current_press_test_idx)}\033[0m"
        assert True_Test_i.shape[0]   == len(current_press_test_idx), f"\033[91mMismatch in selecting the test set. Test set size was expected to be {len(current_press_test_idx)} but got {True_Test_i.shape[0]}\033[0m"
        assert Pred_Test_i.shape[0]   == len(current_press_test_idx), f"\033[91mMismatch in selecting the test set. Test set size was expected to be {len(current_press_test_idx)} but got {Pred_Test_i.shape[0]}\033[0m"
        assert Pred_l_Test_i.shape[0] == len(current_press_test_idx), f"\033[91mMismatch in selecting the test set. Test set size was expected to be {len(current_press_test_idx)} but got {Pred_l_Test_i.shape[0]}\033[0m"
        assert Pred_b_Test_i.shape[0] == len(current_press_test_idx), f"\033[91mMismatch in selecting the test set. Test set size was expected to be {len(current_press_test_idx)} but got {Pred_b_Test_i.shape[0]}\033[0m"

        icfrac = len(current_press_test_idx) / Inmat
        
        assert icfrac <= 1.00, f"\033[91mExpected value of icfrac should be between 0 and 1 but actual value got is {icfrac}.\033[0m"
        
        cfrac.append(icfrac)
        print(f"\nFor {i} th column the icfrac is {icfrac}\n")

        # Calculate metrics for THIS pressure only
        print(f"True_Test_i is {True_Test_i}")
        print(f"Pred_Test_i is {Pred_Test_i}")

        TestMat   = compute_metrics(True_Test_i, Pred_Test_i)
        TestMat_l = compute_metrics(True_Test_i, Pred_l_Test_i)
        TestMat_b = compute_metrics(True_Test_i, Pred_b_Test_i)

        r2_lb.append(TestMat['R²'])
        r2_lo.append(TestMat_l['R²'])
        r2_bo.append(TestMat_b['R²'])

    np.savetxt(f"5_FFig/{irun}/3_Bar_Plot/cfrac.csv", cfrac, delimiter=",", fmt="%.2f")

    assert len(r2_lb) == Inprop, f"\033[91mMismatch in dimensions of r2_lb and Inprop. Expected {Inprop} got {r2_lb}\033[0m"
    assert len(r2_lo) == Inprop, f"\033[91mMismatch in dimensions of r2_lo and Inprop. Expected {Inprop} got {r2_lo}\033[0m"
    assert len(r2_bo) == Inprop, f"\033[91mMismatch in dimensions of r2_bo and Inprop. Expected {Inprop} got {r2_bo}\033[0m"

    assert not all(np.isnan(r2_lb)), f"\033[91mAll R² values are NaN. Something is wrong here\033[0m"
    assert not all(np.isnan(r2_lo)), f"\033[91mAll R² values are NaN. Something is wrong here\033[0m"
    assert not all(np.isnan(r2_bo)), f"\033[91mAll R² values are NaN. Something is wrong here\033[0m"

    x = np.arange(Inprop)

    plt.bar(x, r2_lb, label='LB')
    plt.scatter(x, r2_lo, marker='^', color='black', s=120, label='LO')
    plt.scatter(x, r2_bo, marker='*', color='gold', s=150, label='BO')

    pressure_label = []

    for label in property_label:

        exponent = re.search(r'10\^\{(-?\d+)\}', label).group(1)

        pressure_label.append(rf"$\mathbf{{1 \times 10^{{{exponent}}}}}$")

    plt.xlabel('Adsorption Property', fontdict=fontdict_x)
    #plt.ylabel('R2 unnormalized values', fontdict=fontdict_y)
    plt.ylabel(r'$\mathbf{R^2}$ of test set', fontdict=fontdict_y)

    #plt.title(f'Predictive performance along column of material-property matrix for ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)

    #plt.xticks(ticks = X, labels = X_label, rotation=90, fontdict={'fontsize': 23, 'fontweight': 'bold'})
    #plt.xticks(ticks = x, labels = X_label, rotation=90, fontsize=18, fontweight='bold') # fontweight='bold'
    plt.xticks(ticks=x, labels=pressure_label, rotation=90, fontsize=18, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold') # fontweight='bold'

    plt.legend(loc='upper right', fontsize=20)

    plt.tight_layout()

    plt.savefig(f"5_FFig/{irun}/3_Bar_Plot/R2_bar_plot.pdf", format='pdf')

    plt.close()

    #raise ValueError(f"DEBUGGING GOING !!!!")
        
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    return elapsed_time

def PlotBiasParity(IIIM, IIIM_b, IIIIfrac, IIIbesthp, IIInmat, IIIIirun):

    """
    
    Description:

        This function compares the material bias obtained from the latent-bias
        GLRM model with the material bias obtained from the bias-only model.
        The material biases are plotted against each other to assess their
        similarity and the corresponding R², MAE, and RMSE values are calculated.

    Input:

        IIIM                ==> Learned material matrix (M matrix) from the
                               latent-bias GLRM model. The last row corresponds
                               to the material bias.

        IIIM_b              ==> Learned material matrix (M matrix) from the
                               bias-only GLRM model. The only row corresponds
                               to the material bias.

        IIIIfrac            ==> Known fraction of entries in the
                               pressure-material matrix.

        IIIbesthp           ==> Dictionary containing the best hyperparameters
                               obtained from hyperparameter tuning.

        IIInmat             ==> Number of materials in the
                               pressure-material matrix.

        IIIIirun            ==> Run number for the current GLRM run.

    Output:

        A parity plot comparing the material bias from the latent-bias model
        and the bias-only model is saved. The elapsed time required to
        generate the plot is returned.

    """

    start_time = time.time()

    #np.savetxt("5_FFig/{IIIIirun}/4_Bias_Parity/IIIM.csv", IIIM, delimiter=",")
    #np.savetxt("5_FFig/{IIIIirun}/4_Bias_Parity/IIIM_b.csv", IIIM_b, delimiter=",")

    assert IIIM.shape == (IIIbesthp['dim'] + 1, IIInmat), f"\033[91mExpected IIIM shape {(IIIbesthp['dim'] + 1, IIInmat)}, but got {IIIM.shape}\033[0m"
    assert IIIM_b.shape == (1, IIInmat), f"\033[91mExpected IIIM_b shape {(1, IIInmat)}, but got {IIIM_b.shape}\033[0m"

    mat_bias_lb = IIIM[-1, :]
    mat_bias_b  = IIIM_b[-1, :]

    os.makedirs(f'5_FFig/{IIIIirun}/4_Bias_Parity/', exist_ok=True)   

    plt.figure(figsize=(11, 7))

    plt.scatter(mat_bias_lb, mat_bias_b, color='blue', label=f'Ifrac = {IIIIfrac*100:.0f}% known', alpha=0.6, edgecolors='black', s=80)
    
    xmin = min(np.min(mat_bias_lb), np.min(mat_bias_b))
    xmax = max(np.max(mat_bias_lb), np.max(mat_bias_b))

    plt.plot([xmin, xmax], [xmin, xmax],color='black', linestyle='--', label='Perfect Prediction')

    similarity_metrics = compute_metrics(mat_bias_lb, mat_bias_b)

    # Writing Model performance metrics on plot at the bottom left
    
    x_pos = 0.94
    y_pos = 0.04
    
    text = (f"{'Metric'.ljust(10)}\n"
            f"{'R²'.ljust(10)}{similarity_metrics['R²']:>10.3f}\n"
            f"{'MAE'.ljust(10)}{similarity_metrics['MAE']:>10.3f}\n"
            f"{'RMSE'.ljust(10)}{similarity_metrics['RMSE']:>10.3f}"
            )

    plt.gca().text(
        x_pos, y_pos, text,
        transform=plt.gca().transAxes,
        fontsize=18, color='black', weight='bold',
        horizontalalignment='right', verticalalignment='bottom',
        fontname='Times',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.7)
        )

    plt.xticks(fontsize=23, weight='bold')
    plt.yticks(fontsize=23, weight='bold')
    
    #plt.title(f'Predicted vs True values ({IIIIfrac*100:.0f}% known)', fontdict=fontdict_t)
    plt.title(f'Latent-Bias vs Bias-Only Material Bias ({IIIIfrac*100:.0f}% known)', fontdict=fontdict_t)
    
    plt.xlabel("Material Bias for Latent-Bias model", fontdict=fontdict_x)
    plt.ylabel("Material Bias for Bias-only model", fontdict=fontdict_y)
    
    plt.legend(loc='best', fontsize=20)
    
    plt.savefig(f"5_FFig/{IIIIirun}/4_Bias_Parity/Parity-material-bias.pdf", format='pdf')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    return elapsed_time

def PCAAfter(IM, IP, Ibesthp, Inmat, Inprop, Ioffval):

    """
    
    Description:
    
        This function performs principal component analysis (PCA) on the latent
        material and property vectors obtained from the GLRM. The bias component,
        if included in the GLRM, is excluded before performing PCA. The material
        and property latent vectors are combined and transformed into a common
        2D PCA space so that their relative positions and projections can be
        analyzed.
        
        The resulting 2D PCA coordinates are separated back into material and
        property vectors for subsequent analysis and visualization.
    
    Input:
    
        IM          ==> Material latent vectors obtained from the GLRM.
                       Shape: (k + 1, Inmat) when Ioffval = True and
                       (k, Inmat) when Ioffval = False.
        
        IP          ==> Property latent vectors obtained from the GLRM.
                       Shape: (k + 1, Inprop) when Ioffval = True and
                       (k, Inprop) when Ioffval = False.
        
        Ibesthp     ==> Dictionary containing the optimized GLRM hyperparameters.
                       In particular, Ibesthp['dim'] gives the latent dimension.
        
        Inmat       ==> Number of materials in the matrix.
        
        Inprop      ==> Number of properties in the matrix.
        
        Ioffval     ==> Boolean indicating whether the bias component is included
                       in the GLRM. If True, the last row corresponding to the
                       bias component is removed before PCA. If False, all latent
                       vectors are used.
    
    Output:
    
        MAct        ==> Material latent vectors after excluding the bias component
                       when Ioffval = True.
                       Shape: (k, Inmat).
        
        PAct        ==> Property latent vectors after excluding the bias component
                       when Ioffval = True.
                       Shape: (k, Inprop).
        
        AllLatVec   ==> Combined material and property latent vectors used for PCA.
                       Shape: (k, Inmat + Inprop).
        
        m_vecs_2D   ==> 2D PCA representation of the material latent vectors.
                       Shape: (2, Inmat).
        
        p_vecs_2D   ==> 2D PCA representation of the property latent vectors.
                       Shape: (2, Inprop).
    
    """

    # Stack M and P horizontally (excluding the last row from both)
    # Latent-Bias case
    if Ioffval == True:
        
        MAct = IM[:-1, :]  # shape: (k, Inmat) # For deployable latent-bias model
        PAct = IP[:-1, :]  # shape: (k, Inprop) # For deployable latent-bias model
        
        assert MAct.shape == (Ibesthp['dim'], Inmat), \
                f"\033[91mMismatch in Mact.shape: Expected ({Ibesthp['dim']}, {Inmat}), got {MAct.shape}\033[0m"
        assert PAct.shape == (Ibesthp['dim'], Inprop), \
                f"\033[91mMismatch in Pact.shape: Expected ({Ibesthp['dim']}, {Inprop}, got {PAct.shape})\033[0m"
    
    # Latent only case
    elif Ioffval == False:

        MAct = IM # shape: (k, Inmat) # For deployable latent-bias model
        PAct = IP # shape: (k, Inprop) # For deployable latent-bias model

        assert MAct.shape == (Ibesthp['dim'], Inmat), \
                f"\033[91mMismatch in Mact.shape: Expected ({Ibesthp['dim']}, {Inmat}), got {MAct.shape}\033[0m"
        assert PAct.shape == (Ibesthp['dim'], Inprop), \
                f"\033[91mMismatch in Pact.shape: Expected ({Ibesthp['dim']}, {Inprop}, got {PAct.shape})\033[0m"

    AllLatVec = np.hstack((MAct, PAct))  
    
    assert AllLatVec.shape == (Ibesthp["dim"], Inprop + Inmat)

    pca = PCA(n_components = 2)

    AllLatVec_T = AllLatVec.T

    p_pca = pca.fit_transform(AllLatVec_T) # Python's PCA function expects samples as rows

    p_pca_T = p_pca.T

    m_vecs_2D = p_pca_T[:, :Inmat]
    p_vecs_2D = p_pca_T[:, Inmat:]

    assert m_vecs_2D.shape == (2, Inmat)
    assert p_vecs_2D.shape == (2, Inprop)

    return MAct, PAct, AllLatVec, m_vecs_2D, p_vecs_2D

def PlotPCAAfter(Ip_vecs_2D, Im_vecs_2D, IMat, Inmat, Inprop, Ifrac, irun, Ioffval):

    """
    
    Description:
    
        This function plots the 2D PCA representation of the material and
        property latent vectors obtained from PCAAfter(). The first plot
        shows the material latent vectors together with the property vectors
        represented as arrows in the same PCA space.
        
        For each property, a separate PCA plot is then generated in which
        the material latent vectors are colored according to the corresponding
        property values. The property latent vector is also shown as an arrow
        to visualize the relationship between the property direction and the
        distribution of materials in the PCA space.
        
        The property vectors are normalized before being plotted so that their
        directions can be compared without their magnitudes affecting the
        visualization.

    Input:
    
        Ip_vecs_2D     ==> 2D PCA representation of the property latent vectors.
                          Shape: (2, Inprop).
        
        Im_vecs_2D     ==> 2D PCA representation of the material latent vectors.
                          Shape: (2, Inmat).
        
        IMat           ==> Normalized or unnormalized pressure-material matrix
                          containing the material property values used to color
                          the material points.
                          Shape: (Inprop, Inmat).
        
        Inmat          ==> Number of materials in the pressure-material matrix.
        
        Inprop         ==> Number of properties in the pressure-material matrix.
        
        Ifrac          ==> Known fraction of entries in the pressure-material
                          matrix.
        
        irun           ==> Run number for the current GLRM run.
        
        Ioffval        ==> Boolean indicating whether the bias component is
                          included in the GLRM. Used to identify the output
                          directory for the corresponding model.

    Output:
    
        PCA plots showing the material and property latent vectors in the
        common 2D PCA space are saved for the current GLRM run.
        
        A combined material-property PCA plot is generated along with
        individual material PCA plots for each property, where the material
        points are colored according to their corresponding property values.
        
        The elapsed time required to generate the plots is returned.

    """

    start_time = time.time()

    os.makedirs(f'5_FFig/{irun}/5_PCA_After/{Ioffval}', exist_ok=True)

    # Property Case

    assert Ip_vecs_2D.shape == (2, Inprop), f"\033[91mExpected {(2, Inprop)}, got {dg_pca.shape}\033[0m"
    
    colors = plt.cm.get_cmap('hsv', Inprop)
    
    norm_pca = np.linalg.norm(Ip_vecs_2D.T, axis=1)  # Find the magnitude of each vector
    normalized_pca = Ip_vecs_2D.T / norm_pca[:, np.newaxis]  # Normalize each vector by its magnitude

    plt.figure(figsize = (11,7))

    plt.axhline(0, color='black', linewidth=2)  # x-axis
    plt.axvline(0, color='black', linewidth=2)  # y-axis

    sc = plt.scatter(Im_vecs_2D.T[:,0], Im_vecs_2D.T[:,1], s=30, edgecolor='black', linewidth=1, alpha=0.75, facecolor='none')
    
    plt.xlabel('PCA dimention 1', fontdict=fontdict_x)
    plt.ylabel('PCA dimention 2', fontdict=fontdict_y)

    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')

    for i in range(Inprop):

        #val = float(Icname[i])
        #mantissa, exponent = f"{val:.1e}".split("e")
        #exponent = int(exponent)

        if normalized_pca[i,0] < 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 0.3 # 1.0
            normalized_pca[i,1] = normalized_pca[i,1] * 0.3 # 1.0
        elif normalized_pca[i,0] > 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 0.3 # 1.0
            normalized_pca[i,1] = normalized_pca[i,1] * 0.3 # 1.0

        plt.quiver(0.0, 0.0, normalized_pca[i,0], normalized_pca[i,1], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)

    #plt.savefig(f"5_PCA_After/{Ifrac}/{irun}/{Ioffval}/Mat_Prop_space.pdf", format='pdf')
    plt.savefig(f"5_FFig/{irun}/5_PCA_After/{Ioffval}/Mat_Prop_space.pdf", format='pdf')
        
    plt.close()

    # Material Case

    assert Im_vecs_2D.shape == (2, Inmat), f"\033[91mExpected {(2, Inmat)}, got {dg_pca.shape}\033[0m"

    cmap = get_cmap('rainbow', 7)

    norm       = MinMaxScaler()
    Mat_norm   = norm.fit_transform(IMat.T) # This is done for colorbar in plt.scatter()

    for i in range(Inprop):
        
        plt.figure(figsize = (11,7))

        plt.axhline(0, color='black', linewidth=2)  # x-axis
        plt.axvline(0, color='black', linewidth=2)  # y-axis

        #val = float(Icname[i])
        #mantissa, exponent = f"{val:.1e}".split("e")
        #exponent = int(exponent)

        sc = plt.scatter(Im_vecs_2D.T[:,0], Im_vecs_2D.T[:,1], s=30, c=Mat_norm[:,i], cmap = cmap, edgecolor='black', linewidth=1, alpha=0.75)

        cbar = plt.colorbar(sc)
        #cbar.set_label('\n' + rf'$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ bar}}$',fontsize=23)
        cbar.set_label('\n' + property_label[i], fontsize=20)
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

        if normalized_pca[i,0] < 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 1.0 # 0.3
            normalized_pca[i,1] = normalized_pca[i,1] * 1.0 # 0.3
        elif normalized_pca[i,0] > 0.0:
            normalized_pca[i,0] = normalized_pca[i,0] * 1.0 # 0.3
            normalized_pca[i,1] = normalized_pca[i,1] * 1.0 # 0.3

        #plt.quiver(0.0, 0.0, Ip_vecs_2D[0,i], Ip_vecs_2D[1,i], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)
        plt.quiver(0.0, 0.0, normalized_pca[i,0], normalized_pca[i,1], angles='xy', scale_units='xy', scale=1, color=colors(i), width=0.003, headwidth=6, headlength=7, headaxislength=6)

        #legend_patch = mpatches.Patch(color=colors(i), label=rf"$\mathbf{{Ar\ Uptake\ at\ Pressure\ {mantissa} \times 10^{{{exponent}}}\ bar}}$")
        legend_patch = mpatches.Patch(color=colors(i), label=property_label[i])
        
        plt.legend(handles=[legend_patch], fontsize=18, loc='best', frameon=True)

        plt.savefig(f"5_FFig/{irun}/5_PCA_After/{Ioffval}/{i}_Entire_space.pdf", format='pdf')
        plt.close()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    return elapsed_time

def PlotMatRanking(Ip_vecs_2D, Im_vecs_2D, IMat, Inmat, Inprop, Ifrac, irun, Ioffval):

    """
    
    Description:
    
        This function evaluates how well the material latent vectors reproduce
        the ranking of materials along each property direction in the 2D PCA
        space.
        
        For each property, the corresponding property latent vector is
        normalized and used as a projection direction. All material latent
        vectors are then projected onto this direction to obtain a scalar
        projection value for every material.
        
        The resulting material projections are compared with the corresponding
        true material property values using the Pearson correlation
        coefficient. A parity plot between the standardized true property
        values and the standardized material projections is also generated
        for each property.
        
        In addition to the correlation analysis, the function identifies the
        top and bottom 10, 100, 500, and 1000 materials according to the
        material projections and compares these rankings with the corresponding
        rankings obtained from the true material property values. The number
        of overlapping materials is recorded for each ranking threshold.
        
        The correlation coefficients and ranking overlaps are written to a log
        file for subsequent analysis.

    Input:
    
        Ip_vecs_2D     ==> 2D PCA representation of the property latent vectors.
                          Shape: (2, Inprop).
        
        Im_vecs_2D     ==> 2D PCA representation of the material latent vectors.
                          Shape: (2, Inmat).
        
        IMat           ==> Normalized or unnormalized pressure-material matrix
                          containing the true material property values.
                          Shape: (Inprop, Inmat).
        
        Inmat          ==> Number of materials in the pressure-material matrix.
        
        Inprop         ==> Number of properties in the pressure-material matrix.
        
        Ifrac          ==> Known fraction of entries in the pressure-material
                          matrix.
        
        irun           ==> Run number for the current GLRM run.
        
        Ioffval        ==> Boolean indicating whether the bias component is
                          included in the GLRM. Used to identify the output
                          directory for the corresponding model.

    Output:
    
        Projection plots are generated for each property, showing the
        relationship between the true material property values and the
        corresponding material projections.
        
        The Pearson correlation coefficient between the material projections
        and true material property values is calculated for each property.
        
        The top and bottom material rankings obtained from the projections are
        compared with the rankings obtained from the true material property
        values for the top/bottom 10, 100, 500, and 1000 materials.
        
        Correlation coefficients and ranking-overlap results are saved in an
        output log file.
        
        The elapsed time required to perform the analysis is returned.

    """

    start_time = time.time()
    
    from scipy.stats import pearsonr

    assert IMat.shape == (Inprop, Inmat), \
            f"\033[91mSize mismatch. Expected shape was ({Inprop, Inmat}) got {IMat.shape}\033[0m"
    
    tolerance = 1e-12
    
    top_10_mat_proj   = []
    top_100_mat_proj  = []
    top_500_mat_proj  = []
    top_1000_mat_proj = []

    bottom_10_mat_proj   = []
    bottom_100_mat_proj  = []
    bottom_500_mat_proj  = []
    bottom_1000_mat_proj = []

    os.makedirs(f'5_FFig/{irun}/6_MatRanking/{Ioffval}', exist_ok=True)

    log_file = open(f'5_FFig/{irun}/6_MatRanking/{Ioffval}/output.log', 'w')
    log_file.write(f"\nRun: {irun}, Ifrac: {Ifrac}, Ioffval: {Ioffval}\n\n")
    log_file.write(f"Correlation, top_10_overlap, top_100_overlap, top_500_overlap, top_1000_overlap,bottom_10_overlap, bottom_100_overlap, bottom_500_overlap, bottom_1000_overlap\n\n")

    assert Ip_vecs_2D.shape == (2, Inprop), f"\033[91mExpected {(2, Inprop)}, got {Ip_vecs_2D.shape}\033[0m"
    assert Im_vecs_2D.shape == (2, Inmat), f"\033[91mExpected {(2, Inmat)}, got {Im_vecs_2D.shape}\033[0m"

    Ip_vecs = Ip_vecs_2D.T
    Im_vecs = Im_vecs_2D.T

    for ip in range(Inprop):

        p_vecs = Ip_vecs[ip]

        # Taking all the materials projection along the ip pressure axis 
        p_unit = p_vecs / (np.linalg.norm(p_vecs) + tolerance) # tolerance to avoid unnessasary build up when np.linalg.norm(p_vecs) ~ 0
        MProj = Im_vecs @ p_unit

        if args.existing:
            iTrueUptake = Amp_true_T[ip, :]
        else:
            iTrueUptake = IMat[ip, :]
        
        assert iTrueUptake.shape == (Inmat,), f"Size mismatch. Expected {(Inmat,)}, got {iTrueUptake.shape}"
        assert MProj.shape == (Inmat,), f"Size mismatch. Expected {(Inmat,)}, got {MProj.shape}"

        if np.isnan(iTrueUptake).any():
            print(f"\033[91mNaN found in true uptake for property {ip}\033[0m")

        if np.isnan(MProj).any():
            print(f"\033[91mNaN found in material projection for property {ip}\033[0m")

        if np.std(iTrueUptake) < tolerance:
            print(f"\033[91mTrue uptake is constant for property {ip}\033[0m")

        if np.std(MProj) < tolerance:
            print(f"\033[91mMaterial projection is constant for property {ip}\033[0m")

        r,_ = pearsonr(MProj, iTrueUptake)

        plt.figure(figsize=(11, 7))

        scaler = StandardScaler()

        MProj_scaled = scaler.fit_transform(MProj.reshape(-1,1)).flatten()
        iTrue_scaled = scaler.fit_transform(iTrueUptake.reshape(-1,1)).flatten()

        # Even if we uncomment this one and comment the abpve two line the plots remain more or less the same
        '''
        combined = np.vstack([MProj, iTrueUptake]).T
        scaled = scaler.fit_transform(combined)

        MProj_scaled = scaled[:, 0]
        iTrue_scaled = scaled[:, 1]
        '''

        plt.scatter(iTrue_scaled, MProj_scaled, color='blue', label=f'Ifrac = {Ifrac*100:.0f}% known', alpha=0.6, edgecolors='black', s=80)

        xmin = min(np.min(iTrue_scaled), np.min(MProj_scaled))
        xmax = max(np.max(iTrue_scaled), np.max(MProj_scaled))
       
        plt.plot([xmin, xmax], [xmin, xmax],color='black', linestyle='--', label='Perfect Alignment')

        # Writing Model performance metrics on plot at the bottom left
        
        x_pos = 0.94
        y_pos = 0.04

        text = (f"{'Metric'.ljust(10)}\n"
                f"{'rho'.ljust(10)}{r:>10.3f}"
                )

        plt.gca().text(
            x_pos, y_pos, text,
            transform=plt.gca().transAxes,
            fontsize=18, color='black', weight='bold',
            horizontalalignment='right', verticalalignment='bottom',
            fontname='Times',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.7)
            )

        plt.xticks(fontsize=23, weight='bold')
        plt.yticks(fontsize=23, weight='bold')
        
        plt.title(f'Projections (PCA space) vs true uptake ({Ifrac*100:.0f}% known)', fontdict=fontdict_t)
        
        plt.xlabel("True normalized uptake of Ar at for all the materials ", fontdict=fontdict_x)
        plt.ylabel("Projection of material embeddings", fontdict=fontdict_y)
        
        plt.legend(loc='best', fontsize=20)
        
        plt.savefig(f"5_FFig/{irun}/6_MatRanking/{Ioffval}/{ip}_Projection.pdf", format='pdf')

        plt.close()

        #-----------------------------------------------------------------------------------------#
        # Identifying top materials along the pressure direction ip from 2D PCA projections / plots
        #-----------------------------------------------------------------------------------------#

        top_10_materials = np.argsort(-MProj)[:10]
        top_10_mat_proj.append(top_10_materials)
        true_top_10      = np.argsort(-iTrueUptake)[:10]
        top_10_overlap   = len(set(top_10_materials) & set(true_top_10))

        top_100_materials = np.argsort(-MProj)[:100]
        top_100_mat_proj.append(top_100_materials)
        true_top_100      = np.argsort(-iTrueUptake)[:100]
        top_100_overlap   = len(set(top_100_materials) & set(true_top_100))

        top_500_materials = np.argsort(-MProj)[:500]
        top_500_mat_proj.append(top_500_materials)
        true_top_500      = np.argsort(-iTrueUptake)[:500]
        top_500_overlap   = len(set(top_500_materials) & set(true_top_500))

        top_1000_materials = np.argsort(-MProj)[:1000]
        top_1000_mat_proj.append(top_1000_materials)
        true_top_1000      = np.argsort(-iTrueUptake)[:1000]
        top_1000_overlap   = len(set(top_1000_materials) & set(true_top_1000))

        #--------------------------------------------------------------------------------------------#
        # Identifying bottom materials along the pressure direction ip from 2D PCA projections / plots
        #--------------------------------------------------------------------------------------------#

        bottom_10_materials = np.argsort(MProj)[:10]
        bottom_10_mat_proj.append(bottom_10_materials)
        true_bottom_10      = np.argsort(iTrueUptake)[:10]
        bottom_10_overlap   = len(set(bottom_10_materials) & set(true_bottom_10))

        bottom_100_materials = np.argsort(MProj)[:100]
        bottom_100_mat_proj.append(bottom_100_materials)
        true_bottom_100      = np.argsort(iTrueUptake)[:100]
        bottom_100_overlap   = len(set(bottom_100_materials) & set(true_bottom_100))

        bottom_500_materials = np.argsort(MProj)[:500]
        bottom_500_mat_proj.append(bottom_500_materials)
        true_bottom_500      = np.argsort(iTrueUptake)[:500]
        bottom_500_overlap   = len(set(bottom_500_materials) & set(true_bottom_500))

        bottom_1000_materials = np.argsort(MProj)[:1000]
        bottom_1000_mat_proj.append(bottom_1000_materials)
        true_bottom_1000      = np.argsort(iTrueUptake)[:1000]
        bottom_1000_overlap   = len(set(bottom_1000_materials) & set(true_bottom_1000))

        log_file.write(f"{r:.3f},{top_10_overlap},{top_100_overlap},{top_500_overlap},{top_1000_overlap},{bottom_10_overlap},{bottom_100_overlap},{bottom_500_overlap},{bottom_1000_overlap}\n")

        #print(f"Pressure {ip} Top-10 overlap = {overlap}/10")

    log_file.close()
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    assert len(top_10_mat_proj) == Inprop, f"Size mismatch Expected {Inprop}, got {len(top_10_mat_proj)}"

    for itop in top_10_mat_proj:
        assert len(itop) == 10, f"Size mismatch. Expected 10, got {len(itop)}" 

def PlotCosineSimilarity(IPAct, Ibesthp, Inprop, Ifrac, irun, Ioffval):

    """
    
    Description:
    
        This function calculates the pairwise cosine similarity between all
        property latent vectors obtained from the GLRM and visualizes the
        resulting property-property similarity matrix as a heatmap.
        
        The cosine similarity measures the angular similarity between two
        property latent vectors. A value close to +1 indicates that two
        properties have similar directions in the latent space, a value close
        to -1 indicates opposite directions, and a value close to 0 indicates
        that the two properties are approximately orthogonal.
        
        The resulting cosine similarity matrix is symmetric, with the diagonal
        elements representing the self-similarity of each property.

    Input:
    
        IPAct          ==> Property latent vectors obtained from the GLRM.
                          Shape: (Ibesthp['dim'], Inprop).
        
        Ibesthp        ==> Dictionary containing the optimal GLRM hyperparameters,
                          including the latent dimension ('dim').
        
        Inprop         ==> Number of properties in the pressure-material matrix.
        
        Ifrac          ==> Known fraction of entries in the pressure-material
                          matrix. Used for identifying the current GLRM run.
        
        irun           ==> Run number for the current GLRM run.
        
        Ioffval        ==> Boolean indicating whether the bias component is
                          included in the GLRM. Used to identify the output
                          directory for the corresponding model.

    Output:
    
        cos_sim        ==> Pairwise cosine similarity matrix between all property
                          latent vectors. Shape: (Inprop, Inprop).
        
        A heatmap of the property-property cosine similarity matrix is saved
        as a PDF file for subsequent analysis.

    """

    start_time = time.time()

    tolerance = 1e-12

    assert IPAct.shape == (Ibesthp['dim'], Inprop), \
        f"Mismatch: Expected ({Ibesthp['dim']}, {Inprop}), got {IPAct.shape}"

    os.makedirs(f'5_FFig/{irun}/7_CosineSimilarity/{Ioffval}', exist_ok=True)

    # p_vecs each row = one pressure vector
    p_vecs = IPAct.T   # shape: (Inprop, k)

    cos_sim = np.zeros((Inprop, Inprop))

    for ip in range(Inprop):
        for jp in range(Inprop):

            p_i = p_vecs[ip]
            p_j = p_vecs[jp]

            norm_i = np.linalg.norm(p_i) + tolerance
            norm_j = np.linalg.norm(p_j) + tolerance

            cos_sim[ip, jp] = np.dot(p_i, p_j) / (norm_i * norm_j)

    #plt.figure(figsize=(10, 8))

    colors = sns.color_palette("hls", 8)   # 8 distinct hues
    cmap_hls = LinearSegmentedColormap.from_list("hls_continuous", colors, N=256)

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(
        cos_sim,
        cmap=cmap_hls,
        linewidths=1,
        linecolor='black',
        xticklabels=True,
        yticklabels=True
    )

    cbar = ax.collections[0].colorbar

    cbar.ax.tick_params(labelsize=18, width=1, color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

    ax.collections[0].colorbar.outline.set(linewidth=2, edgecolor='black')

    plt.xticks(fontsize=13, weight='bold', rotation=90)
    #plt.yticks(fontsize=13, weight='bold', rotation=90)
    plt.yticks(fontsize=13, weight='bold')

    plt.xlabel("Property index", fontdict=fontdict_x)
    plt.ylabel("Property index", fontdict=fontdict_y)

    #plt.title(f"Pressure–Pressure Cosine Similarity (latent space, k={Ibesthp['dim']})", fontdict=fontdict_t)
    plt.title(f"Property-property Cosine Similarity (latent space, k={Ibesthp['dim']})")

    plt.savefig(f'5_FFig/{irun}/7_CosineSimilarity/{Ioffval}/cosine_heatmap.pdf', format='pdf')
    plt.close()

    elapsed_time = (time.time() - start_time) / 60

    return cos_sim

def PlotLatGeoIsotherms(IMAct, IPAct, Imat, Inmat, Inprop, Ibesthp, mat_ids, IIIMMat_stan_mean, IIIMMat_stan_std, irun, Ioffval):

    """
    
    Description:
    
        This function generates adsorption isotherms for selected materials
        using the material and property latent vectors obtained from the GLRM.
        
        For each selected material, the material latent vector is combined with
        the property latent vectors to reconstruct the adsorption properties
        at the corresponding pressure points. The predicted values are then
        compared with the true adsorption values from the pressure-material
        matrix.
        
        The reconstructed and true values are initially expressed in the
        standardized space used during GLRM training. The corresponding
        standardization mean and standard deviation are therefore used to
        transform both the predicted and true values back to their original
        physical scale before plotting.
        
        Only properties identified as adsorption properties are used for
        constructing the isotherms. The pressure values are extracted directly
        from the property column names.
        
        The function generates an individual isotherm plot for every selected
        material, allowing the latent-geometry reconstruction to be compared
        directly with the corresponding true adsorption isotherm.

    Input:
    
        IMAct          ==> Material latent vectors obtained from the GLRM after
                          removing the bias component when applicable.
                          Shape: (Ibesthp['dim'], Inmat).
        
        IPAct          ==> Property latent vectors obtained from the GLRM after
                          removing the bias component when applicable.
                          Shape: (Ibesthp['dim'], Inprop).
        
        Imat           ==> Normalized pressure-material matrix containing the
                          true material property values used for comparison.
                          Shape: (Inprop, Inmat).
        
        Inmat          ==> Number of materials in the pressure-material matrix.
        
        Inprop         ==> Number of properties in the pressure-material matrix.
        
        Ibesthp        ==> Dictionary containing the optimal GLRM hyperparameters,
                          including the latent dimension ('dim').
        
        mat_ids        ==> List of material indices for which adsorption
                          isotherms are to be plotted.
        
        IIIMMat_stan_mean
                       ==> Mean values used to standardize the pressure-material
                          matrix. Shape: (Inprop, 1).
        
        IIIMMat_stan_std
                       ==> Standard deviations used to standardize the
                          pressure-material matrix. Shape: (Inprop, 1).
        
        irun           ==> Run number for the current GLRM run.
        
        Ioffval        ==> Boolean indicating whether the bias component is
                          included in the GLRM. Used to identify the output
                          directory for the corresponding model.

    Output:
    
        An adsorption isotherm plot is generated for each material specified
        in mat_ids. The plots show the latent-geometry prediction and the true
        adsorption isotherm as a function of pressure.
        
        The predicted and true adsorption values are converted back from the
        standardized representation to their original physical scale before
        plotting.
        
        The generated isotherm plots are saved as PDF files for subsequent
        analysis.

    """

    if Imat.shape != IIIMMat_stan_mean.shape and IIIMMat_stan_mean.size != 1 and not np.allclose(np.shape(IIIMMat_stan_mean), np.shape(IIIMMat_stan_std)):
        raise ValueError("\033[91mImat, IIIMMat_stan_mean, and IIIMMat_stan_std must have compatible sizes (same shape or scalar values).\033[0m")

    assert Imat.shape == (Inprop, Inmat), \
        f"\033[91mWrong Pressure-material matrix is supplied to PlotLatGeoIsotherms. Expected shape was ({Inprop, Inmat}) got {Imat.shape}\033[0m"

    assert IIIMMat_stan_mean.shape == (Inprop, 1), \
        f"\033[91mSize mismatch in PlotLatGeoIsotherms. Expected shape was ({Inprop, Inmat}) got {IIIMMat_stan_mean.shape}\033[0m"

    assert IIIMMat_stan_std.shape == (Inprop, 1), \
        f"\033[91mSize mismatch in PlotLatGeoIsotherms. Expected shape was ({Inprop, Inmat}) got {IIIMMat_stan_std.shape}\033[0m"

    assert IMAct.shape == (Ibesthp['dim'], Inmat), \
            f"\033[91mMismatch in Mact.shape: Expected ({Ibesthp['dim']}, {Inmat}), got {IMAct.shape}\033[0m"

    assert IPAct.shape == (Ibesthp['dim'], Inprop), \
            f"\033[91mMismatch in Pact.shape: Expected ({Ibesthp['dim']}, {Inprop}, got {IPAct.shape})\033[0m"

    os.makedirs(f'5_FFig/{irun}/8_LatGeoIsotherm/{Ioffval}/', exist_ok=True)

    adsorption_idx = []

    for i, column in enumerate(cname):

        if "_uptake_" in column:
            adsorption_idx.append(i)

    assert len(adsorption_idx) > 0, f"\033[91mNo adsorption properties found. PlotLatGeoIsotherms requires adsorption properties.\033[0m"

    X = []

    for i in adsorption_idx:

        parts = cname[i].split("_")

        assert len(parts) == 5, \
                f"\033[91mInvalid adsorption column name: {cname[i]}. Expected format: Adsorbate_uptake_(mol/kg)_Pressurebar_TemperatureK\033[0m"

        Pressure = float(parts[3].replace("bar", ""))

        X.append(Pressure)

    X = np.array(X)

    Adsorbate = cname[0].split("_")[0]

    assert len(X) == Inprop, f"\033[91mMismatch in pressure points. Expected {Inprop}, got {len(X)}\033[0m"

    for mid in mat_ids:

        plt.figure(figsize=(8,6))

        m_vec = IMAct[:, mid]

        #pred_iso = IPAct.T @ m_vec # entire predicted isotherm
        #true_iso = Imat[:, mid]

        pred_iso = (IPAct[:, adsorption_idx].T @ m_vec)
        true_iso = Imat[adsorption_idx, mid]

        #assert pred_iso.shape == (Inprop,), f"Expected {(Inprop,)}, got {pred_iso.shape}"
        #assert true_iso.shape == (Inprop,), f"Expected {(Inprop,)}, got {true_iso.shape}"

        assert pred_iso.shape == (len(adsorption_idx),), f"Expected {(len(adsorption_idx),)}, got {pred_iso.shape}"
        assert true_iso.shape == (len(adsorption_idx),), f"Expected {(len(adsorption_idx),)}, got {true_iso.shape}"

        #mean_vec = IIIMMat_stan_mean[:, 0]
        #std_vec  = IIIMMat_stan_std[:, 0]

        mean_vec = IIIMMat_stan_mean[adsorption_idx, 0]
        std_vec  = IIIMMat_stan_std[adsorption_idx, 0]

        assert mean_vec.shape == pred_iso.shape, f"Shape mismatch"
        assert std_vec.shape  == pred_iso.shape, f"Shape mismatch"

        pred_iso = (pred_iso * std_vec) + mean_vec
        true_iso = (true_iso * std_vec) + mean_vec

        plt.scatter(X, pred_iso, s=100, marker='s', linestyle='-', facecolors='red', edgecolors='red', linewidth=1.5, alpha=1.0, label="Latent Prediction")
        plt.scatter(X, true_iso, s=100, marker='o', linestyle='-', facecolors='green', edgecolors='green', linewidth=1.5, alpha=1.0, label="True Isotherm")

        plt.xticks(fontsize=20, fontweight='bold') # fontweight='bold'
        plt.yticks(fontsize=20, fontweight='bold') # fontweight='bold'

        plt.legend(loc='best', fontsize=20)

        plt.xlabel('Pressure (bar)', fontdict=fontdict_x)
        #plt.ylabel('Argon uptake (mol/kg)', fontdict=fontdict_y)
        plt.ylabel(f'{Adsorbate} uptake (mol/kg)', fontdict=fontdict_y)
        
        plt.title(f"Material {mid}: Isotherm from latent geometry (k={Ibesthp['dim']})")

        plt.savefig(f'5_FFig/{irun}/8_LatGeoIsotherm/{Ioffval}/{mid}.pdf', format='pdf')

        plt.close()

def InverseDesign(IPAct, IMAct, ip, Inprop, Ibesthp, n_hmof, irun, Ioffval):
    
    """
    Description:
        Generate candidate materials in latent space aligned with a given pressure direction.

    Inputs:
        IPAct        ==> Pressure latent matrix (k × Inprop)
        IMAct        ==> Material latent matrix (k × Inmat)
        ip           ==> Pressure index for which inverse design is to be performed
        Inprop      ==> 
        Ibesthp      ==> Dict containing best hyperparameters
        n_hmof       ==> Number of candidate materials to generate
        irun         ==>
        Ioffval      ==>

    Outputs:
        hmof         ==> All generated material vectors
        huptake      ==> All generated materials adsorption isotherm

    # ------------------------------------------------------------
    # NOTE TO SELF (very important intuition about mod):
    #
    #   mod_min      ==> Minimum scaling along pressure direction
    #   mod_max      ==> Maximum scaling along pressure direction
    #
    # mod is NOT some absolute "quality" number like >1 = good, <1 = bad.
    # That thinking is wrong.
    #
    # What mod is actually doing:
    #     m = mod * p_unit  →  ||m|| = mod
    #
    # So mod is just controlling how "strong" the material is along
    # the pressure direction.
    #
    # Since uptake = m^T p, increasing mod just scales adsorption linearly.
    #
    # BUT the key point:
    #     mod only makes sense RELATIVE to real materials.
    #
    # So always compare with:
    #     real_norms = ||IMAct||
    #
    # Think like this:
    #
    # Example:
    #     if real norms are ~ [1.5, 2.0, 2.5, 3.0]
    #
    # Then:
    #     mod = 0.5   ==> very weak material (worse than anything in dataset)
    #     mod = 2.0   ==> realistic material (like ones we already have)
    #     mod = 3.5   ==> slightly better than best known material
    #     mod = 10    ==> nonsense / extreme extrapolation
    #
    # So the goal is NOT to pick mod > 1.
    # The goal is to span:
    #     weak → realistic ==> slightly better
    #
    # This way we:
    #     - see the trend clearly
    #     - verify model behavior
    #     - show proof-of-concept properly
    #
    # In short:
    #     mod = strength of material (in latent space)
    #     NOT an absolute performance metric
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # NOTE TO SELF (very important intuition about selectivity):
    #
    # In the current construction:
    #     m = mod * p_unit
    #
    # we are ONLY scaling the magnitude of the material along a
    # fixed direction (given by p_vec).
    #
    # So what happens?
    #
    # uptake at ANY pressure j:
    #     uptake_j = m^T p_j = mod * (p_unit^T p_j)
    #
    # ==> increasing mod increases uptake at ALL pressures
    #    in a proportional (linear) way
    #
    # Example:
    #     mod = 1   ==> uptake = [1, 2, 3]
    #     mod = 2   ==> uptake = [2, 4, 6]
    #     mod = 5   ==> uptake = [5,10,15]
    #
    # So the SHAPE of the isotherm does NOT change
    # only the magnitude changes
    #
    # Now for selectivity:
    #
    #     S_ij = uptake_i / uptake_j
    #
    # since both numerator and denominator scale with mod:
    #
    #     S_ij = (mod * A) / (mod * B) = A / B
    #
    # ==> mod cancels out
    #
    # ==> SELECTIVITY DOES NOT CHANGE with mod
    #
    # This means:
    #     current method = increasing adsorption strength only
    #     NOT tuning selectivity
    #
    # Important consequence:
    #     we are exploring a 1D line in latent space
    #     → cannot change relative preference between pressures
    #
    # If we want to control selectivity, we need to change direction:
    #
    #     m = alpha * p_i  -  beta * p_j
    #
    # i.e.:
    #     - align with desired pressure
    #     - suppress competing pressure
    #
    # But that is a next step — current goal is only proof-of-concept
    # for generating high-performing materials.
    # ------------------------------------------------------------

    """

    tolerance = 1e-12

    # -------------------------------
    # Sanity checks
    # -------------------------------
    assert IPAct.shape[0] == Ibesthp['dim'], f"Expected IPAct shape ({Ibesthp['dim']}, Inprop), got {IPAct.shape}"
    assert IMAct.shape[0] == Ibesthp['dim'], f"Expected IMAct shape ({Ibesthp['dim']}, Inmat), got {IMAct.shape}"

    os.makedirs(f'5_FFig/{irun}/InverseDesign/{Ioffval}', exist_ok=True)

    log_file = open(f'5_FFig/{irun}/InverseDesign/{Ioffval}/output.log', 'w')
    #with open(f'5_FFig/{irun}/InverseDesign/{Ioffval}/output.log', 'w') as log_file:

    real_norms = np.linalg.norm(IMAct, axis=0)

    # -------------------------------
    # Get pressure vector
    # -------------------------------
    p_vec = IPAct[:, ip]
    assert p_vec.shape == (Ibesthp['dim'],), f"Expected {(Ibesthp['dim'],)}, got {p_vec.shape}"

    norm_p = np.linalg.norm(p_vec) + tolerance
    p_unit = p_vec / norm_p

    # projection-based baseline. Starting point has some alibi
    mod_min = np.max(IMAct.T @ p_unit)
    mod_max = 2 * mod_min

    # Starting point is purely hypothetical
    #mod_min = 0.5 * np.min(real_norms)
    #mod_max = 1.2 * np.max(real_norms)

    log_file.write(f"mod_min is {mod_min}\n")
    log_file.write(f"mod_max is {mod_max}\n\n")

    # -------------------------------
    # Generate candidates
    # -------------------------------
    
    # These are the manitutes of the n_candidates materials
    mods = np.linspace(mod_min, mod_max, n_hmof)  

    hmof    = []
    huptake = []

    for imod in mods:
        
        # multiplying the manitute with unite vector, there by creating the hypothetical materials
        ihmof = imod * p_unit 

        assert np.isclose(np.linalg.norm(ihmof), imod, atol=1e-10), \
            f"\033[91mNorm mismatch: Expected {imod}, got {np.linalg.norm(ihmof)}\033[0m"
        
        assert ihmof.shape == (Ibesthp['dim'],), \
            f"\033[91mShape mismatch. Expected shape was ({Ibesthp['dim']},) got {ihmof.shape}\033[0m"

        log_file.write(f"ihmof generated for {imod:.4f} is {ihmof}\n\n")
 
        # Predicted isotherm for the hypothetical material
        ihuptake = ihmof @ IPAct   # shape: (Inprop,)
        
        assert ihuptake.shape == (Inprop,), \
            f"\033[91mShape mismatch. Expected shape was ({Inprop},) got {ihuptake.shape}\033[0m"

        hmof.append(ihmof)
        huptake.append(ihuptake)

    hmof    = np.array(hmof)    # shape: (n_hmof, k). Each row corresponds to one material
    huptake = np.array(huptake) # shape: (n_hmof, Inprop). Each row corresponds to entire isotherm for one material

    assert hmof.shape == (n_hmof, Ibesthp['dim']), \
        f"\033[91mShape mismatch. Expected shape was ({n_hmof, Ibesthp['dim']}) got {hmof.shape}\033[0m"

    assert huptake.shape == (n_hmof, Inprop), \
        f"\033[91mShape mismatch. Expected shape was ({n_hmof, Inprop}) got {huptake.shape}\033[0m"

    target_uptake = huptake[:, ip]

    # If my intuition is right then as mod increases the uptake at that pressure should also increase
    # meaning larger mod ==> higher uptake
    assert np.all(np.diff(target_uptake) >= -1e-10), \
        "Uptake is not monotonically increasing with mod"

    gen_norms  = np.linalg.norm(hmof, axis=1)

    log_file.write(f"Max real norm is {np.max(real_norms)}\n")
    log_file.write(f"Max hypo norm is {np.max(gen_norms)}\n")
    log_file.write(f"Min real norm is {np.min(real_norms)}\n")
    log_file.write(f"Min hypo norm is {np.min(gen_norms)}\n\n")

    log_file.write(f"\nTarget uptake range: {target_uptake.min()} to {target_uptake.max()}\n")

    log_file.close()

    return hmof, huptake

def InverseDesign_Perturbation(IPAct, IMAct, ip, Inprop, Inmat, Imat, Ibesthp, n_hmof, irun, top_k, Ioffval):
    
    """
    Description:
        Generate candidate materials by perturbing existing materials along a pressure direction.

    Inputs:
        IPAct        ==> Pressure latent matrix (k × Inprop)
        IMAct        ==> Material latent matrix (k × Inmat)
        ip           ==> Target pressure index
        Inprop      ==> Number of pressures
        Inmat        ==> 
        Imat         ==> 
        Ibesthp      ==> Dict containing best hyperparameters
        n_hmof       ==> Number of candidate materials per base material
        irun         ==> Run index
        Ioffval      ==> Offset flag
        top_k        ==> Number of top materials to perturb (default = 1)

    Outputs:
        hmof         ==> Generated material vectors (n_total, k)
        huptake      ==> Predicted isotherms (n_total, Inprop)

    # ------------------------------------------------------------
    # NOTE TO SELF (core idea of this method):
    #
    # Earlier (v1):
    #     m = mod * p_unit
    #     ==> completely synthetic materials along a single direction
    #
    # Now (v2):
    #     m_new = m_real + mod * p_unit
    #
    # Meaning:
    #     - start from a REAL material
    #     - then push it along adsorption-relevant direction
    #
    # So instead of:
    #     "generate from scratch"
    #
    # we are doing:
    #     "improve existing materials"
    #
    # This is more physically meaningful and interpretable.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # NOTE TO SELF (important geometry insight):
    #
    # v1 explored:
    #     a single line passing through origin
    #
    # v2 explores:
    #     multiple lines, each passing through a real material
    #
    # So:
    #     more diversity
    #     more realistic candidates
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Important information about this line : `m_real = IMAct[:, imat]`
    #
    # Here I am selecting top materials based on REAL uptake:
    #     real_scores = Imat[:, ip]
    #
    # BUT perturbation is done in LATENT space:
    #     m_real = IMAct[:, imat]
    #
    # So I am mixing two spaces:
    #     - selection in original (data) space
    #     - generation in latent space
    #
    # This implicitly assumes:
    #     "ranking of materials is preserved in latent space"
    #
    # i.e.:
    #     if material A > material B in real uptake,
    #     then m_A · p ≈ m_B · p should also reflect that
    #
    # This may not always be strictly true because GLRM is an approximation.
    #
    # So there is a small conceptual risk:
    #     top materials in real space may not be perfectly aligned
    #     with top materials in latent space
    #
    # HOWEVER:
    #     this approach is still useful because:
    #     - selection is physically grounded (based on real data)
    #     - latent space is only used for controlled perturbation
    #
    # Best practice (which I am doing below):
    #     compare real vs latent rankings and log them
    #
    # If rankings are similar ==> method is consistent
    # If not ==> latent space may not be trustworthy for design
    # ------------------------------------------------------------
    """

    tolerance = 1e-12

    # -------------------------------
    # Sanity checks
    # -------------------------------
    assert IPAct.shape[0] == Ibesthp['dim'], \
        f"Expected IPAct shape ({Ibesthp['dim']}, Inprop), got {IPAct.shape}"

    assert IMAct.shape[0] == Ibesthp['dim'], \
        f"Expected IMAct shape ({Ibesthp['dim']}, Inmat), got {IMAct.shape}"

    assert Imat.shape == (Inprop, Inmat), \
        f"\033[91mWrong Pressure-material matrix is supplied to PlotLatGeoIsotherms. Expected shape was ({Inprop, Inmat}) got {Imat.shape}\033[0m"

    os.makedirs(f'5_FFig/{irun}/InverseDesign_Perturbation/{Ioffval}', exist_ok=True)

    with open(f'5_FFig/{irun}/InverseDesign_Perturbation/{Ioffval}/output.log', 'w') as log_file:

        # -------------------------------
        # Get pressure vector
        # -------------------------------
        p_vec = IPAct[:, ip]
        assert p_vec.shape == (Ibesthp['dim'],), f"Expected {(Ibesthp['dim'],)}, got {p_vec.shape}"

        norm_p = np.linalg.norm(p_vec) + tolerance
        p_unit = p_vec / norm_p

        # -------------------------------
        # Select top materials
        # -------------------------------

        #real_scores   = Imat[:, ip]
        real_scores    = Imat[ip, :]
        assert real_scores.shape == (Inmat,), f"Shape mismatch. Expected {(Inmat,)} but got {(real_scores.shape)}"
        top_indices = np.argsort(real_scores)[-top_k:]

        latent_scores = IMAct.T @ p_vec

        log_file.write("Top indices (real): {}\n".format(top_indices))
        log_file.write("Top indices (latent): {}\n".format(np.argsort(latent_scores)[-top_k:]))
        
        log_file.write(f"Top material indices: {top_indices}\n\n")

        hmof = []
        huptake = []

        # -------------------------------
        # Loop over base materials
        # -------------------------------
        for imat in top_indices:

            m_real = IMAct[:, imat]
            assert m_real.shape == (Ibesthp['dim'],), f"Shape mismatch. Expected {(Ibesthp['dim'],)} but got {m_real.shape}"
            norm_real = np.linalg.norm(m_real)

            log_file.write(f"\nBase material index: {imat}\n")
            log_file.write(f"||m_real|| = {norm_real}\n\n")

            overlap = set(top_indices).intersection(set(np.argsort(latent_scores)[-top_k:]))
            log_file.write(f"Overlap between real and latent top-k: {overlap}\n\n")

            # -------------------------------
            # Define perturbation range
            # -------------------------------
            mod_min = 0.0
            mod_max = 0.2 * norm_real   # small perturbation

            mods = np.linspace(mod_min, mod_max, n_hmof)
            assert mods.shape == (n_hmof,), f"Shape mismatch. Expected {(n_hmof,)} but got {mods.shape}"

            for imod in mods:

                # -------------------------------
                # Perturbation
                # -------------------------------
                ihmof = m_real + imod * p_unit

                assert ihmof.shape == (Ibesthp['dim'],), \
                    f"\033[91mShape mismatch. Expected shape was ({Ibesthp['dim']},) got {ihmof.shape}\033[0m"

                log_file.write(f"ihmof generated for {imod:.4f} is {ihmof}\n\n")

                # Predicted isotherm for the hypothetical material
                ihuptake = ihmof @ IPAct # shape: (Inprop,)

                assert ihuptake.shape == (Inprop,), \
                    f"\033[91mShape mismatch. Expected shape was ({Inprop},) got {ihuptake.shape}\033[0m"

                hmof.append(ihmof)
                huptake.append(ihuptake)

        hmof = np.array(hmof)
        huptake = np.array(huptake)

        expected_n = n_hmof * top_k

        assert hmof.shape == (expected_n, Ibesthp['dim']), \
            f"\033[91mShape mismatch. Expected shape was ({expected_n, Ibesthp['dim']}) got {hmof.shape}\033[0m"

        assert huptake.shape == (expected_n, Inprop), \
            f"\033[91mShape mismatch. Expected shape was ({expected_n, Inprop}) got {huptake.shape}\033[0m"

        target_uptake = huptake[:, ip]

        log_file.write(f"\nTarget uptake range: {target_uptake.min()} to {target_uptake.max()}\n")
        log_file.write(f"\nGenerated {hmof.shape[0]} materials\n")

    return hmof, huptake

def PlotInverseDesign(IPAct, IMAct, candidates, ip, Ibesthp, Inmat, Inprop, Icname, irun, Ioffval):

    """
    Description:
        Plot inverse-designed materials (for a single pressure) in PCA space.

    Inputs:
        IPAct       ==> (k x Inprop)
        IMAct       ==> (k x Inmat)
        candidates  ==> (n_candidates x k)
        ip          ==> pressure index used for design
    """

    # ----------------------------
    # Step 1: Prepare matrices
    # ----------------------------
    candidates_T = candidates.T  # (k x n_candidates)

    # Combine original + designed
    M_combined = np.hstack((IMAct, candidates_T))  # (k x Inmat + n_candidates)

    # Stack with pressures (same as your PCAAfter)
    AllLatVec = np.hstack((M_combined, IPAct))  # (k x total)

    # ----------------------------
    # Step 2: PCA
    # ----------------------------
    AllLatVec_T = AllLatVec.T

    pca = PCA(n_components=2)
    p_pca = pca.fit_transform(AllLatVec_T)

    p_pca_T = p_pca.T

    total_mat = M_combined.shape[1]

    m_vecs_2D = p_pca_T[:, :total_mat]
    p_vecs_2D = p_pca_T[:, total_mat:]

    # Split original vs designed
    m_orig_2D = m_vecs_2D[:, :Inmat]
    m_new_2D  = m_vecs_2D[:, Inmat:]

    # ----------------------------
    # Step 3: Plot
    # ----------------------------
    os.makedirs(f'5_FFig/{irun}/InverseDesign/{Ioffval}', exist_ok=True)

    plt.figure(figsize=(11,7))

    # Axes
    plt.axhline(0, color='black', linewidth=2)
    plt.axvline(0, color='black', linewidth=2)

    # Original materials
    plt.scatter(m_orig_2D.T[:,0], m_orig_2D.T[:,1],
                s=30, edgecolor='black', facecolor='none',
                alpha=0.6, label="Original Materials")

    # Designed candidates
    plt.scatter(m_new_2D.T[:,0], m_new_2D.T[:,1],
                s=70, c='red', marker='x',
                label="Designed Candidates")

    # ----------------------------
    # Pressure vectors (same logic as yours)
    # ----------------------------
    colors = plt.cm.get_cmap('hsv', Inprop)

    norm_p = np.linalg.norm(p_vecs_2D.T, axis=1)
    p_normed = p_vecs_2D.T / norm_p[:, None]

    for i in range(Inprop):

        scale = 0.3

        dx = p_normed[i,0] * scale
        dy = p_normed[i,1] * scale

        # Highlight target pressure
        if i == ip:
            plt.quiver(0, 0, dx, dy,
                       color='black', linewidth=2,
                       label="Target Pressure")
        else:
            plt.quiver(0, 0, dx, dy,
                       color=colors(i), alpha=0.5)

    # Labels
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')

    plt.title(f"Inverse Design for Pressure Index {ip}")

    plt.legend()

    plt.savefig(f"5_FFig/{irun}/InverseDesign/{Ioffval}/Pressure_{ip}.pdf")
    plt.close()

def PlotInverseDesignSingle(IPAct, IMAct, candidates, ip, Ibesthp, Inmat, Inprop, Icname, irun, Ioffval):

    """
    Description:
        Plot inverse-designed materials (for a single pressure) in PCA space.

    Inputs:
        IPAct       ==> (k x Inprop)
        IMAct       ==> (k x Inmat)
        candidates  ==> (n_candidates x k)
        ip          ==> pressure index used for design
    """

    # ----------------------------
    # Step 1: Prepare matrices
    # ----------------------------
    candidates_T = candidates.T  # (k x n_candidates)

    # Combine original + designed
    M_combined = np.hstack((IMAct, candidates_T))  # (k x Inmat + n_candidates)

    # Stack with pressures (same as your PCAAfter)
    AllLatVec = np.hstack((M_combined, IPAct))  # (k x total)

    # ----------------------------
    # Step 2: PCA
    # ----------------------------
    AllLatVec_T = AllLatVec.T

    pca = PCA(n_components=2)
    p_pca = pca.fit_transform(AllLatVec_T)

    p_pca_T = p_pca.T

    total_mat = M_combined.shape[1]

    m_vecs_2D = p_pca_T[:, :total_mat]
    p_vecs_2D = p_pca_T[:, total_mat:]

    # Split original vs designed
    m_orig_2D = m_vecs_2D[:, :Inmat]
    m_new_2D  = m_vecs_2D[:, Inmat:]

    # ----------------------------
    # Step 3: Plot
    # ----------------------------
    os.makedirs(f'5_FFig/{irun}/InverseDesign_Single/{Ioffval}', exist_ok=True)

    plt.figure(figsize=(11,7))

    # Axes
    plt.axhline(0, color='black', linewidth=2)
    plt.axvline(0, color='black', linewidth=2)

    # Original materials
    plt.scatter(m_orig_2D.T[:,0], m_orig_2D.T[:,1],
                s=30, edgecolor='black', facecolor='none',
                alpha=0.6, label="Original Materials")

    # Designed candidates
    plt.scatter(m_new_2D.T[:,0], m_new_2D.T[:,1],
                s=70, c='red', marker='x',
                label="Designed Candidates")

    # ----------------------------
    # Pressure vectors (same logic as yours)
    # ----------------------------
    colors = plt.cm.get_cmap('hsv', Inprop)

    norm_p = np.linalg.norm(p_vecs_2D.T, axis=1)
    p_normed = p_vecs_2D.T / norm_p[:, None]

    for i in range(Inprop):

        scale = 0.3

        dx = p_normed[i,0] * scale
        dy = p_normed[i,1] * scale

        # Highlight target pressure
        if i == ip:
            plt.quiver(0, 0, dx, dy,
                       color='black', linewidth=2,
                       label="Target Pressure")
        else:
            plt.quiver(0, 0, dx, dy,
                       color=colors(i), alpha=0.5)

    # Labels
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')

    plt.title(f"Inverse Design for Pressure Index {ip}")

    plt.legend()

    plt.savefig(f"5_FFig/{irun}/InverseDesign_Single/{Ioffval}/Pressure_{ip}.pdf")
    plt.close()

def PlotResult(IIIMat, IIInprop, IIInmat, IIIfrac, IIIirun, IIbesthp, \
    IIP, IIM, IIP_l, IIM_l, IIP_b, IIM_b, ITrue_Test, IPred_Test, IPred_l_Test, IPred_b_Test, \
    ITrue_Train, IPred_Train, IPred_l_Train, IPred_b_Train, IIMMat_pred, \
    IIMMat_l_pred, IIMMat_b_pred, IIimat, \
    IIidx_unknown_frac, IIidx_known_frac,\
    IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, \
    IIMMat_stan_mean, IIMMat_stan_std, IIIoffval):
    
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
        #ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, Iaxis='SEMILOG') # Normalized plots
        ##ETime = PlotIsothermParity(IIimat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, Iaxis='UN_SEMILOG') # Unnormalized plots. This may be erroneous. Use with caution
        #ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, Iaxis='UN_SEMILOG') # Unnormalized plots. This may be erroneous. Use with caution
        #print("\n================================================================================")
        #print("\033[92m[DEBUG XX:] PlotIsothermParity() ran successfully for bias-only case for SEMILOG case\033[0m")
        #print(f"Time taken for ploting isotherm parity plots for bias-only case for SEMILOG case is: {ETime:.4f} minutes")
        #print("================================================================================\n")

        #ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, Iaxis='LINEAR') # Normalized plots
        ##ETime = PlotIsothermParity(IIimat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution
        #ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution

        #ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=True, Iaxis='LINEAR') # Normalized plots
        
        #ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=True, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution

        #ETime = PlotIsothermParity(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=False, Iaxis='LINEAR') # Normalized plots

        #ETime = PlotIsothermParity(IIIMat, IIUNMMat_pred, IIUNMMat_l_pred, IIUNMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac, full=False, Iaxis='UN_LINEAR') # Unnormalized plots. This may be erroneous. Use with caution

        print("\n================================================================================")
        print("\033[92m[DEBUG XX:] PlotIsothermParity() ran successfully for latent-only case for LINEAR case\033[0m")
        print(f"Time taken for ploting isotherm parity plots for latent-only case for LINEAR case is: {ETime:.4f} minutes")
        print("================================================================================\n")

    #------------------------------------------------------------------------------------------#

    ETime = PlotBar(IIimat, IIMMat_pred, IIMMat_l_pred, IIMMat_b_pred, IIInprop, IIInmat, IIIfrac, IIIirun, IIidx_unknown_frac, IIidx_known_frac) # Testing
    #raise ValueError(f"DEBUGGING GOING !!!!")

    ETime = PlotBiasParity(IIM, IIM_b, IIIfrac, IIbesthp, IIInmat, IIIirun)    
    #raise ValueError(f"DEBUGGING GOING !!!!")

    RMAct, RPAct, RAllLatVec, Rm_vecs_2D, Rp_vecs_2D = PCAAfter(IIM, IIP, IIbesthp, IIInmat, IIInprop, True)
    RMAct_l, RPAct_l, RAllLatVec_l, Rm_vecs_2D_l, Rp_vecs_2D_l = PCAAfter(IIM_l, IIP_l, IIbesthp, IIInmat, IIInprop, False)
    
    PlotPCAAfter(Rp_vecs_2D, Rm_vecs_2D, IIIMat, IIInmat, IIInprop, IIIfrac, IIIirun, True) # Latent-bias case
    PlotMatRanking(Rp_vecs_2D, Rm_vecs_2D, IIIMat, IIInmat, IIInprop, IIIfrac, IIIirun, True)
    CosSim = PlotCosineSimilarity(RPAct, IIbesthp, IIInprop, IIIfrac, IIIirun, True)

    mat_ids = [406, 409, 670, 770, 838, 839, 875, 938, 1070, 1429, 1580, 2037, 2229, 2433, 2589, 2693]
    PlotLatGeoIsotherms(RMAct, RPAct, IIimat, IIInmat, IIInprop, IIbesthp, mat_ids, IIMMat_stan_mean, IIMMat_stan_std, IIIirun, True)
    #hmofs, huptakes = InverseDesign(RPAct, RMAct, 15, IIInprop, IIbesthp, 10, IIIirun, True)
    #hmofs_pur, huptakes_pur = InverseDesign_Perturbation(RPAct, RMAct, 15, IIInprop, IIInmat, IIimat, IIbesthp, 10, IIIirun, 1, True)
    ##PlotInverseDesignSingle(RPAct, RMAct, candidates, 15, IIbesthp, IIInmat, IIInprop, pressure, IIIirun, True)

    PlotPCAAfter(Rp_vecs_2D_l, Rm_vecs_2D_l, IIIMat, IIInmat, IIInprop, IIIfrac, IIIirun, False) # latent only case
    PlotMatRanking(Rp_vecs_2D_l, Rm_vecs_2D_l, IIIMat, IIInmat, IIInprop, IIIfrac, IIIirun, False)
    CosSim_l = PlotCosineSimilarity(RPAct_l, IIbesthp, IIInprop, IIIfrac, IIIirun, False)

    #mat_ids = [406, 409, 670, 770, 838, 839, 875, 938, 1070, 1429, 1580, 2037, 2229, 2433, 2589, 2693]
    PlotLatGeoIsotherms(RMAct_l, RPAct_l, IIimat, IIInmat, IIInprop, IIbesthp, mat_ids, IIMMat_stan_mean, IIMMat_stan_std, IIIirun, False)
    #hmofs_l, huptakes_l = InverseDesign(RPAct_l, RMAct_l, 15, IIInprop, IIbesthp, 10, IIIirun, False)
    #hmofs_pur_l, huptakes_pur_l = InverseDesign_Perturbation(RPAct_l, RMAct_l, 15, IIInprop, IIInmat, IIimat, IIbesthp, 10, IIIirun, 1, False)
    ##PlotInverseDesignSingle(RPAct_l, RMAct_l, candidates_l, 15, IIbesthp, IIInmat, IIInprop, pressure, IIIirun, False)

    #raise ValueError(f"DEBUGGING GOING !!!!")

    return RMAct, RPAct, RAllLatVec, Rm_vecs_2D, Rp_vecs_2D

def SaveResult(IIMat, IIfrac, IInprop, IInmat, Iirun, \
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

    if args.existing:
        True_Test    = [Amp_true_T_stan[p, m] for (p, m) in Iidx_unknown_frac]
    else:
        True_Test    = [Iimat[p, m] for (p, m) in Iidx_unknown_frac]        # True normalized values present in test set

    #True_Test    = [Iimat[p, m] for (p, m) in Iidx_unknown_frac]         # True normalized values present in test set
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

    MAct, PAct, AllLatVec, m_vecs_2D, p_vecs_2D = PlotResult(IIMat, IInprop, \
        IInmat, IIfrac, Iirun, Ibesthp, IP, IM, IP_l, IM_l, IP_b, IM_b, True_Test, Pred_Test, \
        Pred_l_Test, Pred_b_Test, True_Train, Pred_Train, Pred_l_Train, \
        Pred_b_Train, IMMat_pred, IMMat_l_pred, IMMat_b_pred, Iimat, \
        Iidx_unknown_frac, Iidx_known_frac, IUNMMat_pred, IUNMMat_l_pred, IUNMMat_b_pred, IMMat_stan_mean, IMMat_stan_std, IIoffval)

    Result = {
    "RIfrac": float(IIfrac),                      # Fraction of known entries
    "RInprop": int(IInprop),                    # # of Pressure points
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

def RUN(Ifrac, IMat, Inprop, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat, irun):
    
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
                
                Inprop     ==>     Same as before 
                
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
    if Isample == "SUBROW":
        [MMat, idx_known_frac, idx_unknown_frac] = PropMatMissing(Ifrac, IMat, Inmat, Inprop, Isample, None, None, None, None,Irow_sub, None)
    elif Isample == "SUBCOL":
        [MMat, idx_known_frac, idx_unknown_frac] = PropMatMissing(Ifrac, IMat, Inmat, Inprop, Isample, None, None, None, None, None, Icol_sub)
    elif args.existing:
        [MMat, idx_known_frac, idx_unknown_frac, Ifrac] = PropMatExisting(IMat, Inmat, Inprop)
    else: 
        [MMat, idx_known_frac, idx_unknown_frac] = PropMatMissing(Ifrac, IMat, Inmat, Inprop, Isample, None, None, None, None, None, None)

    # Step 2: Standardizing the missing Pressure material matrix
    MMat_stan_mean, MMat_stan_std, MMat_stan = standardize_matrix(MMat, 1)

    if args.existing:
        global Amp_true_T_stan
        Amp_true_T_stan = (Amp_true_T - MMat_stan_mean) / MMat_stan_std

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

    besthp, RMSD_val_hpg, RMSD_train_hpg, R2_val_hpg, R2_train_hpg, plot_RMSD, plot_R2 = HPTune(MMat_stan, Inprop, Inmat, Ihpg, idx_train_frac, idx_val_frac, Ifrac, True, irun)

    # Step 7: Train the final latent-bias model on train + validation set datapoint and check its performance on test set
    
    print("\n======================================================================================")
    print(f"\n\n\tTraining final latent-bias model on new training set for Ifrac = {Ifrac}\n\n")
    print("======================================================================================\n")
    
    P, M, model, ch = fit_lrm(MMat_stan, Inprop, Inmat, besthp, idx_known_frac, None, None, True)

    # Step 8: Use the same best hyperparamter as that of latent-bias case and  
    # train the final latent-only model on train + validation set datapoint and 
    #check its performance on test set

    print("\n======================================================================================")
    print(f"\n\n\tTraining final latent-only model on new training set using the best hyperparameters obtained for latent-bias case for Ifrac = {Ifrac}\n\n")
    print("======================================================================================\n")

    P_l, M_l, model_l, ch_l = fit_lrm(MMat_stan, Inprop, Inmat, besthp, idx_known_frac, None, None, False)

    hpara_bias = {"dim": 0, "reg": 0.00}

    print("\n======================================================================================")
    print(f"\n\n\tTraining bias only model on new training set for Ifrac = {Ifrac}\n\n")
    print("======================================================================================\n")

    P_b, M_b, model_b, ch_b = fit_lrm(MMat_stan, Inprop, Inmat, hpara_bias, idx_known_frac, None, None, True)

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

    assert np.allclose(Iori_mat, imat, atol = 1e-6, equal_nan = True),\
            f"\033[91mMismatch in the original standardized matrix\033[0m"

    UNIMat = unstandardize_matrix(imat, imean, istd)
    #MMat_stan_mean, MMat_stan_std, MMat_stan = standardize_matrix(MMat, 1)

    assert np.allclose(IMat, UNIMat, atol = 1e-6, equal_nan = True),\
            f"\033[91mMismatch in the reconstructed unnormalized matrix\033[0m"

    UNMMat_pred   = unstandardize_matrix(MMat_pred, MMat_stan_mean, MMat_stan_std)
    UNMMat_l_pred = unstandardize_matrix(MMat_l_pred, MMat_stan_mean, MMat_stan_std)
    UNMMat_b_pred = unstandardize_matrix(MMat_b_pred, MMat_stan_mean, MMat_stan_std)

    Result = SaveResult(IMat, Ifrac, Inprop, Inmat, irun, \
        idx_unknown_frac, idx_known_frac, besthp, \
        P, M, P_l, M_l, P_b, M_b, \
        MMat_pred, MMat_l_pred, MMat_b_pred, \
        MMat_stan_mean, MMat_stan_std, MMat_stan, \
        UNMMat_pred, UNMMat_l_pred, UNMMat_b_pred, \
        imat, plot_RMSD, plot_R2, Ioffval)

    return Result

def RUNM(Ifrac, IMat, Inprop, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat, nrun):
    
    """

    Description:
    
            Aim is to run RUNM multiple times for the same Ifrac to account 
            for randomness and save all results in a single pickle.
    
    Inputs:

        Ifrac, IMat, Inprop, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat : as in RUN()
        
        nrun : Number of independent runs to perform
    
    Output:

        Saves a single pickle file containing a list of dictionaries, one per run
    """
    
    AllResults = []

    for irun in range(nrun):
        print(f"\n===== Run {irun + 1}/{nrun} for Ifrac = {Ifrac} =====")
        
        # Call your original RUN function
        iresult = RUN(Ifrac, IMat, Inprop, Inmat, Ihpg, Ioffval, Isample, Iori_mean, Iori_std, Iori_mat, irun)
        
        AllResults.append(iresult)
        
        # Ensure folder exists
        os.makedirs(f'5_FFig/Statistical_analysis', exist_ok=True)
    
        # Save all runs in a single pickle
        with open(f"5_FFig/Statistical_analysis/{nrun}_result.pkl", "wb") as f:
            pickle.dump(AllResults, f)
    
        print(f"\n\033[92m The {irun + 1} run for Ifrac = {Ifrac} saved successfully.\033[0m")
    
    print(f"\n\033[92m All {nrun} runs for Ifrac = {Ifrac} saved successfully.\033[0m")

##--------------------------------------------------------------------------------------------##

# -----------------------------
# Command-line arguments
# -----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True, help="Input CSV file")

parser.add_argument("--sample", required=False, choices=["RANDOM", "ROW", "COL", "SUBROW", "SUBCOL"], help="Sampling strategy for PropMatMissing")

parser.add_argument("--row-sub", nargs=2, type=int, default=None, help="Start and end indices for SUBROW sampling")

parser.add_argument("--col-sub", nargs=2, type=int, default=None, help="Start and end indices for SUBCOL sampling")

parser.add_argument("--existing", action="store_true", help="Use the original missing-data pattern in the input matrix")

parser.add_argument("--true-input", required=False, help="Complete reference matrix used for evaluating the existing missing-data pattern")

args = parser.parse_args()

fname       = args.input
sample      = args.sample
Irow_sub    = args.row_sub
Icol_sub    = args.col_sub
true_input  = args.true_input

if Irow_sub is not None:
    Irow_sub = list(range(Irow_sub[0], Irow_sub[1]))

if Icol_sub is not None:
    Icol_sub = list(range(Icol_sub[0], Icol_sub[1]))

if args.existing and sample is not None:
    raise ValueError("\033[91m--existing and --sample cannot be used together.\033[0m")

if not args.existing and sample is None:
    raise ValueError("\033[91mEither --sample or --existing must be provided.\033[0m")

if args.existing and args.true_input is None:
    raise ValueError("\033[91m--true-input must be provided when --existing is used.\033[0m")

if not args.existing and args.true_input is not None:
    raise ValueError("\033[91m--true-input can only be used with --existing.\033[0m")

# ------------------------------------
# Extract information from filename
# -------------------------------------

DBName = fname.removesuffix(".csv").split("-")[0]

# Reading the data file and initial checks

#.csv file contains ==> mof_name,loading1,loading2,....

#df_data = pd.read_csv("Isotherm.csv")

df_data = pd.read_csv(fname)

if args.existing:

    df_true = pd.read_csv(true_input)

    property_data = df_data.drop(columns="mof_name")

    empty_row = property_data.isnull().all(axis=1)
    empty_col = property_data.isnull().all(axis=0)

    n_empty_rows = empty_row.sum()
    n_empty_cols = empty_col.sum()

    print("\n======================================================================================")
    print("\033[93m[EXISTING] Checking original missing-data pattern.\033[0m")
    print("======================================================================================\n")

    print(f"\033[93m[EXISTING] Completely empty material rows       : {n_empty_rows}\033[0m")
    print(f"\033[93m[EXISTING] Completely empty property columns    : {n_empty_cols}\033[0m")
    print("======================================================================================\n")

    # Remove completely empty material rows from MOFXDB
    if n_empty_rows > 0:
        df_data = df_data.loc[~empty_row].copy()

    # Remove completely empty property columns
    if n_empty_cols > 0:
        df_data = df_data.drop(columns=property_data.columns[empty_col])
        df_true = df_true.drop(columns=property_data.columns[empty_col])

    # -------------------------------------------------------------------------
    # Keep only MOFs common to MOFXDB and the reference/RASPA dataset
    # -------------------------------------------------------------------------

    common_mofs = set(df_data["mof_name"]).intersection(df_true["mof_name"])

    print(f"\033[93m[EXISTING] MOFs remaining in MOFXDB                 : {len(df_data)}\033[0m")
    print(f"\033[93m[EXISTING] MOFs available in reference dataset      : {len(df_true)}\033[0m")
    print(f"\033[93m[EXISTING] Common MOFs used for LRM analysis        : {len(common_mofs)}\033[0m")

    # Keep only common MOFs in MOFXDB
    df_data = df_data[df_data["mof_name"].isin(common_mofs)].copy()

    # Keep only common MOFs in reference matrix
    df_true = df_true[df_true["mof_name"].isin(common_mofs)].copy()

    # -------------------------------------------------------------------------
    # Reorder reference matrix to exactly match MOFXDB ordering
    # -------------------------------------------------------------------------

    df_true = df_true.set_index("mof_name").loc[df_data["mof_name"]].reset_index()

    assert df_data.shape == df_true.shape, f"Mismatch between original and reference matrices: {df_data.shape} vs {df_true.shape}"
    assert df_data["mof_name"].tolist() == df_true["mof_name"].tolist(), f"MOF ordering differs between original and reference matrices."
    
    del property_data

mof_name = df_data["mof_name"].tolist()

# All columns except mof_name
cname, property_label = GetPropertyLabels(df_data)
TestGetPropertyLabels()

df_data = df_data[cname] 

[nmat, nprop] = df_data.shape

assert [len(mof_name), len(cname)] == [nmat, nprop], \
        f"\033[91mMismatch in df_data.shape: expected({len(mof_name)}, {len(cname)}), got ({nmat}, {nprop})\033[0m" 

if args.existing:

    df_true = df_true[cname]
    Amp_true = df_true.to_numpy()
    Amp_true_T = Amp_true.T
    assert Amp_true_T.shape == (nprop, nmat), f"\033[91mMismatch in Amp_true_T.shape: expected ({nprop}, {nmat}), got {Amp_true_T.shape}\033[0m"
    
    del df_true, Amp_true
    
# Checking to see if there are any empty rows in 
empty_row_check = df_data.isnull().all(axis=1)
assert not empty_row_check.any(), \
    f"Assertion Failed: Found {empty_row_check.sum()} completely empty row(s) at indices: {df_data.index[empty_row_check].tolist()}"

# Checking to see if there are any empty columns in
empty_col_check = df_data.isnull().all(axis=0)
assert not empty_col_check.any(), \
    f"Assertion Failed: Found completely empty column(s): {df_data.columns[empty_col_check].tolist()}"

if sample == "SUBROW":

    if Irow_sub is None:
        raise ValueError("\033[91m--row-sub must be provided when sample is SUBROW.\033[0m")

    if not all(0 <= r < nprop for r in Irow_sub):
        raise ValueError(f"\033[91mSUBROW indices must be between 0 and {nprop-1}.\033[0m")

elif sample == "SUBCOL":

    if Icol_sub is None:
        raise ValueError("\033[91m--col-sub must be provided when sample is SUBCOL.\033[0m")

    if not all(0 <= c < nmat for c in Icol_sub):
        raise ValueError(f"\033[91mSUBCOL indices must be between 0 and {nmat-1}.\033[0m")

else:

    if Irow_sub is not None or Icol_sub is not None:
        raise ValueError(f"\033[91m--row-sub/--col-sub should only be provided for SUBROW/SUBCOL sampling.\033[0m")

print("\n======================================================================================")
print("\033[92m[DEBUG 1:] df_data.shape Test passed and no completely empty rows or columns detected..\033[0m")
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
#PearsonPlot(df_data.select_dtypes(include='number'), cname)
#PearsonPlot(df_data, property_label)
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
#PlotPCABefore(df_data.select_dtypes(include='number'), cname)
#PlotPCABefore(df_data)
end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print("\033[92m[DEBUG 05:] PlotPCABefore ran successfully.\033[0m")
print(f"\tTime taken to run PlotPCABefore is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

#raise ValueError(f"DEBUGGING GOING !!!!")

##--------------------------------------------------------------------------------------------##

# Creating material-Pressure matrix
Amp = df_data.to_numpy() # shape: (nmat,nprop)

assert Amp.shape == (nmat, nprop), \
        f"\033[91mMismatch in Amp.shape: expected ({nmat}, {nprop}), got {Amp.shape}\033[0m"

# Taking transpose of material-pressure matrix, becasue bias can only be added as columns in LowRanksModel()

# Since Julia implimentation of pyglrm requires the offsets to be added to the columns hence, we will be working with the transpose of the material Pressure matrix (Amp) which is Pressure-material matrix (Amp_T)

# Remember when we were trying to do PCA we learned that we have to remove mean from the columns of original matrix. Here the original matrix is Amp so we must remove mean from columns of Amp i.e. we must remove mean from rows of Amp.T i.e. Amp_T

Amp_T = Amp.T # shape: (nprop,nmat)

assert Amp_T.shape == (nprop, nmat), \
        f"\033[91mMismatch in Amp_T.shape: expected ({nprop}, {nmat}), got {Amp_T.shape}\033[0m"
print("\n======================================================================================")
print(f"\033[92m[DEBUG 06:] Material Pressure matrix (Amp) with size {Amp.shape} and Pressure material matrix (Amp_T) with size {Amp_T.shape} successfully created.\033[0m")
print("======================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Standardize the given matrix. This case handles the case where elements of matrix are missing or NaN

start_time = time.time()

Amp_mean, Amp_std, Amp_stan       = standardize_matrix(Amp, 0)
Amp_T_mean, Amp_T_std, Amp_T_stan = standardize_matrix(Amp_T, 1)

assert np.allclose(Amp_stan.T, Amp_T_stan, atol = 1e-6, equal_nan = True),\
        f"\033[91mMismatch in Amp_stan.T and Amp_T_stan\033[0m"

end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print("\n======================================================================================")
print(f"\033[92m[DEBUG 07:] Standardization of Amp and Amp_T is complete.\033[0m")
print(f"\tTime taken to run Standardization of Amp and Amp_T is : {elapsed_time:.4f} minutes")
print("======================================================================================\n")

#raise ValueError(f"DEBUGGING GOING !!!!")

###############################################################################################

# --------------------------------------------------------------------------------------------
# Generate incomplete pressure-material matrix
# --------------------------------------------------------------------------------------------

if args.existing:

    # Use the missing-data pattern already present in the input CSV
    Amp_T_incomplete, idx_known, idx_unknown, Ifrac = PropMatExisting(Amp_T, nmat, nprop)

    print("\n======================================================================================")
    print("\033[92m[DEBUG 08: EXISTING] Original missing-data pattern was retained successfully.\033[0m")
    print(f"\033[92m[DEBUG 08: EXISTING] Actual known fraction = {Ifrac:.4f}\033[0m")
    print("======================================================================================\n")

else:

    if sample == "RANDOM":
        
        [Amp_T_incomplete, idx_known, idx_unknown] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, None, None, None, None)
        [Amp_T_incomplete_1, idx_known_1, idx_unknown_1] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, None, None, None, None)

    elif sample == "ROW":

        [Amp_T_incomplete, idx_known, idx_unknown] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, 0.3, 5, None, None)
        [Amp_T_incomplete_1, idx_known, idx_unknown_1] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, 0.3, 5, None, None)

    elif sample == "COL":

        [Amp_T_incomplete, idx_known, idx_unknown] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, 0.3, 5, None, None, None, None)
        [Amp_T_incomplete_1, idx_known_1, idx_unknown_1] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, 0.3, 5, None, None, None, None)

    elif sample == "SUBROW":

        #Irow_sub = list(range(0,4))     # For Ar Low Pressure
        #Irow_sub = list(range(4,8))     # For Ar Medium Pressure
        #Irow_sub = list(range(8,nprop)) # For Ar High Pressure

        [Amp_T_incomplete, idx_known, idx_unknown] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, None, None, Irow_sub, None)
        [Amp_T_incomplete_1, idx_known_1, idx_unknown_1] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, None, None, Irow_sub, None)

    elif sample == "SUBCOL":

        #Icol_sub = list(range(0,4)) # 1st to 3rd material only
        #Icol_sub = list(range(8,nmat)) # all the materials
        #Icol_sub = list(range(0,nmat-1)) # all the materials except the last one

        [Amp_T_incomplete, idx_known, idx_unknown] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, None, None, None, Icol_sub)
        [Amp_T_incomplete_1, idx_known_1, idx_unknown_1] = PropMatMissing(0.4, Amp_T, nmat, nprop, sample, None, None, None, None, None, Icol_sub)

    else:

        raise ValueError(f"\033[91mUnknown sampling type: {sample}\033[0m")

    # -------------------------------------------------------------------------
    # Verifying that two artificial sampling calls produce different patterns
    # -------------------------------------------------------------------------

    mask  = np.isnan(Amp_T_incomplete)
    mask1 = np.isnan(Amp_T_incomplete_1) 

    assert not np.array_equal(mask, mask1),\
            f"\033[91mSampling pattern did not change.\033[0m"

    print("\n======================================================================================")
    print(f"\033[92m[DEBUG 08: {sample}] PropMatMissing ran for {sample} case and incomplete Pressure-material matrix was generated successfully.\033[0m")
    print("======================================================================================\n")

    np.savetxt("Amp_T_incomplete.csv", Amp_T_incomplete, delimiter=",")
    np.savetxt("Amp_T_incomplete_1.csv", Amp_T_incomplete_1, delimiter=",")

    del Amp_T_incomplete_1, idx_known_1, idx_unknown_1
    del mask, mask1

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

GLRMTest(False, Amp_T, nmat, nprop, sample)
GLRMTest(True, Amp_T, nmat, nprop, sample)

##--------------------------------------------------------------------------------------------##

# Creating hyper-parameter grid

# Generate the reg values: 10^1.0 to 10^3.0 in 25 log-spaced steps
reg = np.logspace(1.0, 3.0, num = 25)  # 10^1 to 10^3 num = 25
reg = np.round(reg).astype(int)
dim = range(1, nprop)  # 1 to nprop - 1 range(1, nprop)

# Creating hyper-parameter grid
hpg = []

for idim in dim:
    for ireg in reg:
        hpg.append({"dim": idim, "reg": ireg})

for ihpg in hpg:
    print(f"Dimention: {ihpg['dim']}, Regularization: {ihpg['reg']}")

print(f"Printing specific element of hpg : {hpg[0]['dim']} and {hpg[0]['reg']}")

assert len(hpg) == (nprop - 1) * len(reg), \
        f"\033[91mMistmatch in hyperparameter grid size. Expected: {(nprop - 1) * len(reg)}. Got : {len(hpg)}\033[0m"

print("\n================================================================================")
print("\033[92m [DEBUG 11:] Hyper-parameter grid constructed sucessfullly!!!\033[0m")
print("=================================================================================\n")

print("\n================================================================================")
print(f"\033[92m[DEBUG 12:] RUNM() running for Ifrac = 0.99 !!!\033[0m")
print("=================================================================================\n")

RUNM(0.99, Amp_T, nprop, nmat, hpg, offset, sample, Amp_T_mean, Amp_T_std, Amp_T_stan, 2)

print("\n================================================================================")
print("\033[92m[DEBUG 8:] RUNM() ran successfully for Ifrac = 0.99 !!!\033[0m")
print("=================================================================================\n")

##--------------------------------------------------------------------------------------------##

# Deplyoable model is created
print(f"DONE")





