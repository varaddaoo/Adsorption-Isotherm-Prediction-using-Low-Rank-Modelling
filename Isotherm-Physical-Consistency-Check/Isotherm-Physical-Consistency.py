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

def ReadAdsorptionDatabase(IfAdsorption):
    
    """
    
    Description:

            Reads the adsorption database from a csv file.
            The function performs basic sanity checks on the
            database, verifies that the MOF column is present,
            removes duplicate MOFs (if any), and sets the MOF
            names as the DataFrame index.

            The processed pandas DataFrame is returned for
            subsequent quality-control analyses.

    Input:

            IfAdsorption ==> str
                        Name of the csv file containing the
                        adsorption database.

    Output:

            RdfAdsorption ==> pandas.DataFrame
                        Adsorption database indexed by MOF
                        names.

    """

    print(f"\n======================================================\n")

    #######################################################
    # Read adsorption database
    #######################################################

    RdfAdsorption = pd.read_csv(IfAdsorption)

    #######################################################
    # Initial checks
    #######################################################

    assert os.path.isfile(IfAdsorption), \
        f"\033[91mFile {IfAdsorption} not found.\033[0m"

    assert len(RdfAdsorption) > 0, \
        f"\033[91mAdsorption database is empty.\033[0m"

    assert "MOF" in RdfAdsorption.columns, \
        f"\033[91mColumn named MOF was not found.\033[0m"

    #######################################################
    # Duplicate MOFs
    #######################################################

    NDuplicates = RdfAdsorption.duplicated("MOF").sum()

    print(f"\nNumber of MOFs              = {len(RdfAdsorption)}")

    print(f"\nDuplicate MOFs             = {NDuplicates}")

    #######################################################
    # Remove duplicate MOFs
    #######################################################

    RdfAdsorption = RdfAdsorption.drop_duplicates(
        subset="MOF",
        keep="first"
    )

    #######################################################
    # Set MOF as index
    #######################################################

    RdfAdsorption = RdfAdsorption.set_index("MOF")

    RdfAdsorption = RdfAdsorption.sort_index()

    #######################################################
    # Pressure columns
    #######################################################

    RPressureColumns = list(RdfAdsorption.columns)

    #######################################################
    # Final checks
    #######################################################

    assert RdfAdsorption.index.is_unique, \
        f"\033[91mDuplicate MOFs still exist after cleanup.\033[0m"

    assert RdfAdsorption.shape[1] > 0, \
        f"\033[91mNo pressure columns found.\033[0m"

    #######################################################
    # Print summary
    #######################################################

    print(f"\nNumber of unique MOFs       = {len(RdfAdsorption)}")

    print(f"\nNumber of pressure points   = {RdfAdsorption.shape[1]}")

    print(f"\n\033[92m[DEBUG 01:] ReadAdsorptionDatabase() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return RdfAdsorption, RPressureColumns

def MissingValueAnalysis(Idf_Adsorption):
    
    """
    
    Description:

            Computes the number of missing adsorption
            uptakes for every MOF together with the
            percentage completeness of each adsorption
            isotherm.

            The results are written to the file

                    MissingValueAnalysis.csv

            and summary statistics are printed.

    Input:

            Idf_Adsorption ==> pandas.DataFrame

                        Adsorption database with
                        MOFs as index and pressure
                        points as columns.

    Output:

            RMissingValueAnalysis ==> pandas.DataFrame

                        DataFrame containing the
                        number of missing values
                        and completeness of every
                        adsorption isotherm.

    """

    print(f"\n======================================================\n")

    #######################################################
    # Initial checks
    #######################################################

    assert isinstance(Idf_Adsorption, pd.DataFrame), \
        f"\033[91mInput must be a pandas DataFrame.\033[0m"

    assert len(Idf_Adsorption) > 0, \
        f"\033[91mEmpty adsorption database.\033[0m"

    #######################################################
    # Missing value analysis
    #######################################################

    NPressure = len(Idf_Adsorption.columns)

    MissingValueAnalysis = []

    for iMOF in Idf_Adsorption.index:

        Uptake = Idf_Adsorption.loc[iMOF].to_numpy(dtype=float)

        NMissing = np.isnan(Uptake).sum()

        NAvailable = np.sum(~np.isnan(Uptake))

        Completeness = 100.0 * NAvailable / NPressure

        MissingValueAnalysis.append([
            iMOF,
            NAvailable,
            NMissing,
            Completeness
        ])

    #######################################################
    # Convert to DataFrame
    #######################################################

    RMissingValueAnalysis = pd.DataFrame(
        MissingValueAnalysis,
        columns=[
            "MOF",
            "NAvailable",
            "NMissing",
            "Completeness (%)"
        ]
    )

    #######################################################
    # Final checks
    #######################################################

    assert len(RMissingValueAnalysis) == len(Idf_Adsorption), \
        f"\033[91mUnexpected number of MOFs.\033[0m"

    #######################################################
    # Save results
    #######################################################

    RMissingValueAnalysis.to_csv(
        "MissingValueAnalysis.csv",
        index=False
    )

    #######################################################
    # Print summary
    #######################################################

    print("\nMissing value analysis written to MissingValueAnalysis.csv")

    print(f"\nTotal MOFs = {len(RMissingValueAnalysis)}")

    print(f"\nAverage completeness = {RMissingValueAnalysis['Completeness (%)'].mean():.2f}%")

    print(f"\nCompletely available isotherms = "
          f"{np.sum(RMissingValueAnalysis['NMissing']==0)}")

    print(f"\nMOFs containing missing values = "
          f"{np.sum(RMissingValueAnalysis['NMissing']>0)}")

    print(f"\nWorst five MOFs:\n")

    print(
        RMissingValueAnalysis
        .sort_values("NMissing", ascending=False)
        .head(5)
    )

    print(f"\n\033[92m[DEBUG 01:] MissingValueAnalysis() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return RMissingValueAnalysis

def MonotonicityAnalysis(Idf_Adsorption, ITolerance=1.0e-6):
    
    """

    Description:

            Checks whether every adsorption isotherm is
            monotonically increasing with pressure.

            For every MOF, the adsorption uptake difference
            between two consecutive pressure points is
            computed. A negative difference larger than the
            specified numerical tolerance is considered a
            violation of monotonicity.

            For every non-monotonic isotherm, the largest
            adsorption drop together with the corresponding
            pressure interval is reported.

            The results are written to

                    MonotonicityAnalysis.csv

    Input:

            Idf_Adsorption ==> pandas.DataFrame

                        Adsorption database with
                        MOFs as index and pressure
                        points as columns.

            ITolerance ==> float

                        Numerical tolerance used
                        while checking monotonicity.

    Output:

            RMonotonicityAnalysis ==> pandas.DataFrame

                        DataFrame containing the
                        monotonicity statistics of
                        every adsorption isotherm.

    """

    print(f"\n======================================================\n")

    #######################################################
    # Initial checks
    #######################################################

    assert isinstance(Idf_Adsorption, pd.DataFrame), \
        f"\033[91mInput must be a pandas DataFrame.\033[0m"

    assert len(Idf_Adsorption) > 0, \
        f"\033[91mEmpty adsorption database.\033[0m"

    assert ITolerance >= 0.0, \
        f"\033[91mTolerance must be non-negative.\033[0m"

    Pressure = np.asarray(Idf_Adsorption.columns, dtype=float)

    assert len(Pressure) == Idf_Adsorption.shape[1], \
        f"\033[91mPressure columns do not match adsorption database.\033[0m"

    #######################################################
    # Monotonicity analysis
    #######################################################

    MonotonicityAnalysis = []

    for iMOF in Idf_Adsorption.index:

        Uptake = Idf_Adsorption.loc[iMOF].to_numpy(dtype=float)

        ###################################################
        # Ignore missing values
        ###################################################

        Mask = ~np.isnan(Uptake)

        Uptake = Uptake[Mask]

        PressureValid = Pressure[Mask]

        if len(Uptake) < 2:

            NViolations = 0
            LargestDrop = 0.0
            PressureFrom = "NA"
            PressureTo = "NA"

        else:

            Difference = np.diff(Uptake)

            NViolations = np.sum(Difference < -ITolerance)

            if NViolations == 0:

                LargestDrop = 0.0
                PressureFrom = "NA"
                PressureTo = "NA"

            else:

                Index = np.argmin(Difference)

                LargestDrop = Difference[Index]

                PressureFrom = PressureValid[Index]

                PressureTo = PressureValid[Index + 1]

        MonotonicityAnalysis.append(
            [
                iMOF,
                len(Uptake),
                NViolations,
                LargestDrop,
                PressureFrom,
                PressureTo,
                (NViolations == 0)
            ]
        )

    #######################################################
    # Convert to DataFrame
    #######################################################

    RMonotonicityAnalysis = pd.DataFrame(
        MonotonicityAnalysis,
        columns=[
            "MOF",
            "NPoints",
            "NViolations",
            "LargestDrop",
            "PressureFrom",
            "PressureTo",
            "Monotonic"
        ]
    )

    #######################################################
    # Final checks
    #######################################################

    assert len(RMonotonicityAnalysis) == len(Idf_Adsorption), \
        f"\033[91mUnexpected number of MOFs.\033[0m"

    #######################################################
    # Save results
    #######################################################

    RMonotonicityAnalysis.to_csv(
        "MonotonicityAnalysis.csv",
        index=False
    )

    #######################################################
    # Print summary
    #######################################################

    NGood = np.sum(RMonotonicityAnalysis["Monotonic"])

    NBad = len(RMonotonicityAnalysis) - NGood

    print("\nMonotonicity analysis written to MonotonicityAnalysis.csv")

    print(f"\nTolerance used = {ITolerance}")

    print(f"\nTotal MOFs = {len(RMonotonicityAnalysis)}")

    print(f"\nMonotonic isotherms = {NGood}")

    print(f"\nNon-monotonic isotherms = {NBad}")

    print(f"\nWorst five MOFs:\n")

    print(
        RMonotonicityAnalysis
        .sort_values(
            by=["NViolations", "LargestDrop"],
            ascending=[False, True]
        )
        .head(5)
    )

    print(f"\n\033[92m[DEBUG 02:] MonotonicityAnalysis() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return RMonotonicityAnalysis

def PlotProblematicIsotherms(Idf_Adsorption, IMonotonicityAnalysis, NPlots=20):
    
    """
    
    Description:

            Plots the adsorption isotherms corresponding to
            the most non-monotonic MOFs identified by
            MonotonicityAnalysis().

            The MOFs are ranked according to the magnitude
            of the largest adsorption drop. The worst
            NPlots adsorption isotherms are written to

                    ./ProblematicIsotherms/

    Input:

            Idf_Adsorption ==> pandas.DataFrame

                        Adsorption database with
                        MOFs as index and pressure
                        points as columns.

            IMonotonicityAnalysis ==> pandas.DataFrame

                        Output of
                        MonotonicityAnalysis().

            NPlots ==> int

                        Number of problematic
                        adsorption isotherms to plot.

    Output:

            Problematic adsorption isotherms written to

                    ./ProblematicIsotherms/

    """

    print(f"\n======================================================\n")

    #######################################################
    # Initial checks
    #######################################################

    assert isinstance(Idf_Adsorption, pd.DataFrame), \
        f"\033[91mInput adsorption database must be a DataFrame.\033[0m"

    assert isinstance(IMonotonicityAnalysis, pd.DataFrame), \
        f"\033[91mInput monotonicity analysis must be a DataFrame.\033[0m"

    assert NPlots > 0, \
        f"\033[91mNPlots must be positive.\033[0m"

    #######################################################
    # Create output directory
    #######################################################

    os.makedirs(
        "ProblematicIsotherms",
        exist_ok=True
    )

    #######################################################
    # Select only problematic MOFs
    #######################################################

    Problematic = IMonotonicityAnalysis[
        IMonotonicityAnalysis["Monotonic"] == False
    ]

    if len(Problematic) == 0:

        print("\nNo non-monotonic isotherms found.")

        return

    #######################################################
    # Rank according to largest adsorption drop
    #######################################################

    Problematic = Problematic.sort_values(
        by="LargestDrop",
        ascending=True
    )

    #######################################################
    # Plot worst isotherms
    #######################################################

    NGenerated = 0

    Pressure = Idf_Adsorption.columns.astype(float)

    for _, Row in Problematic.head(NPlots).iterrows():

        MOF = Row["MOF"]

        Uptake = Idf_Adsorption.loc[MOF].to_numpy(dtype=float)

        Mask = ~np.isnan(Uptake)

        plt.figure(figsize=(7,5))

        plt.plot(
            Pressure[Mask],
            Uptake[Mask],
            "-o",
            lw=2,
            ms=5
        )

        plt.xscale("log")

        plt.xlabel(
            "Pressure (Pa)",
            fontdict=fontdict_x
        )

        plt.ylabel(
            "Adsorption uptake (mol/kg)",
            fontdict=fontdict_y
        )

        plt.title(
            "%s\nLargest drop = %.4f mol/kg"
            %(
                MOF,
                Row["LargestDrop"]
            ),
            fontdict=fontdict_t
        )

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            "ProblematicIsotherms/%03d_%s.png"
            %(
                NGenerated+1,
                MOF
            ),
            dpi=300
        )

        plt.close()

        NGenerated += 1

    #######################################################
    # Final checks
    #######################################################

    assert NGenerated > 0, \
        f"\033[91mNo plots were generated.\033[0m"

    #######################################################
    # Print summary
    #######################################################

    print(f"\nProblematic isotherms written to")

    print(f"\n./ProblematicIsotherms/")

    print(f"\nNumber of plots generated = {NGenerated}")

    print(f"\n\033[92m[DEBUG 03:] PlotProblematicIsotherms() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return

def RelativeDropAnalysis(Idf, IPressureColumns, ITolerance=1.0e-6):

    """

    Description:

            Computes the largest decrease in adsorption uptake
            for every MOF and expresses this decrease relative
            to the adsorption range of that MOF.

            Only valid adsorption uptakes are considered.
            MOFs exhibiting no decrease are classified as
            Monotonic. Non-monotonic MOFs are further
            classified based on the magnitude of the relative
            decrease.

            The analysis is written to

                    RelativeDropAnalysis.csv

    Input:

            Idf ==> pandas.DataFrame
                    Adsorption database indexed by MOF.

            IPressureColumns ==> list
                    Pressure columns.

            ITolerance ==> float
                    Numerical tolerance used while detecting
                    decreases.

    Output:

            RRelativeDrop ==> pandas.DataFrame

                    Relative drop statistics for every MOF.

    """

    print(f"\n======================================================\n")

    #######################################################
    # Initial checks
    #######################################################

    assert len(IPressureColumns) > 1, \
        f"\033[91mAt least two pressure points are required.\033[0m"

    #######################################################
    # Relative drop analysis
    #######################################################

    RelativeDrop = []

    for iMOF in Idf.index:

        Uptake = Idf.loc[iMOF, IPressureColumns].to_numpy(dtype=float)

        ###################################################
        # Ignore missing values
        ###################################################

        Uptake = Uptake[~np.isnan(Uptake)]

        if len(Uptake) < 2:
            continue

        Difference = np.diff(Uptake)

        NegativeDifference = Difference[Difference < -ITolerance]

        NViolations = len(NegativeDifference)

        if NViolations == 0:

            LargestDrop = 0.0
            RelativeDropPercent = 0.0
            AdsorptionRange = np.max(Uptake) - np.min(Uptake)

            Status = "Monotonic"

        else:

            LargestDrop = abs(np.min(NegativeDifference))

            AdsorptionRange = np.max(Uptake) - np.min(Uptake)

            if AdsorptionRange > ITolerance:

                RelativeDropPercent = \
                    100.0 * LargestDrop / AdsorptionRange

            else:

                RelativeDropPercent = 0.0

            if RelativeDropPercent < 0.5:

                Status = "Numerical Noise"

            elif RelativeDropPercent < 2.0:

                Status = "Needs Inspection"

            else:

                Status = "Potentially Problematic"

        RelativeDrop.append([
            iMOF,
            LargestDrop,
            AdsorptionRange,
            RelativeDropPercent,
            NViolations,
            Status
        ])

    #######################################################
    # Convert to DataFrame
    #######################################################

    RRelativeDrop = pd.DataFrame(
        RelativeDrop,
        columns=[
            "MOF",
            "LargestDrop",
            "AdsorptionRange",
            "RelativeDrop (%)",
            "NViolations",
            "Status"
        ]
    )

    #######################################################
    # Final checks
    #######################################################

    assert len(RRelativeDrop) == len(Idf), \
        f"\033[91mUnexpected number of MOFs.\033[0m"

    #######################################################
    # Save results
    #######################################################

    RRelativeDrop = RRelativeDrop.sort_values(
        "RelativeDrop (%)",
        ascending=False
    )

    RRelativeDrop.to_csv(
        "RelativeDropAnalysis.csv",
        index=False
    )

    #######################################################
    # Print summary
    #######################################################

    print("\nRelative drop analysis written to")

    print("\nRelativeDropAnalysis.csv")

    print(f"\nTotal MOFs = {len(RRelativeDrop)}")

    print(f"\nMonotonic                = {(RRelativeDrop['Status']=='Monotonic').sum()}")

    print(f"\nNumerical Noise          = {(RRelativeDrop['Status']=='Numerical Noise').sum()}")

    print(f"\nNeeds Inspection         = {(RRelativeDrop['Status']=='Needs Inspection').sum()}")

    print(f"\nPotentially Problematic  = {(RRelativeDrop['Status']=='Potentially Problematic').sum()}")

    print("\nWorst five MOFs:\n")

    print(RRelativeDrop.head())

    print(f"\n\033[92m[DEBUG 04:] RelativeDropAnalysis() successfully run !!!\033[0m\n")

    print(f"\n======================================================\n")

    return RRelativeDrop

def PlotRelativeDropIsotherms(Idf, IRelativeDrop, IPressureColumns, ITop=20):

    """

    Description:

            Plots the adsorption isotherms of the most
            problematic MOFs identified by
            RelativeDropAnalysis().

            Only MOFs classified as

                    Potentially Problematic

            are considered. These MOFs are sorted according
            to RelativeDrop (%) and the top ITop MOFs are
            plotted.

            The plots are written to

                    ./RelativeDropIsotherms/

    Input:

            Idf ==> pandas.DataFrame
                    Adsorption database indexed by MOF.

            IRelativeDrop ==> pandas.DataFrame
                    Output of RelativeDropAnalysis().

            IPressureColumns ==> list
                    Pressure columns.

            ITop ==> int
                    Number of MOFs to plot.

    Output:

            Isotherm plots written to

                    ./RelativeDropIsotherms/

    """

    print(f"\n======================================================\n")

    #######################################################
    # Output directory
    #######################################################

    os.makedirs(
        "RelativeDropIsotherms",
        exist_ok=True
    )

    #######################################################
    # Select problematic MOFs
    #######################################################

    # Selecting problamatic MOFs and then sorting them based on RelativeDrop
    Problematic = IRelativeDrop[
        IRelativeDrop["Status"] == "Potentially Problematic"
    ]

    Problematic = Problematic.sort_values(
        "RelativeDrop (%)",
        ascending=False
    )

    Problematic = Problematic.head(ITop)

    #######################################################
    # Plot
    #######################################################

    NPlots = 0

    for _, row in Problematic.iterrows():

        MOF = row["MOF"]

        Uptake = Idf.loc[
            MOF,
            IPressureColumns
        ].to_numpy(dtype=float)

        Pressure = np.array(
            IPressureColumns,
            dtype=float
        )

        mask = ~np.isnan(Uptake)

        Uptake = Uptake[mask]
        Pressure = Pressure[mask]

        plt.figure(figsize=(8,6))

        plt.plot(
            Pressure,
            Uptake,
            "-o",
            linewidth=2,
            markersize=5
        )

        plt.xscale("log")

        plt.xlabel(
            "Pressure (bar)",
            fontdict=fontdict_x
        )

        plt.ylabel(
            "Adsorption uptake (mol/kg)",
            fontdict=fontdict_y
        )

        plt.title(
            "%s\nRelative Drop = %.2f %%   Largest Drop = %.3f   Violations = %d"
            %(
                MOF,
                row["RelativeDrop (%)"],
                row["LargestDrop"],
                row["NViolations"]
            ),
            fontdict=fontdict_t
        )

        plt.tight_layout()

        plt.savefig(
            "RelativeDropIsotherms/%03d_%s.png"
            %(NPlots+1, MOF),
            dpi=300
        )

        plt.close()

        NPlots += 1

    #######################################################
    # Final checks
    #######################################################

    assert NPlots > 0, \
        f"\033[91mNo plots were generated.\033[0m"

    #######################################################
    # Print summary
    #######################################################

    print("\nProblematic isotherms written to")

    print("\n./RelativeDropIsotherms/")

    print(f"\nNumber of plots generated = {NPlots}")

    print(f"\n\033[92m[DEBUG 05:] PlotRelativeDropIsotherms() successfully run !!!\033[0m\n")

    print(f"\n======================================================\n")

    return

def RelativeDropAnalysisNew(Idf, IPressureColumns, ITolerance=1.0e-6, IMinAdsorptionRange=0.1):

    """

    Description:

            Computes the largest decrease in adsorption
            uptake for every MOF and expresses this decrease
            relative to the adsorption range.

            MOFs with negligible adsorption are excluded
            from the relative-drop analysis because very
            small numerical fluctuations may otherwise lead
            to artificially large relative errors.

            The remaining MOFs are classified according to
            their relative drop and ranked using a priority
            score.

            Results are written to

                    RelativeDropAnalysis.csv

    Input:

            Idf ==> pandas.DataFrame

            IPressureColumns ==> list

            ITolerance ==> float

            IMinAdsorptionRange ==> float
                    Minimum adsorption range (mol/kg)
                    required before computing relative
                    drops.

    Output:

            RRelativeDrop ==> pandas.DataFrame

    """

    print(f"\n======================================================\n")

    #######################################################
    # Initial checks
    #######################################################

    assert len(IPressureColumns) > 1, \
        f"\033[91mAt least two pressure points are required.\033[0m"

    #######################################################
    # Relative drop analysis
    #######################################################

    RelativeDrop = []

    for iMOF in Idf.index:

        Uptake = Idf.loc[
            iMOF,
            IPressureColumns
        ].to_numpy(dtype=float)

        ###################################################
        # Ignore missing values
        ###################################################

        Uptake = Uptake[~np.isnan(Uptake)]

        if len(Uptake) < 2:
            continue

        ###################################################
        # Adsorption range
        ###################################################

        AdsorptionRange = np.max(Uptake) - np.min(Uptake)

        ###################################################
        # Negligible adsorption
        ###################################################

        if AdsorptionRange < IMinAdsorptionRange:

            RelativeDrop.append([
                iMOF,
                0.0,
                AdsorptionRange,
                0.0,
                0,
                0.0,
                "Negligible Uptake"
            ])

            continue

        ###################################################
        # Difference
        ###################################################

        Difference = np.diff(Uptake)

        NegativeDifference = Difference[
            Difference < -ITolerance
        ]

        NViolations = len(NegativeDifference)

        ###################################################
        # Perfectly monotonic
        ###################################################

        if NViolations == 0:

            LargestDrop = 0.0

            RelativeDropPercent = 0.0

            PriorityScore = 0.0

            Status = "Monotonic"

        ###################################################
        # Non-monotonic
        ###################################################

        else:

            LargestDrop = abs(
                np.min(NegativeDifference)
            )

            RelativeDropPercent = \
                100.0 * LargestDrop / AdsorptionRange

            PriorityScore = \
                RelativeDropPercent * AdsorptionRange

            if RelativeDropPercent < 0.5:

                Status = "Numerical Noise"

            elif RelativeDropPercent < 2.0:

                Status = "Needs Inspection"

            else:

                Status = "Potentially Problematic"

        RelativeDrop.append([
            iMOF,
            LargestDrop,
            AdsorptionRange,
            RelativeDropPercent,
            NViolations,
            PriorityScore,
            Status
        ])

    #######################################################
    # Convert to dataframe
    #######################################################

    RRelativeDrop = pd.DataFrame(
        RelativeDrop,
        columns=[
            "MOF",
            "LargestDrop",
            "AdsorptionRange",
            "RelativeDrop (%)",
            "NViolations",
            "PriorityScore",
            "Status"
        ]
    )

    #######################################################
    # Sort
    #######################################################

    RRelativeDrop = RRelativeDrop.sort_values(
        "PriorityScore",
        ascending=False
    )

    #######################################################
    # Final checks
    #######################################################

    assert len(RRelativeDrop) == len(Idf), \
        f"\033[91mUnexpected number of MOFs.\033[0m"

    #######################################################
    # Save
    #######################################################

    RRelativeDrop.to_csv(
        "RelativeDropAnalysis.csv",
        index=False
    )

    #######################################################
    # Summary
    #######################################################

    print("\nRelative drop analysis written to")

    print("\nRelativeDropAnalysis.csv")

    print(f"\nTotal MOFs = {len(RRelativeDrop)}")

    print(f"\nNegligible Uptake        = {(RRelativeDrop['Status']=='Negligible Uptake').sum()}")

    print(f"\nMonotonic               = {(RRelativeDrop['Status']=='Monotonic').sum()}")

    print(f"\nNumerical Noise         = {(RRelativeDrop['Status']=='Numerical Noise').sum()}")

    print(f"\nNeeds Inspection        = {(RRelativeDrop['Status']=='Needs Inspection').sum()}")

    print(f"\nPotentially Problematic = {(RRelativeDrop['Status']=='Potentially Problematic').sum()}")

    print("\nWorst five MOFs:\n")

    print(RRelativeDrop.head())

    print(f"\n\033[92m[DEBUG 04:] RelativeDropAnalysis() successfully run !!!\033[0m\n")

    print(f"\n======================================================\n")

    return RRelativeDrop

def PlotRelativeDropIsothermsNew(Idf, IRelativeDrop, IPressureColumns, ITop=20):

    """

    Description:

            Plots the adsorption isotherms of the most
            problematic MOFs identified by
            RelativeDropAnalysis().

            Only MOFs classified as

                    Potentially Problematic

            are considered. These MOFs are sorted according
            to RelativeDrop (%) and the top ITop MOFs are
            plotted.

            The plots are written to

                    ./RelativeDropIsotherms/

    Input:

            Idf ==> pandas.DataFrame
                    Adsorption database indexed by MOF.

            IRelativeDrop ==> pandas.DataFrame
                    Output of RelativeDropAnalysis().

            IPressureColumns ==> list
                    Pressure columns.

            ITop ==> int
                    Number of MOFs to plot.

    Output:

            Isotherm plots written to

                    ./RelativeDropIsotherms/

    """

    print(f"\n======================================================\n")

    #######################################################
    # Output directory
    #######################################################

    os.makedirs(
        "RelativeDropIsotherms",
        exist_ok=True
    )

    #######################################################
    # Select problematic MOFs
    #######################################################

    # Selecting probamatic mofs and then sorting them based on Priority Score
    
    Problematic = IRelativeDrop[
        IRelativeDrop["Status"] == "Potentially Problematic"
    ]

    Problematic = Problematic.sort_values(
        "PriorityScore",
        ascending=False
    )
    
    '''
    # Selecting Negligble Uptake mofs and then sorting them based on Priority Score

    Problematic = IRelativeDrop[
        IRelativeDrop["Status"] == "Potentially Problematic"
    ]

    Problematic = Problematic.sort_values(
        "RelativeDrop (%)",
        ascending=False
    )
    '''

    Problematic = Problematic.head(ITop)

    #######################################################
    # Plot
    #######################################################

    NPlots = 0

    for _, row in Problematic.iterrows():

        MOF = row["MOF"]

        Uptake = Idf.loc[
            MOF,
            IPressureColumns
        ].to_numpy(dtype=float)

        Pressure = np.array(
            IPressureColumns,
            dtype=float
        )

        mask = ~np.isnan(Uptake)

        Uptake = Uptake[mask]
        Pressure = Pressure[mask]

        plt.figure(figsize=(8,6))

        plt.plot(
            Pressure,
            Uptake,
            "-o",
            linewidth=2,
            markersize=5
        )

        plt.xscale("log")

        plt.xlabel(
            "Pressure (bar)",
            fontdict=fontdict_x
        )

        plt.ylabel(
            "Adsorption uptake (mol/kg)",
            fontdict=fontdict_y
        )

        plt.title(
            "%s\nRelative Drop = %.2f %%   Largest Drop = %.3f   Violations = %d"
            %(
                MOF,
                row["RelativeDrop (%)"],
                row["LargestDrop"],
                row["NViolations"]
            ),
            fontdict=fontdict_t
        )

        plt.tight_layout()

        plt.savefig(
            "RelativeDropIsotherms/%03d_%s.png"
            %(NPlots+1, MOF),
            dpi=300
        )

        plt.close()

        NPlots += 1

    #######################################################
    # Final checks
    #######################################################

    assert NPlots > 0, \
        f"\033[91mNo plots were generated.\033[0m"

    #######################################################
    # Print summary
    #######################################################

    print("\nProblematic isotherms written to")

    print("\n./RelativeDropIsotherms/")

    print(f"\nNumber of plots generated = {NPlots}")

    print(f"\n\033[92m[DEBUG 05:] PlotRelativeDropIsotherms() successfully run !!!\033[0m\n")

    print(f"\n======================================================\n")

    return

def ComputeAdsorptionRegionsV1(Idf, IPressureColumns, ITolerance=1.0e-6):

    """

    Description:

            Computes the adsorption regions present in every
            adsorption isotherm.

            Consecutive positive uptake increments are grouped
            together into a single adsorption region.

            This function does not classify the isotherms.
            It only identifies adsorption regions and computes
            the amount of adsorption occurring in every region.

    Input:

            Idf ==> pandas.DataFrame

            IPressureColumns ==> list

            ITolerance ==> float

    Output:

            RAdsorptionRegions ==> pandas.DataFrame

    """

    print(f"\n======================================================\n")

    ########################################################
    # Initial checks
    ########################################################

    assert len(IPressureColumns) > 1, \
        f"\033[91mAt least two pressure points are required.\033[0m"

    ########################################################
    # Region analysis
    ########################################################

    AdsorptionRegions = []

    for iMOF in Idf.index:

        Uptake = Idf.loc[
            iMOF,
            IPressureColumns
        ].to_numpy(dtype=float)

        ####################################################
        # Remove missing values
        ####################################################

        Uptake = Uptake[~np.isnan(Uptake)]

        if len(Uptake) < 2:
            continue

        ####################################################
        # Uptake increments
        ####################################################

        Difference = np.diff(Uptake)

        ####################################################
        # Remove numerical noise
        ####################################################

        Difference[np.abs(Difference) < ITolerance] = 0.0

        ####################################################
        # Ignore negative increments for now
        ####################################################

        Difference[Difference < 0.0] = 0.0

        ####################################################
        # Find adsorption regions
        ####################################################

        Regions = []

        CurrentRegion = 0.0

        for d in Difference:

            if d > 0.0:

                CurrentRegion += d

            else:

                if CurrentRegion > 0.0:

                    Regions.append(CurrentRegion)

                    CurrentRegion = 0.0

        if CurrentRegion > 0.0:

            Regions.append(CurrentRegion)

        ####################################################
        # Region fractions
        ####################################################

        if len(Regions) == 0:

            Fractions = []

        else:

            TotalAdsorption = np.sum(Regions)

            Fractions = np.array(Regions)/TotalAdsorption

            Fractions = Fractions[Fractions > 0.05]

            EffectiveRegions = len(Fractions)

            Fractions = np.sort(Fractions)[::-1]

        ####################################################
        # Store only first three fractions
        ####################################################

        LargestFraction = 0.0
        SecondFraction = 0.0
        ThirdFraction = 0.0

        if len(Fractions) >= 1:
            LargestFraction = Fractions[0]

        if len(Fractions) >= 2:
            SecondFraction = Fractions[1]

        if len(Fractions) >= 3:
            ThirdFraction = Fractions[2]

        AdsorptionRegions.append([

            iMOF,

            len(Regions),

            LargestFraction,

            SecondFraction,

            ThirdFraction,

            Regions

        ])

    ########################################################
    # Convert to dataframe
    ########################################################

    RAdsorptionRegions = pd.DataFrame(

        AdsorptionRegions,

        columns=[

            "MOF",

            "NRegions",

            "LargestFraction",

            "SecondFraction",

            "ThirdFraction",

            "Regions"

        ]

    )

    ########################################################
    # Final checks
    ########################################################

    assert len(RAdsorptionRegions) == len(Idf), \
        f"\033[91mUnexpected number of MOFs.\033[0m"

    ########################################################
    # Save
    ########################################################

    RAdsorptionRegions.to_csv(

        "AdsorptionRegions.csv",

        index=False

    )

    ########################################################
    # Summary
    ########################################################

    print("\nAdsorption regions written to")

    print("\nAdsorptionRegions.csv")

    print(f"\nTotal MOFs = {len(RAdsorptionRegions)}")

    print("\nDistribution of number of regions\n")

    print(
        RAdsorptionRegions["NRegions"].value_counts().sort_index()
    )

    print("\nExample entries\n")

    print(RAdsorptionRegions.head())

    print(f"\n\033[92m[DEBUG 06:] ComputeAdsorptionRegions() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return RAdsorptionRegions

def ComputeAdsorptionRegions(Idf, IPressureColumns, ITolerance=1.0e-6, IRegionFraction=0.05):

    """

    Description:

            Computes the adsorption regions present in every
            adsorption isotherm.

            Consecutive positive uptake increments are grouped
            together into adsorption regions. Very small regions
            contributing less than IRegionFraction of the total
            adsorption are ignored while computing the effective
            number of adsorption regions.

    Input:

            Idf ==> pandas.DataFrame

            IPressureColumns ==> list

            ITolerance ==> float

            IRegionFraction ==> float

                    Minimum fraction of total adsorption
                    required for a region to be considered
                    physically significant.

    Output:

            RAdsorptionRegions ==> pandas.DataFrame

    """

    print(f"\n======================================================\n")

    ########################################################
    # Initial checks
    ########################################################

    assert len(IPressureColumns) > 1, \
        f"\033[91mAt least two pressure points are required.\033[0m"

    ########################################################
    # Region analysis
    ########################################################

    AdsorptionRegions = []

    for iMOF in Idf.index:

        Uptake = Idf.loc[
            iMOF,
            IPressureColumns
        ].to_numpy(dtype=float)

        ####################################################
        # Remove missing values
        ####################################################

        Uptake = Uptake[~np.isnan(Uptake)]

        if len(Uptake) < 2:
            continue

        ####################################################
        # Uptake increments
        ####################################################

        Difference = np.diff(Uptake)

        ####################################################
        # Remove numerical noise
        ####################################################

        Difference[np.abs(Difference) < ITolerance] = 0.0

        ####################################################
        # Ignore negative increments
        ####################################################

        Difference[Difference < 0.0] = 0.0

        ####################################################
        # Identify adsorption regions
        ####################################################

        Regions = []

        CurrentRegion = 0.0

        for d in Difference:

            if d > 0.0:

                CurrentRegion += d

            else:

                if CurrentRegion > 0.0:

                    Regions.append(CurrentRegion)

                    CurrentRegion = 0.0

        if CurrentRegion > 0.0:

            Regions.append(CurrentRegion)

        ####################################################
        # Region fractions
        ####################################################

        if len(Regions) == 0:

            Fractions = np.array([])

        else:

            TotalAdsorption = np.sum(Regions)

            Fractions = np.array(Regions) / TotalAdsorption

            Fractions = np.sort(Fractions)[::-1]

        ####################################################
        # Effective regions
        ####################################################

        EffectiveFractions = Fractions[
            Fractions >= IRegionFraction
        ]

        NRegions = len(EffectiveFractions)

        ####################################################
        # Largest three fractions
        ####################################################

        LargestFraction = 0.0
        SecondFraction = 0.0
        ThirdFraction = 0.0

        if len(EffectiveFractions) >= 1:
            LargestFraction = EffectiveFractions[0]

        if len(EffectiveFractions) >= 2:
            SecondFraction = EffectiveFractions[1]

        if len(EffectiveFractions) >= 3:
            ThirdFraction = EffectiveFractions[2]

        ####################################################
        # Store
        ####################################################

        AdsorptionRegions.append([

            iMOF,

            NRegions,

            LargestFraction,

            SecondFraction,

            ThirdFraction,

            Regions,

            Fractions.tolist()

        ])

    ########################################################
    # Convert to DataFrame
    ########################################################

    RAdsorptionRegions = pd.DataFrame(

        AdsorptionRegions,

        columns=[

            "MOF",

            "NRegions",

            "LargestFraction",

            "SecondFraction",

            "ThirdFraction",

            "Regions",

            "Fractions"

        ]

    )

    ########################################################
    # Final checks
    ########################################################

    assert len(RAdsorptionRegions) == len(Idf), \
        f"\033[91mUnexpected number of MOFs.\033[0m"

    ########################################################
    # Save
    ########################################################

    RAdsorptionRegions.to_csv(

        "AdsorptionRegions.csv",

        index=False

    )

    ########################################################
    # Summary
    ########################################################

    print("\nAdsorption regions written to")

    print("\nAdsorptionRegions.csv")

    print(f"\nTotal MOFs = {len(RAdsorptionRegions)}")

    print(f"\nMinimum region fraction = {100*IRegionFraction:.1f}%")

    print("\nDistribution of effective regions\n")

    print(
        RAdsorptionRegions["NRegions"]
        .value_counts()
        .sort_index()
    )

    print("\nExample entries\n")

    print(
        RAdsorptionRegions.head()
    )

    print(f"\n\033[92m[DEBUG 06:] ComputeAdsorptionRegions() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return RAdsorptionRegions

def IsothermClassification(IAdsorptionRegions, ILangmuirFraction=0.90):

    """

    Description:

            Classifies adsorption isotherms based on the
            adsorption regions computed previously.

            The classification is based on the effective
            number of adsorption regions together with the
            largest adsorption fraction.

    Input:

            IAdsorptionRegions ==> pandas.DataFrame

                        Output of
                        ComputeAdsorptionRegions()

            ILangmuirFraction ==> float

                        Minimum largest adsorption
                        fraction required to classify
                        an isotherm as Langmuir-type.

    Output:

            RClassification ==> pandas.DataFrame

    """

    print(f"\n======================================================\n")

    ########################################################
    # Initial checks
    ########################################################

    RequiredColumns = [

        "MOF",

        "NRegions",

        "LargestFraction",

        "SecondFraction",

        "ThirdFraction"

    ]

    for iCol in RequiredColumns:

        assert iCol in IAdsorptionRegions.columns, \
            f"\033[91mColumn {iCol} not found.\033[0m"

    ########################################################
    # Classification
    ########################################################

    Classification = []

    for _, row in IAdsorptionRegions.iterrows():

        MOF = row["MOF"]

        NRegions = int(row["NRegions"])

        LargestFraction = row["LargestFraction"]

        ####################################################
        # Negligible adsorption
        ####################################################

        if LargestFraction == 0.0:

            Status = "Negligible Uptake"

        ####################################################
        # Langmuir-type
        ####################################################

        elif NRegions == 1 and LargestFraction >= ILangmuirFraction:

            Status = "Langmuir-type"

        ####################################################
        # Single-step
        ####################################################

        elif NRegions == 2:

            Status = "Single-step"

        ####################################################
        # Double-step
        ####################################################

        elif NRegions == 3:

            Status = "Double-step"

        ####################################################
        # Multi-step
        ####################################################

        elif NRegions >= 4:

            Status = "Multi-step"

        ####################################################
        # Borderline
        ####################################################

        else:

            Status = "Needs Inspection"

        Classification.append([

            MOF,

            NRegions,

            LargestFraction,

            row["SecondFraction"],

            row["ThirdFraction"],

            Status

        ])

    ########################################################
    # Convert to dataframe
    ########################################################

    RClassification = pd.DataFrame(

        Classification,

        columns=[

            "MOF",

            "NRegions",

            "LargestFraction",

            "SecondFraction",

            "ThirdFraction",

            "Classification"

        ]

    )

    ########################################################
    # Save
    ########################################################

    RClassification.to_csv(

        "IsothermClassification.csv",

        index=False

    )

    ########################################################
    # Summary
    ########################################################

    print("\nIsotherm classification written to")

    print("\nIsothermClassification.csv")

    print("\nClassification summary\n")

    print(

        RClassification["Classification"]

        .value_counts()

    )

    print("\nExample entries\n")

    print(

        RClassification.head()

    )

    print(f"\n\033[92m[DEBUG 07:] IsothermClassification() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return RClassification

def PlotIsothermClasses(Idf, IClassification, IPressureColumns, INPlots=20):

    """

    Description:

            Generates representative adsorption isotherms
            belonging to every isotherm class.

    """

    print(f"\n======================================================\n")

    #######################################################
    # Initial checks
    #######################################################

    assert "Classification" in IClassification.columns

    #######################################################
    # Create output directory
    #######################################################

    RootDirectory = "IsothermClasses"

    os.makedirs(
        RootDirectory,
        exist_ok=True
    )

    #######################################################
    # Classes
    #######################################################

    Classes = [

        "Langmuir-type",

        "Single-step",

        "Double-step",

        "Multi-step",

        "Needs Inspection",

        "Negligible Uptake"

    ]

    #######################################################
    # Loop over classes
    #######################################################

    TotalPlots = 0

    for iClass in Classes:

        Directory = os.path.join(

            RootDirectory,

            iClass.replace(" ", "_")

        )

        os.makedirs(

            Directory,

            exist_ok=True

        )

        ###################################################
        # Select MOFs
        ###################################################

        Temp = IClassification[

            IClassification["Classification"] == iClass

        ].copy()

        ###################################################
        # Representative ordering
        ###################################################

        if iClass == "Langmuir-type":

            Temp = Temp.sort_values(

                "LargestFraction",

                ascending=False

            )

        elif iClass == "Single-step":

            Temp = Temp.sort_values(

                "SecondFraction",

                ascending=False

            )

        elif iClass == "Double-step":

            Temp["Score"] = (

                Temp["SecondFraction"]

                +

                Temp["ThirdFraction"]

            )

            Temp = Temp.sort_values(

                "Score",

                ascending=False

            )

        elif iClass == "Multi-step":

            Temp["Score"] = (

                Temp["SecondFraction"]

                +

                Temp["ThirdFraction"]

            )

            Temp = Temp.sort_values(

                "Score",

                ascending=False

            )

        ###################################################
        # Plot
        ###################################################

        for _, row in Temp.head(INPlots).iterrows():

            MOF = row["MOF"]

            Uptake = Idf.loc[

                MOF,

                IPressureColumns

            ].to_numpy(dtype=float)

            plt.figure(figsize=(7,6))

            plt.plot(

                IPressureColumns,

                Uptake,

                "-o",

                linewidth=2,

                markersize=5

            )
            
            plt.xscale("log")

            plt.xlabel(

                "Pressure",

                fontdict=fontdict_x

            )

            plt.ylabel(

                "Uptake (mol/kg)",

                fontdict=fontdict_y

            )

            plt.title(

                "%s\n\n"

                "Class : %s\n"

                "Effective Regions : %d\n"

                "Largest Fraction : %.3f\n"

                "Second Fraction : %.3f"

                %(

                    MOF,

                    row["Classification"],

                    row["NRegions"],

                    row["LargestFraction"],

                    row["SecondFraction"]

                ),

                fontsize=12

            )

            plt.tight_layout()

            plt.savefig(

                os.path.join(

                    Directory,

                    MOF + ".png"

                ),

                dpi=300

            )

            plt.close()

            TotalPlots += 1

    #######################################################
    # Summary
    #######################################################

    print("\nRepresentative isotherms written to")

    print("\n./IsothermClasses/")

    print(f"\nTotal plots generated = {TotalPlots}")

    print(f"\n\033[92m[DEBUG 08:] PlotIsothermClasses() successfully run !!!\033[0m")

    print(f"\n======================================================\n")

    return

# Reading Input file

fAdsorption = "sorted-core-2019-ar.csv"

print(f"\nReading file {fAdsorption} ...")

df_Adsorption, PressureColumns = ReadAdsorptionDatabase(fAdsorption)

#######################################################
# Missing value analysis
#######################################################

MissingValueResults = MissingValueAnalysis(df_Adsorption)

#######################################################
# Monotonicity analysis
#######################################################

MonotonicityResults = MonotonicityAnalysis(df_Adsorption)

#######################################################
# Plot problematic isotherms
#######################################################

PlotProblematicIsotherms(df_Adsorption, MonotonicityResults, NPlots=20)

#RelativeDrop = RelativeDropAnalysis(df_Adsorption, PressureColumns)

#PlotRelativeDropIsotherms(df_Adsorption, RelativeDrop, PressureColumns, ITop=20)

RelativeDrop = RelativeDropAnalysisNew(df_Adsorption, PressureColumns)

PlotRelativeDropIsothermsNew(df_Adsorption, RelativeDrop, PressureColumns, ITop=20)

#AdsorptionRegions = ComputeAdsorptionRegions(df_Adsorption, PressureColumns)

AdsorptionRegions = ComputeAdsorptionRegions(df_Adsorption, PressureColumns, ITolerance=1.0e-6, IRegionFraction=0.05)

print("\nLargest fraction statistics\n")

print(AdsorptionRegions["LargestFraction"].describe())

print("\nSecond fraction statistics\n")

print(AdsorptionRegions["SecondFraction"].describe())

Classification = IsothermClassification(AdsorptionRegions, ILangmuirFraction=0.90)

PlotIsothermClasses(df_Adsorption, Classification, PressureColumns, INPlots=20)

