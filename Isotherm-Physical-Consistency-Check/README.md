# Ar Isotherm Physical Consistency Analysis

## Overview

This folder contains Python code for analysing the physical consistency and shape of argon (Ar) adsorption isotherms in metal–organic frameworks (MOFs). The analysis is motivated by the need to assess whether adsorption isotherms exhibit the expected behaviour with increasing pressure.

The script currently analyses the Ar adsorption database and provides tools for identifying non-monotonic isotherms, quantifying uptake decreases, and characterising adsorption-isotherm shapes.

## Analysis Workflow

The script performs the following analyses:

1. **Missing-value analysis:** Quantifies the completeness of the adsorption isotherm for each MOF.
2. **Monotonicity analysis:** Identifies decreases in uptake between consecutive pressure points using a specified numerical tolerance.
3. **Relative-drop analysis:** Quantifies the largest uptake decrease relative to the overall adsorption range and flags potentially problematic isotherms.
4. **Isotherm visualisation:** Generates plots of selected isotherms for further inspection.
5. **Adsorption-region analysis:** Groups consecutive positive uptake increments into adsorption regions.
6. **Isotherm classification:** Classifies isotherms as Langmuir-type, single-step, double-step, multi-step, or other categories based on the implemented criteria.

## Requirements

The script requires Python and the following packages:

- NumPy
- pandas
- Matplotlib
- seaborn
- SciPy
- scikit-learn
- tqdm

Install the dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

## Input Data

The input filename is specified in the script:

```python
fAdsorption = "sorted-core-2019-ar.csv"
```

The input CSV file should contain a `MOF` column and separate columns for adsorption uptake at different pressure points. The script uses the MOF identifiers as the DataFrame index.

Ensure that the pressure columns are numeric and ordered consistently with increasing pressure.

## Running the Analysis

Place the input CSV file in the working directory and run:

```bash
python ar-isotherm-physical-consistency.py
```

The script generates CSV summaries and plots in the working directory, including missing-value statistics, monotonicity results, relative-drop analysis, adsorption-region statistics, isotherm classifications, and representative isotherm plots.

## Scope and Limitations

The numerical tolerance used for monotonicity analysis helps distinguish small numerical fluctuations from larger uptake decreases. The relative-drop analysis also helps prioritise isotherms for visual inspection.

The current script analyses the specified input database. To assess the physical consistency of low-rank-model reconstructions, the input must instead contain the reconstructed uptake values. An explicit non-negativity check should also be performed before drawing conclusions about the physical consistency of reconstructed isotherms.

The isotherm classifications are based on the criteria implemented in the script and should be interpreted as computational classifications rather than definitive assignments of adsorption mechanisms.

## Reproducibility

The random seeds and plotting settings are defined in the script. The input dataset and analysis parameters should be documented when reporting results to facilitate reproducibility.

