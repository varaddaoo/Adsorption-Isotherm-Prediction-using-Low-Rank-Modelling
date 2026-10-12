# Latent–Structure Correlation Analysis

This script analyses the correlations between material latent factors obtained from Generalized Low Rank Models (GLRM) and structural descriptors of metal–organic frameworks (MOFs).

The structural descriptors considered are:

- Void fraction (VF)
- Gravimetric surface area (GSA)
- Volumetric surface area (VSA)
- Pore-limiting diameter (PLD)
- Largest cavity diameter (LCD)

Pearson and Spearman correlation coefficients are calculated and visualized using heatmaps.

## Requirements

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- SciPy

## Usage

Update the input CSV path, GLRM result pickle path, and other settings in the script before running it.

```bash
python latent-structure-correlation.py
```

The script saves correlation tables and figures in the output directory specified in the code.

**Note:** The material ordering in the structural-property CSV must match the ordering used to generate the GLRM latent factors.
