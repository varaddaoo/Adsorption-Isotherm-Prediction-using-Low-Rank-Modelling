# GLRM for Reconstruction of Adsorption Matrices

This repository contains the implementation of a machine learning framework based on **Generalized Low Rank Models (GLRM)** for reconstructing incomplete adsorption matrices and predicting adsorption isotherms across a material–pressure space.

The approach leverages the inherent **low-rank structure of adsorption data**, enabling accurate prediction of missing adsorption values while preserving the physical characteristics of adsorption isotherms.

---

## Scientific Motivation

High-throughput screening of nanoporous materials often produces **incomplete adsorption datasets**, where adsorption values are missing for several material–pressure combinations. Traditional approaches require expensive molecular simulations or experiments to fill these gaps.

This work introduces a **GLRM-based framework** that learns a low-dimensional representation of adsorption behavior, allowing the reconstruction of missing entries in the adsorption matrix.

Within this framework, the adsorption matrix \(X\) is approximated as

\[
X \approx MP^T
\]

where

- \(M\) represents latent **material factors**
- \(P\) represents latent **pressure factors**

Adsorption loadings are therefore interpreted as the **inner product between material and pressure latent vectors**, providing both predictive capability and geometric interpretability.

---

## Features

- Reconstruction of incomplete adsorption matrices
- Prediction of adsorption isotherms across pressures
- Latent-space analysis of materials and pressure conditions
- Visualization tools for adsorption behavior and model performance
- Reproducible pipeline for adsorption matrix reconstruction

---
