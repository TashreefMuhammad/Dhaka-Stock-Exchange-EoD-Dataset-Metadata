# Dhaka-Stock-Exchange-EoD-Dataset-Metadata

## Temporal Coverage Bias in Financial Panel Data
### Coverage-Aware Structuring Framework with Evidence from the Dhaka Stock Exchange

This repository contains the code, dataset structure, and experimental pipeline used in the study:

**Temporal Coverage Bias in Financial Panel Data: A Coverage-Aware Structuring Framework with Evidence from the Dhaka Stock Exchange**

Submitted to *Financial Innovation (Springer)*.

## Overview

Financial panel datasets frequently align multiple financial instruments across a shared calendar. However, instruments are listed at different times, resulting in heterogeneous observation windows.

Naively extending time series backward to match a common calendar introduces **temporal coverage bias**, which can distort statistical estimates of volatility and risk.

This project proposes a **coverage-aware dataset structuring framework** that explicitly encodes instrument availability through an availability matrix and evaluates the statistical consequences of naive temporal alignment.

## Key Contributions

- Introduces a **coverage-aware dataset structuring framework** for financial panel datasets.
- Formalizes **temporal coverage bias** arising from naive temporal alignment of financial instruments.
- Demonstrates that naive temporal padding can distort volatility estimates by:
  - ~20% reduction in return volatility
  - ~26% distortion in conditional variance estimates.
- Provides reproducible experiments using **ARIMA and GARCH models**.

## Dataset

The dataset contains end-of-day trading records for instruments listed on the **Dhaka Stock Exchange (DSE)**.

Coverage:

- Period: **October 2012 – January 2026**
- Instruments: **486**
- Asset classes:
  - equities
  - treasury bills
  - mutual funds
  - bonds

Two dataset versions are provided:

1. **Unadjusted dataset** – raw historical price records
2. **Adjusted dataset** – incorporates corporate action adjustments


## Coverage-Aware Representation

Instrument availability across time is encoded using an **availability matrix**:

A(i,t) ∈ {0,1,2,3}

Where:

0 = no observation available  
1 = observation available in adjusted dataset only  
2 = observation available in unadjusted dataset only  
3 = observation available in both datasets

This representation preserves heterogeneous listing windows and avoids artificial temporal padding.

## Experimental Pipeline

The experimental evaluation compares two dataset constructions:

1. **Coverage-aware dataset**
2. **Naively aligned dataset with temporal padding**

Steps:

1. Construct dataset representations
2. Compute log returns
3. Fit ARIMA models for illustrative analysis
4. Estimate conditional variance using GARCH models
5. Compute distortion metrics between the two constructions

## Key Findings

Across 53 instruments:

- Mean return volatility distortion ≈ **20%**
- Mean conditional variance distortion ≈ **26%**

These results demonstrate that naive temporal alignment can significantly bias volatility estimates.

## Reproducibility

Install dependencies:
```
pip install -r requirements.txt
```
Run experiments:
```
python scripts/arima_single_demo.py
python scripts/experiments/coverage_vs_naive.py
python scripts/experiments/cross_instrument_arima.py
python scripts/experiments/multi_instrument_volatility_robustness.py
```

Generate figures:

```
python scripts/generate_figures.py
python scripts/generate_distortion_plots.py
```

And any other .py files associated in the repository should run easily. The codes should also run seamlessly in anaconda.

## License

This repository is released for academic and research use.
