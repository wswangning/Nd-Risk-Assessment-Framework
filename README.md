# Integrated AIVIVE-PBPK-QIVIVE Framework for Neodymium Nitrate Risk Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![R 4.2+](https://img.shields.io/badge/R-4.2+-blue.svg)](https://www.r-project.org/)

## Overview
This repository contains the complete implementation of the integrated computational toxicology framework described in the paper:
**"An Integrated AIVIVE-PBPK-QIVIVE Framework with HTTK Validation for Probabilistic Risk Assessment of Neodymium Nitrate"**.

The framework combines:
- **AIVIVE**: AI-enhanced in vitro to in vivo extrapolation using conditional GANs
- **PBPK**: Physiologically based pharmacokinetic modeling with nonlinear kinetics
- **QIVIVE**: Quantitative in vitro to in vivo extrapolation
- **HTTK validation**: High-throughput toxicokinetic model calibration for metal compounds

## Key Features
- ✅ **Mechanistic prediction** of toxicity pathways (p53, apoptosis, ferroptosis)
- ✅ **Nonlinear kinetics** modeling (concentration-dependent protein binding)
- ✅ **Probabilistic risk assessment** with Monte Carlo simulation
- ✅ **Global sensitivity analysis** using Sobol indices
- ✅ **Modular parameter interface** supporting multiple data sources

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/Nd-Risk-Assessment-Framework.git
cd Nd-Risk-Assessment-Framework

# Create conda environment
conda env create -f environment.yml
conda activate nd_risk

# Install R dependencies (for HTTK)
Rscript scripts/install_R_dependencies.R
