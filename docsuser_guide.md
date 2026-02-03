# User Guide: Integrated Risk Assessment Framework

## Overview

This framework provides tools for mechanism-based risk assessment of chemicals, with a focus on data-limited compounds like rare earth elements. The framework integrates:

1. **AIVIVE**: AI-enhanced in vitro to in vivo extrapolation
2. **PBPK**: Physiologically based pharmacokinetic modeling
3. **QIVIVE**: Quantitative in vitro to in vivo extrapolation
4. **HTTK calibration**: Adaptation of high-throughput models for metal compounds

## Installation

### Prerequisites
- Python 3.9 or higher
- R 4.2 or higher (for HTTK modeling)
- Git

### Quick Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Nd-Risk-Assessment-Framework.git
cd Nd-Risk-Assessment-Framework

# Create conda environment
conda env create -f environment.yml
conda activate nd_risk

# Install R dependencies
Rscript scripts/install_R_dependencies.R