{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration for Neodymium Nitrate Risk Assessment\n",
    "\n",
    "This notebook explores the experimental data used in the integrated risk assessment framework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import yaml\n",
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "# Setup\n",
    "sns.set_style(\"whitegrid\")\n",
    "plt.rcParams['figure.figsize'] = (12, 8)\n",
    "plt.rcParams['font.size'] = 12\n",
    "\n",
    "# Project paths\n",
    "PROJECT_ROOT = Path('..').resolve()\n",
    "DATA_DIR = PROJECT_ROOT / 'data'\n",
    "CONFIG_DIR = PROJECT_ROOT / 'config'\n",
    "\n",
    "print(f\"Project root: {PROJECT_ROOT}\")\n",
    "print(f\"Data directory: {DATA_DIR}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load configuration\n",
    "config_path = CONFIG_DIR / 'model_params.yaml'\n",
    "with open(config_path, 'r') as f:\n",
    "    config = yaml.safe_load(f)\n",
    "\n",
    "print(\"Configuration loaded successfully\")\n",
    "print(f\"AIVIVE input dimension: {config['aivive']['input_dim']}\")\n",
    "print(f\"QIVIVE EC50: {config['qivive']['ec50']} μM\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Experimental Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Experimental parameters\n",
    "exp_config_path = CONFIG_DIR / 'experimental_params.yaml'\n",
    "with open(exp_config_path, 'r') as f:\n",
    "    exp_config = yaml.safe_load(f)\n",
    "\n",
    "# Plasma protein binding data\n",
    "pb_data = exp_config['experimental']['plasma_protein_binding']\n",
    "pb_df = pd.DataFrame({\n",
    "    'Concentration (μg/mL)': pb_data['concentrations'],\n",
    "    'Free Fraction (fu)': pb_data['fu_values'],\n",
    "    'Binding Rate (%)': 100 * (1 - np.array(pb_data['fu_values']))\n",
    "})\n",
    "\n",
    "print(\"Plasma Protein Binding Data:\")\n",
    "print(pb_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot plasma protein binding\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))\n",
    "\n",
    "# Free fraction plot\n",
    "ax1.plot(pb_df['Concentration (μg/mL)'], pb_df['Free Fraction (fu)'], \n",
    "         'o-', linewidth=2, markersize=10, color='steelblue')\n",
    "ax1.set_xscale('log')\n",
    "ax1.set_xlabel('Concentration (μg/mL)')\n",
    "ax1.set_ylabel('Free Fraction (fu)')\n",
    "ax1.set_title('Plasma Protein Binding - Free Fraction')\n",
    "ax1.grid(True, alpha=0.3)\n",
    "\n",
    "# Binding rate plot\n",
    "ax2.plot(pb_df['Concentration (μg/mL)'], pb_df['Binding Rate (%)'], \n",
    "         's--', linewidth=2, markersize=10, color='coral')\n",
    "ax2.set_xscale('log')\n",
    "ax2.set_xlabel('Concentration (μg/mL)')\n",
    "ax2.set_ylabel('Binding Rate (%)')\n",
    "ax2.set_title('Plasma Protein Binding - Binding Rate')\n",
    "ax2.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Partition Coefficient Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Partition coefficient data\n",
    "k_data = exp_config['experimental']['partition_coefficients']\n",
    "k_df = pd.DataFrame({\n",
    "    'Concentration (μM)': k_data['concentrations'],\n",
    "    'Partition Coefficient (K)': k_data['k_values'],\n",
    "    'log10(Concentration)': np.log10(k_data['concentrations'])\n",
    "})\n",
    "\n",
    "print(\"Partition Coefficient Data:\")\n",
    "print(k_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot partition coefficients\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "ax.plot(k_df['Concentration (μM)'], k_df['Partition Coefficient (K)'], \n",
    "        'o-', linewidth=2, markersize=12, color='seagreen')\n",
    "\n",
    "# Add annotations\n",
    "for i, row in k_df.iterrows():\n",
    "    ax.annotate(f\"K={row['Partition Coefficient (K)']:.3f}\", \n",
    "                xy=(row['Concentration (μM)'], row['Partition Coefficient (K)']),\n",
    "                xytext=(10, 10), textcoords='offset points',\n",
    "                fontsize=10, color='darkgreen')\n",
    "\n",
    "ax.set_xlabel('Concentration (μM)')\n",
    "ax.set_ylabel('Partition Coefficient (K)')\n",
    "ax.set_title('Concentration-Dependent Partition Coefficients in HepG2 Cells')\n",
    "ax.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. In Vitro Toxicity Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# In vitro toxicity data\n",
    "ec50 = exp_config['experimental']['hepg2_ec50']\n",
    "ec50_ci = exp_config['experimental']['ec50_ci']\n",
    "\n",
    "print(f\"HepG2 EC50: {ec50} μM\")\n",
    "print(f\"95% Confidence Interval: [{ec50_ci[0]}, {ec50_ci[1]}] μM\")\n",
    "print(f\"Range: {ec50_ci[1] - ec50_ci[0]:.1f} μM\")\n",
    "\n",
    "# Create dose-response curve (simulated)\n",
    "doses = np.logspace(-1, 3, 50)  # 0.1 to 1000 μM\n",
    "hill_coeff = config['qivive']['hill_coefficient']\n",
    "\n",
    "# Hill equation\n",
    "def hill_equation(dose, ec50, n, bottom=0, top=100):\n",
    "    return bottom + (top - bottom) / (1 + (ec50/dose)**n)\n",
    "\n",
    "response = hill_equation(doses, ec50, hill_coeff)\n",
    "\n",
    "# Plot dose-response curve\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "ax.plot(doses, response, 'b-', linewidth=3, label=f'Hill curve (n={hill_coeff})')\n",
    "ax.axvline(ec50, color='r', linestyle='--', linewidth=2, label=f'EC50 = {ec50} μM')\n",
    "ax.fill_betweenx([0, 100], ec50_ci[0], ec50_ci[1], \n",
    "                 alpha=0.2, color='red', label='95% CI')\n",
    "\n",
    "ax.set_xscale('log')\n",
    "ax.set_xlabel('Concentration (μM)')\n",
    "ax.set_ylabel('Response (% of max)')\n",
    "ax.set_title('Dose-Response Curve for Neodymium Nitrate in HepG2 Cells')\n",
    "ax.legend(loc='upper left')\n",
    "ax.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Statistical Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create summary dataframe\n",
    "summary_data = {\n",
    "    'Parameter': [\n",
    "        'EC50 (HepG2)',\n",
    "        'Free Fraction (fu, mid-concentration)',\n",
    "        'Partition Coefficient (K, mid-concentration)',\n",
    "        'Hill Coefficient',\n",
    "        'Reference Dose (RfD)'\n",
    "    ],\n",
    "    'Value': [\n",
    "        f\"{ec50} μM\",\n",
    "        f\"{pb_df['Free Fraction (fu)'].iloc[1]:.3f}\",\n",
    "        f\"{k_df['Partition Coefficient (K)'].iloc[1]:.3f}\",\n",
    "        f\"{hill_coeff}\",\n",
    "        f\"{config['risk']['reference_dose']} mg/kg/day\"\n",
    "    ],\n",
    "    'Range/Uncertainty': [\n",
    "        f\"[{ec50_ci[0]}, {ec50_ci[1]}] μM\",\n",
    "        f\"[{pb_df['Free Fraction (fu)'].min():.3f}, {pb_df['Free Fraction (fu)'].max():.3f}]\",\n",
    "        f\"[{k_df['Partition Coefficient (K)'].min():.3f}, {k_df['Partition Coefficient (K)'].max():.3f}]\",\n",
    "        \"N/A\",\n",
    "        \"Based on NOAEL/100\"\n",
    "    ],\n",
    "    'Source': [\n",
    "        'HepG2 cytotoxicity assay',\n",
    "        'Equilibrium dialysis',\n",
    "        'HepG2 exposure experiment',\n",
    "        'Dose-response fitting',\n",
    "        'Literature + safety factor'\n",
    "    ]\n",
    "}\n",
    "\n",
    "summary_df = pd.DataFrame(summary_data)\n",
    "print(\"Experimental Data Summary:\")\n",
    "print(summary_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save summary to CSV\n",
    "output_path = PROJECT_ROOT / 'outputs' / 'data_summary.csv'\n",
    "output_path.parent.mkdir(parents=True, exist_ok=True)\n",
    "summary_df.to_csv(output_path, index=False)\n",
    "print(f\"Summary saved to: {output_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create correlation matrix for key parameters\n",
    "# (In real data, this would use actual measurements)\n",
    "\n",
    "# Simulated parameter correlations\n",
    "np.random.seed(42)\n",
    "n_samples = 100\n",
    "\n",
    "# Generate correlated parameters\n",
    "ec50_samples = np.random.lognormal(mean=np.log(ec50), sigma=0.3, size=n_samples)\n",
    "# fu correlates negatively with log concentration\n",
    "log_conc = np.random.uniform(-1, 2, n_samples)\n",
    "fu_samples = 0.5 / (1 + np.exp(0.5 * (log_conc - 1))) + np.random.normal(0, 0.05, n_samples)\n",
    "fu_samples = np.clip(fu_samples, 0.01, 0.99)\n",
    "# K shows inverted U-shape relationship\n",
    "k_samples = 0.1 * log_conc * np.exp(-0.3 * log_conc) + np.random.normal(0, 0.01, n_samples)\n",
    "k_samples = np.clip(k_samples, 0.02, 0.15)\n",
    "\n",
    "# Create dataframe\n",
    "sim_data = pd.DataFrame({\n",
    "    'EC50 (μM)': ec50_samples,\n",
    "    'Free Fraction (fu)': fu_samples,\n",
    "    'Partition Coefficient (K)': k_samples,\n",
    "    'log10(Concentration)': log_conc\n",
    "})\n",
    "\n",
    "# Calculate correlations\n",
    "corr_matrix = sim_data.corr()\n",
    "\n",
    "# Plot correlation heatmap\n",
    "fig, ax = plt.subplots(figsize=(10, 8))\n",
    "\n",
    "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', \n",
    "            center=0, square=True, linewidths=1, ax=ax)\n",
    "\n",
    "ax.set_title('Parameter Correlation Matrix (Simulated Data)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Export for Next Steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save processed data for next steps\n",
    "processed_data = {\n",
    "    'plasma_protein_binding': pb_df.to_dict('list'),\n",
    "    'partition_coefficients': k_df.to_dict('list'),\n",
    "    'in_vitro_toxicity': {\n",
    "        'ec50': ec50,\n",
    "        'ec50_ci': ec50_ci,\n",
    "        'hill_coefficient': hill_coeff\n",
    "    },\n",
    "    'parameter_correlations': corr_matrix.to_dict()\n",
    "}\n",
    "\n",
    "output_json = PROJECT_ROOT / 'outputs' / 'processed_experimental_data.json'\n",
    "with open(output_json, 'w') as f:\n",
    "    json.dump(processed_data, f, indent=2)\n",
    "\n",
    "print(f\"Processed data saved to: {output_json}\")\n",
    "print(\"\\nData exploration complete. Next steps:\")\n",
    "print(\"1. Run HTTK calibration (notebooks/02_HTTK_Calibration.ipynb)\")\n",
    "print(\"2. Train AIVIVE model (notebooks/03_AIVIVE_Training.ipynb)\")\n",
    "print(\"3. Run full risk assessment (scripts/run_full_pipeline.py)\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}