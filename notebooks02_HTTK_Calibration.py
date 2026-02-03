{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# HTTK Model Calibration for Neodymium Nitrate\n",
    "\n",
    "This notebook demonstrates the calibration of HTTK models for metal compounds using experimental data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import and setup\n",
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "from src.models.httk_calibration import HTTKCalibrator\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import yaml\n",
    "import json\n",
    "\n",
    "# Load configuration\n",
    "with open('../config/model_params.yaml', 'r') as f:\n",
    "    config = yaml.safe_load(f)\n",
    "\n",
    "# Initialize calibrator\n",
    "calibrator = HTTKCalibrator(config['httk'])"
   ]
  }
  # ... 更多代码 ...
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}