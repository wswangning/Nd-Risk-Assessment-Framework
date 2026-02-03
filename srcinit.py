"""
Integrated AIVIVE-PBPK-QIVIVE Framework for Neodymium Nitrate Risk Assessment

This package provides tools for mechanism-based risk assessment of rare earth elements.
"""

__version__ = "1.0.0"
__author__ = "Shanghai Municipal Center for Disease Control and Prevention"
__email__ = "wangning@scdc.sh.cn"

from . import data_processing
from . import models
from . import analysis
from . import visualization
from . import utils

__all__ = ["data_processing", "models", "analysis", "visualization", "utils"]