"""
TransActNet: Human Activity Recognition with Enhanced CNN and Transformer Attention Blocks.

Modules:
- model: Defines the CNN-Transformer architecture.
- data: Handles data loading and preprocessing.
- training: Contains training scripts and utilities.
- evaluation: Provides model evaluation and benchmarking functions.
"""

__version__ = "1.0.0"

# Import core functionality
import numpy as np
import torch
import torch.nn as nn
import argparse

# Import utility functions
from util import load_data_n_model

# Import all models (from models/__init__.py)
from models import *  # This imports all classes and functions from NTU_Fi_model, UT_HAR_model, and WIDAR_model
