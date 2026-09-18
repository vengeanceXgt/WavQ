#!/usr/bin/env python
"""CPU-only training wrapper. Sets environment to prevent CUDA DLL loading issues."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

# Prevent torch from trying to load CUDA DLLs on systems with limited virtual memory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Now safe to import torch
import torch
torch.set_default_device("cpu")

from rf_analyzer.amc.train_amc import main
main()
