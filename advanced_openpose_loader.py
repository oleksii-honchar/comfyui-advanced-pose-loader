"""
AdvancedOpenposeLoader - Enhanced pose-guided image generation with ControlNet

This module loads ControlNet models and pose images, applies pose guidance to
the generation process, and optionally applies spatial fade masks for localized
pose influence.

Features:
- Automatic pose image encoding via the loaded VAE
- Support for multiple pose formats (PNG, JPG)
- Strength control (global + per-channel)
- Optional spatial fade masks for localized pose influence
- Multiple ControlNet models chained together
"""

from __future__ import annotations

import os
import sys
import hashlib
import numpy as np
from pathlib import Path

# Add node root to path so src imports work regardless of cwd
NODE_ROOT = os.path.abspath(os.path.dirname(__file__))
if NODE_ROOT not in sys.path:
    sys.path.insert(0, NODE_ROOT)

# Now import src modules
from src.node import AdvancedOpenposeLoader, CONTROLNET_NODE_CLASS_NAME, POSE_LOADER_NODE_CLASS_NAME

__version__ = "2.0.0"

NODE_CLASS_MAPPINGS = {
    CONTROLNET_NODE_CLASS_NAME: AdvancedOpenposeLoader,
    POSE_LOADER_NODE_CLASS_NAME: AdvancedOpenposeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    CONTROLNET_NODE_CLASS_NAME: "Advanced OpenPose ControlNet",
    POSE_LOADER_NODE_CLASS_NAME: "Advanced Pose Image Loader",
}