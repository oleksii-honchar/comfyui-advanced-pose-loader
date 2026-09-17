"""Pose image loading from disk."""
from __future__ import annotations
import torch
import comfy.utils


def load_pose_from_file(pose_path):
    """Load a pose image from a file path."""
    try:
        # Use comfy's image loading utilities
        pose_tensor = comfy.utils.load_image(pose_path, "RGB")
        # Convert to [0, 1] range tensor
        if pose_tensor.max() > 1.0:
            pose_tensor = pose_tensor / 255.0
        return pose_tensor
    except Exception as e:
        raise ValueError(f"Failed to load pose image from {pose_path}: {e}")