"""Pose image loading from disk.

Supports loading pose images from a standardized folder structure:
poses/{folder_name}/
    ├── openpose.png
    ├── openpose_hand.png
    ├── openpose_full.png
    ├── canny.png
    ├── depth.png
    └── normal.png

Handles path resolution for both Docker (/opt/comfyui/poses) and local
environments via ComfyUI's folder_paths module.
"""
from __future__ import annotations
import os
import sys
import logging
import torch
import comfy.utils
from PIL import Image
import numpy as np
import folder_paths

logger = logging.getLogger(__name__)

# All six pose types supported by the AdvancedOpenposeLoader
POSE_TYPES = [
    "openpose",
    "openpose_hand",
    "openpose_full",
    "canny",
    "depth",
    "normal",
]


def resolve_pose_folder(folder_name):
    """Resolve the pose folder path for a given folder name.

    Handles both Docker (/opt/comfyui/poses) and local environments.
    In Docker, checks for /opt/comfyui. On local systems, resolves
    via ComfyUI's folder_paths input directory or falls back to CWD.

    Args:
        folder_name: Name of the pose folder (e.g., 'lora-FB-L1')

    Returns:
        Absolute path to the pose folder
    """
    # Check if we're in Docker environment
    if os.path.exists('/opt/comfyui'):
        base_path = '/opt/comfyui/poses'
    else:
        # Local environment - use ComfyUI's input directory
        try:
            input_dir = folder_paths.input_dir()
            base_path = os.path.join(os.path.dirname(input_dir), 'poses')
        except Exception:
            # Fallback to current directory
            base_path = os.path.join(os.getcwd(), 'poses')

    full_path = os.path.join(base_path, folder_name)

    # Validate folder exists and is readable
    if not os.path.exists(full_path):
        raise ValueError(f"Pose folder does not exist: {full_path}")
    if not os.path.isdir(full_path):
        raise ValueError(f"Pose folder is not a directory: {full_path}")
    if not os.access(full_path, os.R_OK):
        raise ValueError(f"Pose folder is not readable: {full_path}")

    logger.info(f"Resolved pose folder: {full_path}")
    return full_path


def list_pose_images(folder_path):
    """List pose type files matching filename convention in the given folder.

    Searches for files matching: openpose.png, openpose_hand.png,
    openpose_full.png, canny.png, depth.png, normal.png (case-insensitive).

    Args:
        folder_path: Path to the pose folder

    Returns:
        Dict mapping pose type names to file paths for each found pose image.
        Missing types are simply not included (graceful handling).
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Pose folder does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Pose folder is not a directory: {folder_path}")
    if not os.access(folder_path, os.R_OK):
        raise ValueError(f"Pose folder is not readable: {folder_path}")

    result = {}
    try:
        files = os.listdir(folder_path)
    except PermissionError as e:
        raise ValueError(f"Cannot list files in pose folder {folder_path}: {e}")

    # Build lookup map: lowercase filename stem -> actual filename
    file_map = {}
    for f in files:
        if f.lower().endswith('.png'):
            stem = os.path.splitext(f)[0].lower()
            file_map[stem] = f

    for pose_type in POSE_TYPES:
        # Check for exact match first, then case-insensitive
        if pose_type in file_map:
            file_path = os.path.join(folder_path, file_map[pose_type])
            result[pose_type] = file_path
            logger.info(f"Found pose image: {pose_type} -> {file_map[pose_type]}")
        else:
            logger.info(f"Missing pose image: {pose_type}")

    if not result:
        logger.warning(f"No pose images found in {folder_path}")

    return result


def load_pose_image(file_path):
    """Load a specific pose image file and return as tensor.

    Uses comfy.utils.load_image and normalizes to [0, 1] range if needed.

    Args:
        file_path: Path to the pose image file

    Returns:
        Tensor representing the pose image in [0, 1] range
    """
    try:
        # Use Pillow directly (comfy.utils.load_image removed in newer ComfyUI)
        img = Image.open(file_path)
        img = img.convert("RGB")
        pose_tensor = torch.from_numpy(np.array(img, dtype=np.float32))
        # Convert to [0, 1] range tensor
        if pose_tensor.max() > 1.0:
            pose_tensor = pose_tensor / 255.0
        logger.info(f"Loaded pose image: {file_path}")
        return pose_tensor
    except Exception as e:
        raise ValueError(f"Failed to load pose image from {file_path}: {e}")
