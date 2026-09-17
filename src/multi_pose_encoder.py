"""Multi-pose encoder for FLUX.2 Fun ControlNet.

Handles loading, resizing, and encoding multiple pose types from a pose folder
into the FLUX.2 Fun ControlNet control context format.

FLUX.2 uses a 36-dim control context [16, 4, 16] (control: 16, mask: 4, inpaint: 16),
NOT the 260-dim FLUX.1 structure. This is because FLUX.2 uses the 16-channel latent
space of flux2-vae.safetensors.

Pose types (6 total, from pose folder):
  - openpose: body pose keypoints
  - openpose_hand: hand pose keypoints
  - openpose_full: body + hand + face pose
  - canny: edge detection
  - depth: depth map
  - normal: surface normal map
"""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

# Lazy imports — these only resolve in actual ComfyUI environment
_comfy_model_management = None
_comfy_utils = None

def _get_comfy_model_management():
    global _comfy_model_management
    if _comfy_model_management is None:
        import comfy.model_management as mm
        _comfy_model_management = mm
    return _comfy_model_management

def _get_comfy_utils():
    global _comfy_utils
    if _comfy_utils is None:
        import comfy.utils as utils
        _comfy_utils = utils
    return _comfy_utils

from src.pose_preprocessor import generate_spatial_fade_mask, apply_spatial_fade

logger = logging.getLogger(__name__)


def encode_pose_image(vae, pose_image):
    """VAE-encode a pose image to 16-channel latent.

    Args:
        vae: ComfyUI VAE object
        pose_image: Tensor [B, H, W, C] in [0, 1] range

    Returns:
        16-channel latent tensor [B, 16, H/8, W/8]
    """
    mm = _get_comfy_model_management()
    device = mm.get_torch_device()
    dtype = torch.bfloat16 if mm.should_use_bf16() else torch.float16

    # Normalize to [0, 1] if needed
    if pose_image.max() > 1.0:
        pose_image = pose_image / 255.0

    with torch.no_grad():
        pose_latent = vae.encode(pose_image.to(device))

    return pose_latent


def build_control_context(latent, fade_mode="none", fade_strength=0.5):
    """Build 36-dim control context [control(16), mask(4), inpaint(16)].

    Args:
        latent: 16-channel latent tensor [B, 16, H, W]
        fade_mode: Spatial fade mode ('none', 'top', 'bottom', 'left', 'right')
        fade_strength: Fade strength (0.0-1.0)

    Returns:
        36-channel control context tensor [B, 36, H, W]
    """
    device = latent.device
    dtype = latent.dtype
    bs, channels, h, w = latent.shape

    # Control channels: the VAE-encoded latent (16 channels)
    control = latent

    # Mask channels: spatial fade mask (4 channels)
    if fade_mode == "none" or fade_strength <= 0.0:
        mask = torch.ones((bs, 4, h, w), device=device, dtype=dtype)
    else:
        fade_mask = generate_spatial_fade_mask(h, w, fade_mode, fade_strength)
        mask = fade_mask.expand(bs, 4, h, w).to(dtype)

    # Inpaint channels: zeros (reserved for future)
    inpaint = torch.zeros((bs, 16, h, w), device=device, dtype=dtype)

    # Concatenate: [control(16), mask(4), inpaint(16)] = 36
    control_context = torch.cat([control, mask, inpaint], dim=1)

    return control_context


def encode_all_poses(vae, pose_images, pose_types, fade_mode="none", fade_strength=0.5):
    """Encode all pose types and build control contexts.

    Args:
        vae: ComfyUI VAE object
        pose_images: dict of {pose_type: tensor[B, H, W, C]}
        pose_types: list of pose types to process
        fade_mode: Spatial fade mode for all poses
        fade_strength: Fade strength for all poses

    Returns:
        list of (pose_type, control_context) tuples
    """
    contexts = []

    for pose_type in pose_types:
        if pose_type not in pose_images:
            logger.warning(f"Missing pose type: {pose_type} (skipping)")
            continue

        pose_image = pose_images[pose_type]
        logger.info(f"Encoding pose type: {pose_type}")

        # Encode to 16-channel latent
        pose_latent = encode_pose_image(vae, pose_image)

        # Build 36-dim control context
        context = build_control_context(pose_latent, fade_mode, fade_strength)

        contexts.append((pose_type, context))
        logger.info(f"Encoded {pose_type}: {pose_image.shape} -> {context.shape}")

    if not contexts:
        logger.warning("No pose types were encoded")

    return contexts
