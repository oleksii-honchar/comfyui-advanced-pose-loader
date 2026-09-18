"""Multi-pose encoder for FLUX.2-dev-Fun-Controlnet-Union.

Handles loading, resizing, and encoding multiple pose types from a pose folder
into the FLUX.2 Fun ControlNet control context format.

Uses FLUX.2 VAE (16-channel encoder) with spatial patchification to produce
128-channel latents at H/16, W/16 resolution, matching the reference implementation.
Builds a 260-dim 3D control context [control(128), mask(4), inpaint(128)] with
flattened spatial dimensions [B, seq, 260].

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
    """VAE-encode a pose image to 128-channel latent (flattened to 3D).

    Uses FLUX.2 VAE (16-channel encoder) and patchifies the output to 128 channels
    by rearranging 2x2 spatial patches. This matches the reference implementation's
    approach of using the main workflow's VAE for control encoding.

    Args:
        vae: ComfyUI VAE object (FLUX.2 VAE, 16-channel encoder)
        pose_image: Tensor [B, H, W, C] in [0, 1] range

    Returns:
        3D tensor [B, seq, 128] where seq = (H/16) * (W/16)
    """
    mm = _get_comfy_model_management()
    device = mm.get_torch_device()

    # Normalize to [0, 1] if needed
    if pose_image.max() > 1.0:
        pose_image = pose_image / 255.0

    with torch.no_grad():
        # FLUX.2 VAE produces 16-channel latents at H/8, W/8
        pose_latent = vae.encode(pose_image.to(device))

    # Patchify: rearrange 2x2 spatial patches to expand channels 16->128
    # [B, 16, H/8, W/8] -> [B, 128, H/16, W/16]
    pose_patched = _patchify(pose_latent)

    # Flatten spatial dimensions: [B, 128, H/16, W/16] -> [B, seq, 128]
    pose_flat = pose_patched.flatten(2).permute(0, 2, 1)

    return pose_flat


def _patchify(x):
    """Convert [B, C, H, W] -> [B, C*4, H/2, W/2] by rearranging 2x2 patches.

    This matches the reference implementation's patchification step that
    converts 16-channel FLUX.2 VAE latents to 128 channels.
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, c * 4, h // 2, w // 2)
    return x


def build_control_context(latent, fade_mode="none", fade_strength=0.5):
    """Build 260-dim 3D control context [control(128), mask(4), inpaint(128)].

    Args:
        latent: Flattened 128-channel latent tensor [B, seq, 128]
        fade_mode: Spatial fade mode ('none', 'top', 'bottom', 'left', 'right')
        fade_strength: Fade strength (0.0-1.0)

    Returns:
        260-channel 3D control context tensor [B, seq, 260]
    """
    device = latent.device
    dtype = latent.dtype
    bs, seq, _ = latent.shape

    # Control channels: the flattened 128-channel latent
    control = latent

    # Mask channels: spatial fade or ones (4 channels)
    if fade_mode == "none" or fade_strength <= 0.0:
        mask = torch.ones((bs, seq, 4), device=device, dtype=dtype)
    else:
        # Infer spatial dimensions from sequence length (assume square)
        spatial = int(seq ** 0.5)
        fade_mask = generate_spatial_fade_mask(spatial, spatial, fade_mode, fade_strength)
        # Flatten fade mask to 3D: [B, 1, H, W] -> [B, seq, 1]
        fade_mask_3d = fade_mask.unsqueeze(0).flatten(2).permute(0, 2, 1)
        mask = fade_mask_3d.expand(bs, seq, 4).to(dtype)

    # Inpaint channels: zeros (reserved for future)
    inpaint = torch.zeros((bs, seq, 128), device=device, dtype=dtype)

    # Concatenate: [control(128), mask(4), inpaint(128)] = 260
    control_context = torch.cat([control, mask, inpaint], dim=2)

    return control_context


def encode_all_poses(vae, pose_images, pose_types, fade_mode="none", fade_strength=0.5):
    """Encode all pose types and build 260-channel 3D control contexts.

    Args:
        vae: ComfyUI VAE object (128-channel encoder)
        pose_images: dict of {pose_type: tensor[B, H, W, C]}
        pose_types: list of pose types to process
        fade_mode: Spatial fade mode for all poses
        fade_strength: Fade strength for all poses

    Returns:
        list of (pose_type, control_context) tuples
        Each control_context is a 260-channel 3D tensor [B, seq, 260]
    """
    contexts = []

    for pose_type in pose_types:
        if pose_type not in pose_images:
            logger.warning(f"Missing pose type: {pose_type} (skipping)")
            continue

        pose_image = pose_images[pose_type]
        logger.info(f"Encoding pose type: {pose_type}")

        # Encode to 128-channel flattened latent [B, seq, 128]
        pose_latent = encode_pose_image(vae, pose_image)

        # Build 260-dim 3D control context
        context = build_control_context(pose_latent, fade_mode, fade_strength)

        contexts.append((pose_type, context))
        logger.info(f"Encoded {pose_type}: {pose_image.shape} -> {context.shape}")

    if not contexts:
        logger.warning("No pose types were encoded")

    return contexts
