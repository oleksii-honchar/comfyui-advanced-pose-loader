"""VAE encoding for pose images.

Encodes pose images to latent space using the VAE and prepares them for
FLUX.2 Fun ControlNet integration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils


def patchify(x):
    """Convert [B, C, H, W] -> [B, C*4, H/2, W/2] by rearranging 2x2 patches.
    
    This is required for FLUX.2 Fun ControlNet's control context format.
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, c * 4, h // 2, w // 2)
    return x


def encode_pose_image(vae, pose_image, strength=0.8):
    """Encode pose image to latent space for FLUX.2 Fun ControlNet.
    
    Args:
        vae: ComfyUI VAE object
        pose_image: Pose image tensor [B, H, W, C]
        strength: Control strength (not used in encoding, just metadata)
    
    Returns:
        torch.Tensor: Encoded pose latent tensor [1, 260, H/16, W/16]
    """
    device = comfy.model_management.get_torch_device()
    dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16
    
    # Normalize pose image to [0, 1]
    if pose_image.max() > 1.0:
        pose_image = pose_image / 255.0
    
    # Encode pose to latent space using VAE
    with torch.no_grad():
        pose_latent = vae.encode(pose_image.to(device))
    
    # Reshape for FLUX.2 Fun ControlNet (128 channels at h/16, w/16)
    pose_flat = pose_latent.to(device=device, dtype=dtype).flatten(2).permute(0, 2, 1)
    
    # Build control context: [pose(128), mask(4), inpaint(128)] = 260
    bs, seq, _ = pose_flat.shape
    mask_flat = torch.zeros((bs, seq, 4), device=device, dtype=dtype)
    inpaint_flat = torch.zeros((bs, seq, 128), device=device, dtype=dtype)
    control_context = torch.cat([pose_flat, mask_flat, inpaint_flat], dim=2)
    
    return control_context