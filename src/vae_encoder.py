"""VAE loading and pose image encoding utilities."""
from __future__ import annotations
import torch
import comfy.utils
import comfy.sd


def load_vae(vae_path):
    """Load a VAE model from the given path."""
    if not vae_path or not vae_path.strip():
        return None
    vae_path = vae_path.strip()
    if vae_path.endswith(".safetensors"):
        vae = comfy.sd.load_vae(vae_path)
    else:
        raise ValueError(f"Unsupported VAE file type: {vae_path}. Use .safetensors.")
    return vae


def encode_pose_image(vae, pose_image, strength=0.8):
    """Encode a pose image to latent space using the VAE."""
    # Ensure pose_image is a tensor with correct shape
    if not isinstance(pose_image, torch.Tensor):
        raise ValueError(f"pose_image must be a tensor, got {type(pose_image)}")

    # If pose_image is [C, H, W] (no batch), add batch dimension
    if pose_image.dim() == 3:
        pose_image = pose_image.unsqueeze(0)

    # Move to same device as VAE
    device = next(vae.parameters()).device
    pose_image = pose_image.to(device)

    # Encode to latent space
    pose_latent = vae.encode(pose_image)

    # Adjust latent strength
    if hasattr(pose_latent, 'mean') and hasattr(pose_latent, 'std'):
        # Standardize latent values
        mean = pose_latent.mean().item()
        std = pose_latent.std().item()
        if std > 0:
            pose_latent = (pose_latent - mean) / std
        pose_latent = pose_latent * strength
    else:
        pose_latent = pose_latent * strength

    return pose_latent