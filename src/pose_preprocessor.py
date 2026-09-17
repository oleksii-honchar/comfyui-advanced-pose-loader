"""Pose image preprocessing: resize and spatial fade.

Provides resize_to_1024 and apply_spatial_fade for pose control contexts
with structure [control(16), mask(4), inpaint(16)].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_to_1024(image_tensor):
    """Resize image tensor to exactly 1024x1024 with padding.

    Maintains aspect ratio by scaling to fit within 1024x1024, then padding
    with black (0.0) to fill the remaining space. Centered.

    Uses bilinear interpolation. Handles batched images.
    Input shape: (batch, height, width, channels)
    Output shape: (batch, 1024, 1024, channels)
    """
    if image_tensor.shape[1] == 1024 and image_tensor.shape[2] == 1024:
        return image_tensor

    # ComfyUI convention is BHWC; F.interpolate expects BCTHW
    b, h, w, c = image_tensor.shape
    image = image_tensor.permute(0, 3, 1, 2)

    # Calculate scale to fit within 1024x1024 while maintaining aspect ratio
    target_size = 1024
    scale = min(target_size / h, target_size / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Scale image
    resized = F.interpolate(
        image, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Create padded canvas
    padded = torch.zeros(
        (b, c, target_size, target_size),
        device=image.device,
        dtype=image.dtype
    )

    # Calculate padding to center image
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    # Paste scaled image onto canvas
    padded[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return padded.permute(0, 2, 3, 1).to(image_tensor.dtype)


def generate_spatial_fade_mask(height, width, fade_mode="none", fade_strength=0.5):
    """Generate a spatial fade mask where 1.0 = full strength, 0.0 = no strength.

    fade_mode: 'none', 'top', 'bottom', 'left', 'right'
    fade_strength: 0.0 (no fade) to 1.0 (entire image fades)
    """
    if fade_mode == "none" or fade_strength <= 0.0:
        return torch.ones((1, 1, height, width), dtype=torch.float32)

    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32)

    # Create base gradient (0 at fade origin, 1 at opposite side)
    if fade_mode == "top":
        # Top fades out; y goes 0..1 top to bottom
        grad = y.unsqueeze(1).repeat(1, width)
    elif fade_mode == "bottom":
        # Bottom fades out; 1-y goes 1..0 top to bottom
        grad = (1.0 - y).unsqueeze(1).repeat(1, width)
    elif fade_mode == "left":
        # Left fades out; x goes 0..1 left to right
        grad = x.unsqueeze(0).repeat(height, 1)
    elif fade_mode == "right":
        # Right fades out; 1-x goes 1..0 left to right
        grad = (1.0 - x).unsqueeze(0).repeat(height, 1)
    else:
        raise ValueError(f"Unknown fade_mode: {fade_mode}")

    # grad is 0 at fade origin, 1 at far side.
    # Apply fade_strength: scale the gradient so the fade region is fade_strength wide.
    # Points where grad >= fade_strength get full strength (1.0).
    # Points where grad < fade_strength get interpolated strength (grad / fade_strength).
    mask = torch.clamp(grad / fade_strength, 0.0, 1.0)

    return mask.unsqueeze(0).unsqueeze(0)


def apply_spatial_fade(control_context, fade_mode="none", fade_strength=0.5):
    """Generate and apply spatial fade mask to control context's mask channels.

    control_context: [control(16), mask(4), inpaint(16)] = 36 channels
    Only modifies mask channels (16-19). Control and inpaint unchanged.
    """
    if fade_mode == "none":
        return control_context

    b, c, h, w = control_context.shape

    # Validate structure
    if c != 36:
        raise ValueError(
            f"Expected control context with 36 channels, got {c}"
        )

    # Generate fade mask (batch size 1)
    fade_mask = generate_spatial_fade_mask(h, w, fade_mode, fade_strength)

    # Broadcast to batch size and 4 mask channels
    mask_channels = fade_mask.expand(b, 4, h, w)

    # Construct result: control + masked + inpaint
    result = torch.cat(
        [control_context[:, :16, :, :], mask_channels, control_context[:, 20:, :, :]],
        dim=1,
    )
    return result