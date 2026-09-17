"""Spatial fade mask creation and application."""
from __future__ import annotations
import torch
import comfy.utils


def create_spatial_fade_mask(height, width, fade_ratio):
    """Create a fade mask that transitions from 1 (full strength) to 0 (no strength)."""
    # Create horizontal gradient
    x = torch.linspace(0, 1, width)
    x_mask = 1.0 - x
    x_mask = x_mask.unsqueeze(0).repeat(height, 1)

    # Create vertical gradient
    y = torch.linspace(0, 1, height)
    y_mask = 1.0 - y
    y_mask = y_mask.unsqueeze(1).repeat(1, width)

    # Combine with minimum (both must be strong)
    mask = torch.minimum(x_mask, y_mask)

    # Normalize and fade
    mask = torch.clamp((mask - (1.0 - fade_ratio)) / fade_ratio, 0, 1)

    # Convert to float tensor
    mask = mask.to(torch.float32).unsqueeze(0)  # Add channel dimension

    return mask


def apply_spatial_fade_to_control_net(control_net, control_signal, fade_mask):
    """Apply spatial fade mask to ControlNet control signal."""
    # Expand mask to match control signal dimensions
    mask = fade_mask.expand_as(control_signal)
    # Apply mask
    return control_signal * mask


def apply_spatial_fade_to_control_signal(control_signal, fade_mask):
    """Apply spatial fade mask to a control signal."""
    # Ensure mask has the right shape
    if fade_mask.dim() == 2:
        fade_mask = fade_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Expand mask to match control signal
    mask = fade_mask.expand_as(control_signal)
    return control_signal * mask