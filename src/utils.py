"""Utility functions for the AdvancedOpenposeLoader node."""
from __future__ import annotations
import torch


def get_strength_defaults(model):
    """Get default strength values based on model architecture."""
    # Flux-specific defaults for better pose adherence
    # These values work better with Flux's multi-stage architecture
    strength = 0.75
    strength_begin = 0.0
    strength_end = 1.0
    # Flux-specific: use higher default for body region
    body_strength = 0.9
    face_strength = 0.7
    hand_strength = 0.7
    return (strength, strength_begin, strength_end, body_strength, face_strength, hand_strength)


def cleanup_pose(control_net, strength, strength_begin, strength_end, pose_conditioning,
                 body_strength=None, face_strength=None, hand_strength=None):
    """Cleanup function to restore original ControlNet values after sampling."""
    # Restore original strength values
    control_net.strength = strength
    control_net.strength_begin = strength_begin
    control_net.strength_end = strength_end
    # Remove pose_conditioning from conditioning items
    if pose_conditioning is not None:
        for item in pose_conditioning:
            item["control"] = None
    return None