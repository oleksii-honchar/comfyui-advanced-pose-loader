"""Main AdvancedOpenposeLoader node class."""
from __future__ import annotations

import torch
import comfy.utils

# Import src modules
from .utils import get_strength_defaults, cleanup_pose
from .vae_encoder import load_vae, encode_pose_image
from .pose_loader import load_pose_from_file
from .controlnet import load_controlnet
from .spatial_fade import create_spatial_fade_mask, apply_spatial_fade_to_control_signal

CONTROLNET_NODE_CLASS_NAME = "AdvancedOpenposeLoader"
POSE_LOADER_NODE_CLASS_NAME = "AdvancedPoseLoader"


class AdvancedOpenposeLoader:
    """Advanced OpenPose ControlNet with spatial fade masking."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net": ("CONTROL_NET",),
                "pose_image": ("IMAGE",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_begin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_spatial_fade": ("BOOLEAN", {"default": False}),
                "fade_ratio": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "fade_x": ("BOOLEAN", {"default": True}),
                "fade_y": ("BOOLEAN", {"default": True}),
                "body_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hand_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_pose_conditioning"
    CATEGORY = "conditioning/controlnet"

    def apply_pose_conditioning(
        self,
        model,
        control_net,
        pose_image,
        strength=None,
        strength_begin=None,
        strength_end=None,
        use_spatial_fade=False,
        fade_ratio=0.1,
        fade_x=True,
        fade_y=True,
        body_strength=None,
        face_strength=None,
        hand_strength=None,
    ):
        # Get original strength values for restoration
        orig_strength = control_net.strength
        orig_strength_begin = control_net.strength_begin
        orig_strength_end = control_net.strength_end

        # Use provided or default strengths
        if strength is None:
            strength, strength_begin, strength_end, body_strength, face_strength, hand_strength = get_strength_defaults(model)
        if strength_begin is None:
            strength_begin = 0.0
        if strength_end is None:
            strength_end = 1.0

        # Encode pose image to latent space
        pose_latent = encode_pose_image(model, pose_image, strength=strength)

        # Build conditioning with pose control
        conditioning = []
        for item in model["conds"]:
            new_item = item.copy()
            new_item["control"] = pose_latent
            conditioning.append(new_item)

        # Apply spatial fade if requested
        if use_spatial_fade:
            height, width = pose_image.shape[1], pose_image.shape[2]
            fade_mask = create_spatial_fade_mask(height, width, fade_ratio)
            for item in conditioning:
                if "control" in item:
                    item["control"] = apply_spatial_fade_to_control_signal(item["control"], fade_mask)

        return (conditioning,)