"""Main AdvancedOpenposeLoader node class.

Uses the FLUX.2 Fun ControlNet mechanism (same as photo-pose-use workflow) to properly
apply pose conditioning to FLUX.2 models. Creates a ControlNetWrapper that accumulates
control signals in transformer_options, which are processed by the FLUX.2 Fun patch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import comfy.utils
import comfy.model_management

from .utils import get_strength_defaults
from .vae_encoder import encode_pose_image
from .pose_loader import load_pose_from_file
from .controlnet import load_controlnet, ControlNetWrapper
from .spatial_fade import create_spatial_fade_mask, apply_spatial_fade_to_control_signal


class AdvancedOpenposeLoader:
    """Advanced OpenPose ControlNet with spatial fade masking.

    Uses the FLUX.2 Fun ControlNet mechanism to properly apply pose conditioning
    to FLUX.2 models (both 4B and 9B variants).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "controlnet_name": ("COMBO", {
                    "options": ["FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors",
                                "FLUX.2-dev-Fun-Controlnet-Pose-2602-fp8.safetensors"]
                }),
                "pose_image": ("IMAGE",),
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "use_spatial_fade": ("BOOLEAN", {"default": False}),
                "fade_ratio": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "body_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hand_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply_pose_conditioning"
    CATEGORY = "conditioning/controlnet"

    def apply_pose_conditioning(
        self,
        model,
        conditioning,
        vae,
        controlnet_name,
        pose_image,
        target_width=0,
        target_height=0,
        strength=None,
        use_spatial_fade=False,
        fade_ratio=0.1,
        body_strength=None,
        face_strength=None,
        hand_strength=None,
    ):
        """Apply pose conditioning using FLUX.2 Fun ControlNet mechanism.
        
        Encodes the pose image to latent space, builds the 260-channel control context,
        creates a ControlNetWrapper, and sets it in the conditioning with
        control_apply_to_uncond=True (same as Flux2FunControlNetApply).
        """
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16
        
        # Get strength defaults if not provided
        if strength is None:
            strength, _, _, body_strength, face_strength, hand_strength = get_strength_defaults(model)
        
        # Resize pose image to target dimensions if specified
        resized_pose = pose_image
        if target_width > 0 and target_height > 0:
            resized_pose = comfy.utils.resize_image(pose_image, target_width, target_height, "lanczos")
        
        # Load FLUX.2 Fun ControlNet model
        print("[AdvancedOpenposeLoader] Loading controlnet model...")
        controlnet = load_controlnet(controlnet_name)
        
        # Encode pose image to latent space and build 260-channel control context
        print("[AdvancedOpenposeLoader] Encoding pose to latent space...")
        control_context = encode_pose_image(vae, resized_pose, strength)
        
        # Detect low VRAM mode
        try:
            from comfy.model_management import vram_state, VRAMState
            low_vram = vram_state in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM)
        except (ImportError, AttributeError):
            low_vram = False
        
        print(f"[AdvancedOpenposeLoader] Mode: pose control, strength: {strength}, low_vram: {low_vram}")
        
        # Create ControlNetWrapper (same pattern as Flux2FunControlNetApply)
        wrapper = ControlNetWrapper(controlnet, control_context, strength, low_vram)
        
        # Apply spatial fade if requested
        if use_spatial_fade:
            height, width = resized_pose.shape[1], resized_pose.shape[2]
            fade_mask = create_spatial_fade_mask(height, width, fade_ratio)
            control_context = apply_spatial_fade_to_control_signal(control_context, fade_mask)
            wrapper.control_context = control_context
        
        # Build conditioning with pose control
        # Same pattern as Flux2FunControlNetApply: set control and control_apply_to_uncond
        positive_cond = []
        negative_cond = []
        
        for item in conditioning:
            # Apply pose conditioning to positive
            positive_item = item.copy()
            existing_control = positive_item.get('control', None)
            if existing_control is not None:
                wrapper.previous_controlnet = existing_control
            positive_item['control'] = wrapper
            positive_item['control_apply_to_uncond'] = True
            positive_cond.append(positive_item)
            
            # Negative conditioning without pose
            negative_cond.append(item.copy())
        
        print(f"[AdvancedOpenposeLoader] Applied pose control (strength={strength})")
        
        # Return model (unchanged), positive conditioning with pose, negative conditioning
        return (model, positive_cond, negative_cond)