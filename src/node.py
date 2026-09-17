"""Main AdvancedOpenposeLoader node class.

Uses the FLUX.2 Fun ControlNet mechanism by importing and using the working
Flux2FunControlNetLoader and Flux2FunControlNetApply nodes from the local
reference implementation at /Volumes/Data/www/misc/comfyui-flux2fun-controlnet/.
"""
from __future__ import annotations

import sys
import os

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the working Flux2 Fun ControlNet implementation from local reference
sys.path.insert(0, "/Volumes/Data/www/misc/comfyui-flux2fun-controlnet")
from nodes import Flux2FunControlNetLoader, Flux2FunControlNetApply


class AdvancedOpenposeLoader:
    """Advanced OpenPose ControlNet with spatial fade masking.

    Uses the same FLUX.2 Fun ControlNet mechanism as the working photo-pose-use
    workflow by delegating to Flux2FunControlNetLoader and Flux2FunControlNetApply.
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
    ):
        """Apply pose conditioning using FLUX.2 Fun ControlNet mechanism.
        
        Delegates to Flux2FunControlNetLoader and Flux2FunControlNetApply for
        proper control signal generation and application.
        """
        import comfy.utils
        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        
        # Get strength defaults if not provided
        if strength is None:
            strength = 0.8

        # Resize pose image to target dimensions if specified
        resized_pose = pose_image
        if target_width > 0 and target_height > 0:
            resized_pose = comfy.utils.resize_image(pose_image, target_width, target_height, "lanczos")

        # Load controlnet using Flux2FunControlNetLoader
        loader = Flux2FunControlNetLoader()
        (controlnet,) = loader.load_controlnet(controlnet_name)

        # Apply controlnet using Flux2FunControlNetApply
        applier = Flux2FunControlNetApply()
        (positive_cond,) = applier.apply_controlnet(
            conditioning,
            controlnet,
            vae,
            strength,
            control_image=resized_pose,
            mask=None,
            inpaint_image=None
        )

        # Negative conditioning without pose
        negative_cond = conditioning

        print(f"[AdvancedOpenposeLoader] Applied pose control (strength={strength})")

        # Return model (unchanged), positive conditioning with pose, negative conditioning
        return (model, positive_cond, negative_cond)