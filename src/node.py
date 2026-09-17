"""Main AdvancedOpenposeLoader node class.

Uses the FLUX.2 Fun ControlNet mechanism by delegating to the working
Flux2FunControlNetLoader and Flux2FunControlNetApply nodes.

This node loads pose images, generates ControlNet conditioning for FLUX.2
diffusion models, and supports spatial fade masking for natural blending.
"""
from __future__ import annotations

import os
import sys
import torch

class AdvancedOpenposeLoader:
    """Advanced OpenPose ControlNet with spatial fade masking.

    Uses the same FLUX.2 Fun ControlNet mechanism as the working photo-pose-use
    workflow by delegating to Flux2FunControlNetLoader and Flux2FunControlNetApply.

    Supports single pose image or multiple poses (batch) with spatial fade
    control (vertical/horizontal gradients, corners).
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
                "fade_top": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade_bottom": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade_left": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade_right": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply_pose_conditioning"
    CATEGORY = "conditioning/controlnet"

    def _generate_spatial_fade_mask(self, image, fade_top, fade_bottom, fade_left, fade_right):
        """Generate a spatial fade mask where 1.0 = full strength, 0.0 = no strength."""
        b, h, w, _ = image.shape
        mask = torch.ones((1, 1, h, w), dtype=image.dtype, device=image.device)

        if fade_top > 0:
            top_range = int(h * fade_top)
            top_grad = torch.linspace(0.0, 1.0, top_range, device=image.device, dtype=image.dtype)
            mask[:, :, :top_range, :] = top_grad.view(-1, 1)

        if fade_bottom > 0:
            bottom_range = int(h * fade_bottom)
            bottom_grad = torch.linspace(1.0, 0.0, bottom_range, device=image.device, dtype=image.dtype)
            mask[:, :, h - bottom_range:, :] = bottom_grad.view(-1, 1)

        if fade_left > 0:
            left_range = int(w * fade_left)
            left_grad = torch.linspace(0.0, 1.0, left_range, device=image.device, dtype=image.dtype)
            mask[:, :, :, :left_range] = left_grad.view(1, -1)

        if fade_right > 0:
            right_range = int(w * fade_right)
            right_grad = torch.linspace(1.0, 0.0, right_range, device=image.device, dtype=image.dtype)
            mask[:, :, :, w - right_range:] = right_grad.view(1, -1)

        return mask

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
        fade_top=0.0,
        fade_bottom=0.0,
        fade_left=0.0,
        fade_right=0.0,
    ):
        """Apply pose conditioning using FLUX.2 Fun ControlNet mechanism.

        Delegates to Flux2FunControlNetLoader and Flux2FunControlNetApply for
        proper control signal generation and application.

        Supports spatial fade masking for natural blending of pose conditioning.
        """
        import comfy.utils
        import comfy.model_management

        if strength is None:
            strength = 0.8

        # Lazy import of Flux2Fun nodes at runtime (not module load time)
        # so they work in the ComfyUI environment with all dependencies
        flux2fun_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comfyui-flux2fun-controlnet")
        if not os.path.exists(flux2fun_path):
            flux2fun_path = "/home/tuiteraz/behemoth-lan/comfyui/data/ComfyUI/custom_nodes/comfyui-flux2fun-controlnet"

        sys.path.insert(0, flux2fun_path)
        from nodes import Flux2FunControlNetLoader, Flux2FunControlNetApply

        # Resize pose image to target dimensions if specified
        resized_pose = pose_image
        if target_width > 0 and target_height > 0:
            resized_pose = comfy.utils.resize_image(pose_image, target_width, target_height, "lanczos")

        # Apply spatial fade mask if any fade parameters are set
        if fade_top > 0 or fade_bottom > 0 or fade_left > 0 or fade_right > 0:
            fade_mask = self._generate_spatial_fade_mask(
                resized_pose, fade_top, fade_bottom, fade_left, fade_right
            )
            # Apply mask to pose image (broadcast mask over batch)
            mask_expanded = fade_mask.expand_as(resized_pose)
            faded_pose = resized_pose * mask_expanded
            print(f"[AdvancedOpenposeLoader] Applied spatial fade mask (top={fade_top}, bottom={fade_bottom}, left={fade_left}, right={fade_right})")
        else:
            faded_pose = resized_pose

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
            control_image=faded_pose,
            mask=None,
            inpaint_image=None
        )

        # Negative conditioning without pose
        negative_cond = conditioning

        print(f"[AdvancedOpenposeLoader] Applied pose control (strength={strength}, controlnet={controlnet_name})")

        # Return model (unchanged), positive conditioning with pose, negative conditioning
        return (model, positive_cond, negative_cond)