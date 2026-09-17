import logging
import os
import torch
import folder_paths

from src.pose_loader import resolve_pose_folder, list_pose_images, load_pose_image
from src.pose_preprocessor import resize_to_1024
from src.multi_pose_encoder import encode_pose_image, build_control_context
from src.flux2fun_integration import (
    patch_transformer_for_control,
    cleanup_control_patch,
)

logger = logging.getLogger(__name__)

POSE_TYPES = [
    "openpose",
    "openpose_hand",
    "openpose_full",
    "canny",
    "depth",
    "normal",
]

DEFAULT_STRENGTHS = {
    "openpose": 0.75,
    "openpose_hand": 0.80,
    "openpose_full": 0.85,
    "canny": 0.0,
    "depth": 0.0,
    "normal": 0.0,
}


class AdvancedOpenposeLoader:
    """Advanced pose loader with FLUX.2 Fun Control integration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "conditioning": ("CONDITIONING", {}),
                "vae": ("VAE", {}),
                "controlnet": ("FLUX2_FUN_CONTROLNET", {}),
                "pose_folder_name": ("STRING", {
                    "default": "",
                }),
            },
            "optional": {
                "openpose_strength": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "openpose_hand_strength": ("FLOAT", {
                    "default": 0.80, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "openpose_full_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "canny_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "depth_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "normal_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "spatial_fade": ("BOOLEAN", {
                    "default": False,
                }),
                "spatial_fade_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "AdvancedPoseLoader"

    def execute(self, model, conditioning, pose_folder_name, vae, controlnet,
                openpose_strength=0.75, openpose_hand_strength=0.80,
                openpose_full_strength=0.85, canny_strength=0.0, depth_strength=0.0,
                normal_strength=0.0, spatial_fade=False, spatial_fade_strength=1.0,
                debug=False):
        """Execute the advanced pose loading pipeline."""
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Starting pipeline for folder: {pose_folder_name}")

        # Step 1: Resolve pose folder path
        pose_folder = resolve_pose_folder(pose_folder_name)
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Pose folder: {pose_folder}")

        # Step 2: Load pose images
        available_images = list_pose_images(pose_folder)
        pose_images = {}
        for pose_type in POSE_TYPES:
            if pose_type in available_images:
                pose_images[pose_type] = load_pose_image(available_images[pose_type])

        if not pose_images:
            raise ValueError(f"No pose images found in {pose_folder}")

        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Loaded {len(pose_images)} pose types")

        # Step 3: Resize to 1024x1024
        processed_images = {}
        for pose_type, image in pose_images.items():
            if debug:
                logger.info(f"[AdvancedOpenposeLoader] Resizing {pose_type} to 1024x1024")
            processed_images[pose_type] = resize_to_1024(image)

        # Step 4: Use provided VAE and encode
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Encoding with provided VAE")
        vae_model = vae

        # Build control contexts
        control_contexts = {}
        strengths = {
            "openpose": openpose_strength,
            "openpose_hand": openpose_hand_strength,
            "openpose_full": openpose_full_strength,
            "canny": canny_strength,
            "depth": depth_strength,
            "normal": normal_strength,
        }

        for pose_type, pose_image in processed_images.items():
            if debug:
                logger.info(f"[AdvancedOpenposeLoader] Encoding pose: {pose_type}")
            latent = encode_pose_image(vae_model, pose_image)
            context = build_control_context(
                latent,
                fade_mode=spatial_fade,
                fade_strength=spatial_fade_strength
            )
            control_contexts[pose_type] = context

        # Step 5: Use provided ControlNet model (reused for all pose types)
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Using provided ControlNet")

        # Step 6: Apply FLUX.2 Fun Control via transformer patching
        self._register_control_contexts(controlnet, control_contexts, strengths)

        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Pipeline complete")

        return (model, conditioning, conditioning)

    def _register_control_contexts(self, controlnet, control_contexts, strengths):
        """Register control contexts for this sampling run."""
        from comfy.model_patcher import ModelPatcher

        patch = {
            "controlnet": controlnet,
            "contexts": control_contexts,
            "strengths": strengths,
        }

        def apply_patch(model, *args, **kwargs):
            patch_transformer_for_control(
                model, patch["controlnet"], None,
                strength=patch["strengths"]["openpose"]
            )
            return model

        def remove_patch(model, *args, **kwargs):
            cleanup_control_patch()
            return model

        patch_id = "advanced_openpose_loader"
        ModelPatcher.get_all_model_patches()[patch_id] = (apply_patch, remove_patch)


NODE_CLASS_MAPPINGS = {
    "AdvancedOpenposeLoader": AdvancedOpenposeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedOpenposeLoader": "Advanced Pose Loader",
}
