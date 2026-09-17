"""AdvancedOpenposeLoader node with redesigned interface.

Loads multiple pose images from a folder, encodes them with VAE,
builds FLUX.2 Fun ControlNet control contexts, and applies pose
conditioning via monkey-patching.

Interface:
  Required inputs (top):
    - vae: VAE model dropdown
    - model: base model
    - conditioning: conditioning input
    - controlnet: ControlNet model dropdown
    - folder_name: pose folder name
    - strength_openpose: strength slider (default 0.75)
    - strength_openpose_hand: strength slider (default 0.80)
    - strength_openpose_full: strength slider (default 0.85)
    - strength_canny: strength slider (default 0.0)
    - strength_depth: strength slider (default 0.0)
    - strength_normal: strength slider (default 0.0)

  Optional inputs (bottom):
    - spatial_fade: dropdown (none/top/bottom/left/right)
    - spatial_fade_strength: slider (default 0.5)
    - debug: boolean toggle

  Outputs:
    - model: unchanged model
    - positive: conditioned conditioning
    - negative: same as positive
"""
from __future__ import annotations

import logging
import torch
import comfy.utils
import comfy.model_management
import folder_paths

from src.pose_loader import resolve_pose_folder, list_pose_images, load_pose_image
from src.pose_preprocessor import resize_to_1024
from src.multi_pose_encoder import encode_pose_image, build_control_context
from src.flux2fun_integration import (
    apply_controlnet_model,
    patch_transformer_for_control,
    unpatch_transformer,
    build_control_chain,
)

logger = logging.getLogger(__name__)

# Six pose types supported
POSE_TYPES = [
    "openpose",
    "openpose_hand",
    "openpose_full",
    "canny",
    "depth",
    "normal",
]

# Default strengths for each pose type
DEFAULT_STRENGTHS = {
    "openpose": 0.75,
    "openpose_hand": 0.80,
    "openpose_full": 0.85,
    "canny": 0.0,
    "depth": 0.0,
    "normal": 0.0,
}


def get_vae_options():
    """Auto-discover available VAE models."""
    try:
        options = list(folder_paths.get_filename_list("vae"))
    except Exception:
        options = []
    # Ensure flux2-vae.safetensors is always an option (default for FLUX.2)
    if "flux2-vae.safetensors" not in options:
        options.append("flux2-vae.safetensors")
    return options, "flux2-vae.safetensors"


def get_controlnet_options():
    """Auto-discover available ControlNet models."""
    try:
        options = list(folder_paths.get_filename_list("controlnet"))
    except Exception:
        options = []
    # Ensure at least one FLUX.2 Fun ControlNet option is available
    if not any("fun" in name.lower() for name in options):
        options.append("FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors")
    return options, options[0]


class AdvancedOpenposeLoader:
    """Advanced pose conditioning with FLUX.2 Fun ControlNet."""

    def __init__(self):
        self.loaded_vae = None
        self.loaded_controlnets = []
        self.active_patches = []

    @classmethod
    def INPUT_TYPES(cls):
        vae_options, vae_default = get_vae_options()
        cn_options, cn_default = get_controlnet_options()

        return {
            "required": {
                "vae": ("COMBO", {"options": vae_options, "default": vae_default}),
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "controlnet": ("COMBO", {"options": cn_options, "default": cn_default}),
                "folder_name": ("STRING", {"default": ""}),
                "strength_openpose": ("FLOAT", {
                    "default": DEFAULT_STRENGTHS["openpose"],
                    "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "strength_openpose_hand": ("FLOAT", {
                    "default": DEFAULT_STRENGTHS["openpose_hand"],
                    "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "strength_openpose_full": ("FLOAT", {
                    "default": DEFAULT_STRENGTHS["openpose_full"],
                    "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "strength_canny": ("FLOAT", {
                    "default": DEFAULT_STRENGTHS["canny"],
                    "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "strength_depth": ("FLOAT", {
                    "default": DEFAULT_STRENGTHS["depth"],
                    "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "strength_normal": ("FLOAT", {
                    "default": DEFAULT_STRENGTHS["normal"],
                    "min": 0.0, "max": 1.0, "step": 0.01
                }),
            },
            "optional": {
                "spatial_fade": ("COMBO", {
                    "options": ["none", "top", "bottom", "left", "right"],
                    "default": "none"
                }),
                "spatial_fade_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply_pose_conditioning"
    CATEGORY = "conditioning/controlnet"

    def apply_pose_conditioning(
        self,
        vae_name,
        model,
        conditioning,
        controlnet_name,
        folder_name,
        strength_openpose,
        strength_openpose_hand,
        strength_openpose_full,
        strength_canny,
        strength_depth,
        strength_normal,
        spatial_fade="none",
        spatial_fade_strength=0.5,
        debug=False,
    ):
        """Apply multi-pose conditioning using FLUX.2 Fun ControlNet."""
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Starting pipeline:")
            logger.info(f"  vae={vae_name}")
            logger.info(f"  controlnet={controlnet_name}")
            logger.info(f"  folder={folder_name}")
            logger.info(f"  strengths: openpose={strength_openpose}, hand={strength_openpose_hand}, full={strength_openpose_full}, canny={strength_canny}, depth={strength_depth}, normal={strength_normal}")
            logger.info(f"  spatial_fade={spatial_fade} (strength={spatial_fade_strength})")
            logger.info(f"  debug={debug}")

        # Step 1: Load VAE
        device = comfy.model_management.get_torch_device()
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Loading VAE: {vae_path}")
        if self.loaded_vae is None or self.loaded_vae._vae_file != vae_path:
            self.loaded_vae = comfy.sd.VAE()
            self.loaded_vae.load_vae(vae_path)
        vae = self.loaded_vae.vae

        # Step 2: Resolve pose folder and list pose images
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Loading pose images from: {folder_name}")
        pose_folder = resolve_pose_folder(folder_name)
        pose_images = list_pose_images(pose_folder)

        # Step 3: Load and preprocess each pose image
        processed_images = {}
        for pose_type in POSE_TYPES:
            if pose_type not in pose_images:
                if debug:
                    logger.info(f"[AdvancedOpenposeLoader] Missing pose image: {pose_type}")
                continue
            if debug:
                logger.info(f"[AdvancedOpenposeLoader] Loading pose image: {pose_type}")
            pose_image = load_pose_image(pose_images[pose_type])
            pose_image = resize_to_1024(pose_image)
            processed_images[pose_type] = pose_image

        # Step 4: Encode each pose image to control context
        control_contexts = {}
        strengths = {
            "openpose": strength_openpose,
            "openpose_hand": strength_openpose_hand,
            "openpose_full": strength_openpose_full,
            "canny": strength_canny,
            "depth": strength_depth,
            "normal": strength_normal,
        }

        for pose_type, pose_image in processed_images.items():
            if debug:
                logger.info(f"[AdvancedOpenposeLoader] Encoding pose: {pose_type}")
            latent = encode_pose_image(vae, pose_image)
            context = build_control_context(
                latent,
                fade_mode=spatial_fade,
                fade_strength=spatial_fade_strength
            )
            control_contexts[pose_type] = context

        # Step 5: Load ControlNet models (one per pose type)
        controlnet_models = []
        for pose_type in POSE_TYPES:
            if pose_type not in control_contexts:
                continue
            if debug:
                logger.info(f"[AdvancedOpenposeLoader] Loading ControlNet for: {pose_type}")
            cn_path = folder_paths.get_full_path("controlnet", controlnet_name)
            controlnet = apply_controlnet_model(model, cn_path)
            controlnet_models.append((pose_type, controlnet))

        # Step 6: Build control chain with individual strengths
        hints = [control_contexts[pose_type] for pose_type, _ in controlnet_models]
        strengths_list = [strengths[pose_type] for pose_type, _ in controlnet_models]
        model_list = [cn for _, cn in controlnet_models]

        # Step 7: Chain controlnets and apply conditioning
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Chaining {len(controlnet_models)} controlnets")
        wrappers = build_control_chain(model_list, hints, strengths_list)

        # Step 8: Apply control via transformer patching
        model_obj = model.model if hasattr(model, 'model') else model
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Patching transformer for control")
        original_forward = patch_transformer_for_control(
            model_obj,
            model_list[0],
            hints[0] if hints else None,
            strengths_list[0] if strengths_list else 0.75
        )
        self.active_patches.append((model_obj, original_forward))

        # Return model (unchanged), positive conditioning, negative conditioning
        positive_cond = conditioning
        negative_cond = conditioning

        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Pipeline complete")

        return (model, positive_cond, negative_cond)

    def cleanup(self):
        """Clean up active patches and release resources."""
        for model_obj, original_forward in self.active_patches:
            unpatch_transformer(model_obj, original_forward)
        self.active_patches = []
        self.loaded_vae = None
        self.loaded_controlnets = []
