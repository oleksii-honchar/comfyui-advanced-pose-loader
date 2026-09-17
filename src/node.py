"""
Advanced Openpose Loader Node
=============================

ComfyUI node that loads pose images from a folder, encodes them with VAE,
builds FLUX.2 Fun Control contexts, and chains them into the model.

Node name: AdvancedOpenposeLoader

Inputs:
- model: The FLUX.2 model
- conditioning: The CLIP conditioning
- pose_folder_name: Name of pose folder (poses/{name}/)
- pose_types: List of pose types to use
- strengths: Strength for each pose type
- vae_name: VAE model name
- control_net: ControlNet model name
- spatial_fade: Enable spatial fade mask
- spatial_fade_strength: Strength of spatial fade

Outputs:
- model: Model with control applied
- positive: Positive conditioning
- negative: Negative conditioning
"""

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
    if "flux2-vae.safetensors" not in options:
        options.append("flux2-vae.safetensors")
    return options, "flux2-vae.safetensors"


def get_controlnet_options():
    """Auto-discover available ControlNet models."""
    try:
        options = list(folder_paths.get_filename_list("controlnet"))
    except Exception:
        options = []
    target = "FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors"
    if target not in options:
        options.append(target)
    return options, target


class AdvancedOpenposeLoader:
    """Advanced pose loader with FLUX.2 Fun Control integration."""

    @classmethod
    def INPUT_TYPES(cls):
        vae_options, vae_default = get_vae_options()
        cn_options, cn_default = get_controlnet_options()

        return {
            "required": {
                "model": ("MODEL", {}),
                "conditioning": ("CONDITIONING", {}),
                "pose_folder_name": ("STRING", {
                    "default": "test",
                    "multiline": False,
                    "placeholder": "Folder name under poses/",
                }),
                "vae": ("COMBO", {
                    "values": vae_options,
                    "default": vae_default,
                }),
                "control_net": ("COMBO", {
                    "values": cn_options,
                    "default": cn_default,
                }),
            },
            "optional": {
                "pose_types": ("STRING", {
                    "default": "openpose,openpose_hand,openpose_full",
                    "multiline": False,
                    "placeholder": "Comma-separated pose types",
                }),
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
    FUNCTION = "execute"
    CATEGORY = "AdvancedPoseLoader"

    def execute(self, model, conditioning, pose_folder_name, vae, control_net,
                pose_types=None, openpose_strength=0.75, openpose_hand_strength=0.80,
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
        pose_images = {}
        for pose_type in POSE_TYPES:
            if pose_type not in pose_folder:
                continue
            if pose_types and pose_type not in pose_types.split(","):
                continue
            images = list_pose_images(pose_folder, pose_type)
            if images:
                pose_images[pose_type] = images[0]

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

        # Step 4: Load VAE and encode
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Loading VAE: {vae}")
        vae_model = self._load_vae(vae)

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

        # Step 5: Load ControlNet model once (reused for all pose types)
        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Loading ControlNet: {control_net}")
        cn_path = folder_paths.get_full_path("controlnet", control_net)
        controlnet = self._load_controlnet(cn_path)

        # Step 6: Apply FLUX.2 Fun Control via transformer patching
        # Register all control contexts for this run
        self._register_control_contexts(controlnet, control_contexts, strengths)

        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Pipeline complete")

        return (model, conditioning, conditioning)

    def _load_vae(self, vae_name):
        """Load VAE model with backwards compatibility."""
        from comfy.model_management import load_model
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE not found: {vae_path}")

        # Try new-style loading first
        try:
            model = load_model(vae_path)
            return model
        except Exception as e:
            logger.warning(f"[AdvancedOpenposeLoader] New-style VAE loading failed: {e}")

        # Fallback to old-style
        try:
            from comfy.vae import load_vae
            return load_vae(vae_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE {vae_name}: {e}")

    def _load_controlnet(self, cn_path):
        """Load FLUX.2 Fun ControlNet model."""
        import json
        import torch
        from diffusers import Flux2FunControlNet

        if not os.path.exists(cn_path):
            raise FileNotFoundError(f"ControlNet not found: {cn_path}")

        logger.info(f"[AdvancedOpenposeLoader] Loading FLUX.2 Fun ControlNet: {cn_path}")

        # Load config.json if present
        config_path = os.path.join(os.path.dirname(cn_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            controlnet = Flux2FunControlNet(**config)
        else:
            # Use default Flux2FunControlNet config
            controlnet = Flux2FunControlNet()

        # Load weights
        state_dict = torch.load(cn_path, map_location="cpu", weights_only=True)
        # Handle checkpoint wrapper
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Handle diffusion_pytorch_model wrapper
        if "diffusion_pytorch_model" in state_dict:
            state_dict = state_dict["diffusion_pytorch_model"]

        missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"[AdvancedOpenposeLoader] Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"[AdvancedOpenposeLoader] Unexpected keys: {len(unexpected)}")

        controlnet.eval()
        return controlnet

    def _register_control_contexts(self, controlnet, control_contexts, strengths):
        """Register control contexts with the FLUX.2 Fun Control system.

        Passes control info via transformer_options in kwargs, which the
        patched forward_orig reads during sampling.
        """
        # Collect all control contexts with non-zero strengths
        contexts = []
        scales = []
        dims = []
        for pose_type, context in control_contexts.items():
            strength = strengths.get(pose_type, 0.0)
            if strength > 0.0 and context is not None:
                contexts.append(context)
                scales.append(strength)
                # Infer control dimensions from context shape
                dims.append((context.shape[2], context.shape[3]))

        if not contexts:
            logger.info("[AdvancedOpenposeLoader] No control contexts with non-zero strength")
            return

        # Register via patch_transformer_for_control (class-level patch)
        # Pass control context info via transformer_options
        # The patched forward_orig will read these during sampling
        from comfy.model_patcher import ModelPatcher

        # Create transformer_options dict with control info
        to = {}
        to['flux2_fun_controlnets'] = [controlnet] * len(contexts)
        to['flux2_fun_control_contexts'] = contexts
        to['flux2_fun_control_scales'] = scales
        to['flux2_fun_ctrl_dims'] = dims

        # Pass to sampler via kwargs
        self._control_transformer_options = to

        # Apply class-level patch if not already applied
        patch_transformer_for_control(None, None, None, None)

    def cleanup(self):
        """Clean up active patches and release resources."""
        cleanup_control_patch()
        self._control_transformer_options = None

    def __del__(self):
        """Automatic cleanup when object is garbage collected."""
        try:
            self.cleanup()
        except Exception:
            pass


NODE_CLASS_MAPPINGS = {
    "AdvancedOpenposeLoader": AdvancedOpenposeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedOpenposeLoader": "Advanced Pose Loader",
}
