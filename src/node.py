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
        strengths = {
            "openpose": openpose_strength,
            "openpose_hand": openpose_hand_strength,
            "openpose_full": openpose_full_strength,
            "canny": canny_strength,
            "depth": depth_strength,
            "normal": normal_strength,
        }

        # Store pose types and their strengths in wrapper
        wrapper = AdvancedOpenposeWrapper(
            controlnet=controlnet,
            strengths=strengths,
            spatial_fade=spatial_fade,
            spatial_fade_strength=spatial_fade_strength,
        )

        for pose_type, pose_image in processed_images.items():
            if debug:
                logger.info(f"[AdvancedOpenposeLoader] Encoding pose: {pose_type}")
            latent = encode_pose_image(vae_model, pose_image)
            fade_mode = "top" if spatial_fade else "none"
            context = build_control_context(
                latent,
                fade_mode=fade_mode,
                fade_strength=spatial_fade_strength
            )
            wrapper.add_pose_type(pose_type, context)

        # Step 5: Store wrapper in conditioning's control field
        c = [[t[0], t[1].copy()] for t in conditioning]
        for t in c:
            existing_control = t[1].get('control', None)
            if existing_control is not None:
                wrapper.previous_controlnet = existing_control
            t[1]['control'] = wrapper
            t[1]['control_apply_to_uncond'] = True

        if debug:
            logger.info(f"[AdvancedOpenposeLoader] Pipeline complete")

        return (model, c, c)


class AdvancedOpenposeWrapper:
    """Wrapper for advanced pose loader control contexts.
    
    Follows the same pattern as ControlNetWrapper in the deps reference implementation.
    Supports multiple pose types with individual control contexts.
    """
    
    def __init__(self, controlnet, strengths, spatial_fade=False, spatial_fade_strength=1.0):
        self.controlnet = controlnet
        self.strengths = strengths
        self.spatial_fade = spatial_fade
        self.spatial_fade_strength = spatial_fade_strength
        self.poses = []  # List of (pose_type, context) tuples
        self.previous_controlnet = None
    
    def get_extra_hooks(self):
        """Return extra hooks for the controlnet.
        
        Flux2Fun uses forward method patching instead of hooks, so return None.
        This method is required by ComfyUI's get_hooks_from_cond.
        """
        return None
    
    def add_pose_type(self, pose_type, context):
        self.poses.append((pose_type, context))
    
    def pre_run(self, model, percent_to_timestep_function):
        """Apply the Flux2Fun patch at sampling start."""
        from src.flux2fun_integration import patch_transformer_for_control
        patch_transformer_for_control(model, self.controlnet, None, strength=self.strengths.get("openpose", 0.75))
        if self.previous_controlnet:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)
    
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
        """Populate transformer_options with control context data."""
        control_prev = None
        if self.previous_controlnet:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
        
        if transformer_options:
            # Initialize lists if this is the first Flux2Fun controlnet in the chain
            if 'flux2_fun_controlnets' not in transformer_options:
                transformer_options['flux2_fun_controlnets'] = []
                transformer_options['flux2_fun_control_contexts'] = []
                transformer_options['flux2_fun_control_scales'] = []
                transformer_options['flux2_fun_ctrl_dims'] = []
            
            # Add each pose type's context to the lists
            for pose_type, context in self.poses:
                strength = self.strengths.get(pose_type, 0.0)
                if strength == 0.0:
                    continue
                # Estimate dimensions from context shape
                if context is not None:
                    b, c, h, w = context.shape
                    transformer_options['flux2_fun_controlnets'].append(self.controlnet)
                    transformer_options['flux2_fun_control_contexts'].append(context)
                    transformer_options['flux2_fun_control_scales'].append(strength)
                    transformer_options['flux2_fun_ctrl_dims'].append((h // 2, w // 2))
        
        output = {"input": [], "output": []}
        if control_prev:
            output["input"] = control_prev.get("input", [])
            output["output"] = control_prev.get("output", [])
        return output
    
    def cleanup(self):
        """Remove the Flux2Fun patch at sampling end."""
        from src.flux2fun_integration import cleanup_control_patch
        cleanup_control_patch()
        if self.previous_controlnet:
            self.previous_controlnet.cleanup()
    
    def get_models(self):
        return self.previous_controlnet.get_models() if self.previous_controlnet else []
    
    def get_extra_hooks(self):
        return self.previous_controlnet.get_extra_hooks() if self.previous_controlnet else []
    
    def inference_memory_requirements(self, dtype):
        import torch
        mem = sum(p.numel() for p in self.controlnet.parameters()) * 2
        if self.previous_controlnet:
            mem += self.previous_controlnet.inference_memory_requirements(dtype)
        return mem


NODE_CLASS_MAPPINGS = {
    "AdvancedOpenposeLoader": AdvancedOpenposeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedOpenposeLoader": "Advanced Pose Loader",
}
