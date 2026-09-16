"""
Advanced OpenPose Loader — A custom ComfyUI node that consolidates the pose-based
image generation pipeline: pose loading, VAE encoding, reference latent creation,
and ControlNet application.

Node inputs:
- folder_name: Name of pose folder containing all pose assets
- conditioning: Existing conditioning to modify
- model: Existing model to modify
- strength_openpose: ControlNet strength for OpenPose (0.0-2.0, default 0.75)
- strength_openpose_hand: ControlNet strength for OpenPose hand (0.0-2.0, default 0.80)
- strength_openpose_full: ControlNet strength for OpenPose full (0.0-2.0, default 0.85)
- strength_canny: ControlNet strength for Canny edges (0.0-2.0, default 0.0)
- strength_depth: ControlNet strength for Depth (0.0-2.0, default 0.0)
- strength_normal: ControlNet strength for Normal (0.0-2.0, default 0.0)
- vae: VAE model for encoding pose images
- controlnet: Flux2 Fun ControlNet model for pose guidance
- spatial_fade: Spatial fade mode (none, top, bottom, left, right, default none)
- spatial_fade_strength: Spatial fade strength (0.0-1.0, default 0.5)
- debug: Enable debug output (default false)

Node outputs:
- model: Modified model with pose reference latents
- positive: Modified positive conditioning with ControlNet wrappers
- negative: Modified negative conditioning with ControlNet wrappers
"""

import os
import numpy as np
import torch
from PIL import Image

# ComfyUI modules — available on the remote ComfyUI instance
import comfy.utils
import folder_paths

# Pose-type-specific default ControlNet strengths
# Different pose sources (openposes.com vs posedepot) require different
# ControlNet strengths for optimal results.
STRENGTH_DEFAULTS = {
    'openpose': 0.75,
    'openpose_hand': 0.80,
    'openpose_full': 0.85,
    'canny': 0.85,
    'depth': 0.60,
    'normal': 0.40,
}


class AdvancedOpenposeLoader:
    """
    Advanced Pose Loader node for ComfyUI.
    
    Consolidates the pose-based image generation pipeline into a single node:
    - Loads pose images from a specified folder
    - Internally loads and uses VAE for encoding
    - Creates reference latents (Flux2Klein mechanism)
    - Loads and applies ControlNet (Flux2 Fun mechanism)
    """

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "advanced_openpose_load"
    CATEGORY = "Custom/Pose"

    # This node specializes in OpenPose format only (from openposes.com keypoint skeletons)

    # Base path for pose assets
    # Uses ComfyUI's folder_paths system, falls back to /opt/comfyui/poses (Docker)
    def _get_poses_base_path(self):
        try:
            path = folder_paths.get_full_path("poses", "")
            if path:
                return path
        except Exception:
            pass
        # Fallback for Docker container
        return "/opt/comfyui/poses"

    def __init__(self):
        self.loaded_vae = None

    def _load_vae(self, vae_name="flux2-vae.safetensors"):
        """
        Load the specified VAE model internally.
        
        Args:
            vae_name: Name of the VAE model file to load
            
        Returns:
            The loaded VAE model with encode/decode methods
            
        Raises:
            FileNotFoundError: If the VAE model file is not found
        """
        # Check if already loaded (cache) - use same VAE for all calls
        if self.loaded_vae is not None:
            return self.loaded_vae

        vae_path = folder_paths.get_full_path("vae", vae_name)
        
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found: {vae_path}")
        
        # Load VAE state dict and wrap in VAE object
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        self.loaded_vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        return self.loaded_vae

    def _resize_pose_image(self, image):
        """
        Resize pose image to 1024x1024 (Flux2 Fun ControlNet recommended).
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image object
        """
        return image.resize((1024, 1024), Image.LANCZOS)

    def _encode_pose_image(self, pose_image):
        """
        Resize and encode pose image to latent.
        
        Args:
            pose_image: PIL Image object (the loaded pose image)
            
        Returns:
            Encoded pose latent (tensor)
        """
        # Resize to 512x512
        resized_image = self._resize_pose_image(pose_image)
        
        # Convert PIL Image to tensor [1, H, W, C] in [0, 1] range
        import numpy as np
        img_array = np.array(resized_image.convert("RGB")) / 255.0
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)  # [1, H, W, C]
        
        # Load VAE if not already loaded
        vae = self._load_vae()
        
        # Encode the pose image to latent
        pose_latent = vae.encode(img_tensor)
        return pose_latent

    def _load_pose_image(self, folder_name, pose_type="openpose"):
        """
        Load a specific pose image from poses/{folder_name}/{pose_type}.png
        
        Args:
            folder_name: Pose folder name (e.g. 'lora-FB-L1')
            pose_type: Pose type (openpose, openpose_hand, openpose_full, canny, depth, normal)
        
        Returns:
            PIL Image object of the pose image, or None if not found
        """
        base = self._get_poses_base_path()
        # Folder-per-pose structure: poses/{folder_name}/{pose_type}.png
        image_path = os.path.join(base, folder_name, f"{pose_type}.png")

        if not os.path.exists(image_path):
            return None

        return Image.open(image_path)

    def _list_available_poses(self):
        """List available pose folders for error messages."""
        base = self._get_poses_base_path()
        try:
            return [f for f in os.listdir(base)
                    if os.path.isdir(os.path.join(base, f))]
        except Exception:
            return []


    def _store_reference_latents(self, conditioning, pose_latent):
        """
        Store pose latent in conditioning metadata for Flux2Klein reference latent mechanism.
        
        This stores the pose latent in the conditioning metadata so that during sampling,
        the Flux2Klein model can extract it and use it for pose guidance via cross-attention.
        
        Args:
            conditioning: List of tuples (cond_tensor, cond_metadata)
            pose_latent: The pose latent tensor to store as a reference latent
        
        Returns:
            Modified conditioning with reference_latents and reference_latents_method
            set in each entry's metadata
        """
        modified_conditioning = []
        for cond in conditioning:
            cond_tensor, cond_metadata = cond
            # Copy metadata to avoid mutating the original
            new_metadata = cond_metadata.copy()
            # Add reference latents to metadata (Flux2Klein mechanism)
            new_metadata["reference_latents"] = [pose_latent]
            new_metadata["reference_latents_method"] = "index"
            modified_conditioning.append((cond_tensor, new_metadata))
        return modified_conditioning

    def _get_controlnet_patch_functions(self, model_version):
        """
        Get ControlNet patch functions for the specified model version.
        
        Different model versions use different ControlNet patch functions due to
        differences in layer configurations.
        
        Args:
            model_version: Model version string (flux2, flux, fluxdev, flux_kontext)
            
        Returns:
            Tuple of (patch_forward_fn, patch_backward_fn) to use for ControlNet patches
            
        Raises:
            ValueError: If model_version is not supported
        """
        if model_version == "flux2":
            # Flux2 uses controlnet_config from comfy.controlnet
            from comfy.controlnet import controlnet_config
            return controlnet_config, controlnet_config
        elif model_version == "flux":
            from comfy.controlnet import controlnet_config
            return controlnet_config, controlnet_config
        elif model_version == "fluxdev":
            from comfy.controlnet import controlnet_config
            return controlnet_config, controlnet_config
        elif model_version == "flux_kontext":
            from comfy.controlnet import controlnet_config
            return controlnet_config, controlnet_config
        else:
            raise ValueError(f"Unsupported model version: {model_version}")

    def _load_controlnet(self, controlnet_name="FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors"):
        """
        Load the Flux2 Fun ControlNet model internally.
        
        Loads the model on CPU to avoid VRAM conflicts with other models.
        The model will be moved to GPU during ControlNet application.
        
        Args:
            controlnet_name: Name of the ControlNet model file to load
            
        Returns:
            The loaded Flux2FunControlNet model instance
            
        Raises:
            FileNotFoundError: If the ControlNet model file is not found
        """
        import importlib.util
        import sys
        
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        
        if not os.path.exists(controlnet_path):
            raise FileNotFoundError(f"ControlNet model not found: {controlnet_path}")
        
        # Load state dict
        state_dict = comfy.utils.load_torch_file(controlnet_path)
        
        # Detect architecture from weights
        control_in_dim = state_dict["control_img_in.weight"].shape[1]
        hidden_size = state_dict["control_img_in.weight"].shape[0]
        num_blocks = max(int(k.split(".")[1]) for k in state_dict if k.startswith("control_transformer_blocks.")) + 1
        
        # Load Flux2FunControlNet class from the custom node
        module_path = "/opt/comfyui/custom_nodes/comfyui-flux2fun-controlnet"
        sys_module_name = "comfyui_x_flux2fun_x_controlnet"
        
        if sys_module_name not in sys.modules:
            module_spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[sys_module_name] = module
            module_spec.loader.exec_module(module)
        
        # Get Flux2FunControlNet class from the module's nodes submodule
        nodes_module = importlib.import_module(f"{sys_module_name}.nodes")
        Flux2FunControlNet = nodes_module.Flux2FunControlNet
        
        # Create model on CPU (avoid VRAM conflicts)
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16
        
        controlnet = Flux2FunControlNet(
            hidden_size=hidden_size, num_attention_heads=48,
            attention_head_dim=hidden_size // 48, mlp_ratio=3.0,
            control_in_dim=control_in_dim, num_blocks=num_blocks,
            dtype=dtype, device="cpu"
        )
        
        # Load weights (keep on CPU to save VRAM)
        controlnet.load_state_dict(state_dict, strict=False)
        controlnet.eval()
        
        # The controlnet is stored in conditioning metadata, not returned as model.
        # It will be picked up by the flux_patch module during sampling.
        # flux_patch handles VRAM management (CPU/GPU offloading) automatically.
        
        return controlnet

    def _build_control_context(self, control_latent, mask_latent=None, inpaint_latent=None):
        """
        Build control context from pose latents.
        
        Control context structure: [control(128), mask(4), inpaint(128)] = 260 dimensions
        
        Args:
            control_latent: Pose latent tensor (control signal) - shape [B, C, H, W]
            mask_latent: Optional mask latent tensor (default: zeros)
            inpaint_latent: Optional inpaint latent tensor (default: zeros)
            
        Returns:
            Control context tensor with shape [batch, sequence_length, 260]
        """
        # control_latent shape: [B, C, H, W] where C=16 for Flux2 VAE
        # After flatten and permute: [B, H*W, C]
        print(f"[DEBUG] control_latent.shape before squeeze: {control_latent.shape}")
        print(f"[DEBUG] control_latent.shape: {control_latent.shape}")
        print(f"[DEBUG] control_latent.shape: {control_latent.shape}")
        shape = control_latent.shape
        if len(shape) == 5:
            # WanVAE produces 5D tensor (B, T, C, H, W) - squeeze time dim for ControlNet
            control_latent = control_latent[:, 0, :, :, :]  # Take first time step
        batch_size, channels, h, w = control_latent.shape
        
        # Flatten spatial dimensions and permute to [batch, sequence_length, channels]
        control_flat = control_latent.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Mask: create zeros with 4 channels at the latent spatial resolution
        mask_flat = torch.zeros(batch_size, h * w, 4, dtype=control_latent.dtype, device=control_latent.device)
        
        # Inpaint: create zeros with same channel count as control
        inpaint_flat = torch.zeros(batch_size, h * w, channels, dtype=control_latent.dtype, device=control_latent.device)
        
        # Concatenate along channel dimension: [batch, sequence_length, 260]
        control_context = torch.cat([control_flat, mask_flat, inpaint_flat], dim=2)
        return control_context

    def _create_spatial_fade_mask(self, mode, fade_strength, h=32, w=32):
        """
        Create a spatial fade mask based on mode and strength.
        
        Args:
            mode: Fade mode (none, top, bottom, left, right)
            fade_strength: Fade strength (0.0-1.0)
            h: Height of the mask
            w: Width of the mask
            
        Returns:
            Float mask tensor of shape (h, w) with values between 0 and 1
        """
        mask = torch.ones(h, w)
        
        if mode == "none":
            return mask
        
        if mode == "top":
            # Fade from top to bottom (strong at top, weak at bottom)
            for i in range(h):
                factor = 1.0 - (fade_strength * (i / (h - 1)))
                mask[i, :] = factor
        elif mode == "bottom":
            # Fade from bottom to top (strong at bottom, weak at top)
            for i in range(h):
                factor = 1.0 - (fade_strength * (1 - i / (h - 1)))
                mask[i, :] = factor
        elif mode == "left":
            # Fade from left to right (strong at left, weak at right)
            for j in range(w):
                factor = 1.0 - (fade_strength * (j / (w - 1)))
                mask[:, j] = factor
        elif mode == "right":
            # Fade from right to left (strong at right, weak at left)
            for j in range(w):
                factor = 1.0 - (fade_strength * (1 - j / (w - 1)))
                mask[:, j] = factor
        
        return mask

    def _apply_spatial_fade(self, control_context, mask):
        """
        Apply spatial fade mask to control context.
        
        Args:
            control_context: Control context tensor [batch, sequence_length, channels]
            mask: Spatial fade mask [h, w]
            
        Returns:
            Faded control context tensor
        """
        # mask shape: (h, w)
        # control_context shape: (1, seq_len, 260)
        # Need to reshape mask to match seq_len (h*w)
        mask_flat = mask.flatten().unsqueeze(0).unsqueeze(2)  # (1, h*w, 1)
        control_context = control_context * mask_flat
        return control_context

    def _apply_controlnet(self, conditioning, controlnet, control_context, strength, spatial_fade="none", spatial_fade_strength=0.5, model_version="flux2"):
        """
        Create ControlNet wrapper and apply to conditioning.
        
        Uses the existing Flux2 Fun ControlNet ControlNetWrapper pattern:
        wraps the controlnet and control context, then stores it in conditioning
        metadata so it gets picked up during sampling.
        
        Args:
            conditioning: List of tuples (cond_tensor, cond_metadata)
            controlnet: Loaded Flux2FunControlNet model
            control_context: Control context tensor [batch, sequence_length, 260]
            strength: ControlNet strength (0.0-2.0, user-facing)
            spatial_fade: Spatial fade mode (none, top, bottom, left, right)
            spatial_fade_strength: Spatial fade strength (0.0-1.0)
            model_version: Model version for ControlNet patches (flux2, flux, fluxdev, flux_kontext)
            
        Returns:
            Modified conditioning with ControlNet wrapper applied
        """
        import sys
        import os
        
        # Find the flux2fun custom node directory
        comfyui_path = "/opt/comfyui"
        flux2fun_path = os.path.join(comfyui_path, "custom_nodes", "comfyui-flux2fun-controlnet")
        
        if not os.path.exists(flux2fun_path):
            raise FileNotFoundError(f"Flux2 Fun ControlNet node not found at {flux2fun_path}")
        
        # Add the flux2fun directory to sys.path for flux_patch import
        if flux2fun_path not in sys.path:
            sys.path.insert(0, flux2fun_path)
        
        # Import flux_patch for patching
        from flux_patch import apply_patch, remove_patch
        
        # Define ControlNetWrapper inline (copied from comfyui-flux2fun-controlnet)
        class ControlNetWrapper:
            def __init__(self, controlnet, control_context, strength, ctrl_h, ctrl_w, low_vram=False):
                self.controlnet = controlnet
                self.control_context = control_context
                self.strength = strength
                self.ctrl_h = ctrl_h
                self.ctrl_w = ctrl_w
                self.low_vram = low_vram
                self.previous_controlnet = None
                self.multigpu_clones = {}
                self.extra_hooks = None
                
                if low_vram and control_context is not None:
                    self.control_context = control_context.cpu()
            
            def pre_run(self, model, percent_to_timestep_function):
                apply_patch()
                if self.previous_controlnet:
                    self.previous_controlnet.pre_run(model, percent_to_timestep_function)
            
            def get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
                control_prev = None
                if self.previous_controlnet:
                    control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
                
                if transformer_options:
                    if 'flux2_fun_controlnets' not in transformer_options:
                        transformer_options['flux2_fun_controlnets'] = []
                        transformer_options['flux2_fun_control_contexts'] = []
                        transformer_options['flux2_fun_control_scales'] = []
                        transformer_options['flux2_fun_ctrl_dims'] = []
                        transformer_options['flux2_fun_low_vram'] = self.low_vram
                    
                    transformer_options['flux2_fun_controlnets'].append(self.controlnet)
                    transformer_options['flux2_fun_control_contexts'].append(self.control_context)
                    transformer_options['flux2_fun_control_scales'].append(self.strength)
                    transformer_options['flux2_fun_ctrl_dims'].append((self.ctrl_h, self.ctrl_w))
                
                output = {"input": [], "output": []}
                if control_prev:
                    output["input"] = control_prev.get("input", [])
                    output["output"] = control_prev.get("output", [])
                return output
            
            def get_extra_hooks(self):
                out = []
                if self.extra_hooks is not None:
                    out.append(self.extra_hooks)
                if self.previous_controlnet is not None:
                    out += self.previous_controlnet.get_extra_hooks()
                return out
            
            def get_models(self):
                return self.previous_controlnet.get_models() if self.previous_controlnet else []
            
            def inference_memory_requirements(self, dtype):
                mem = sum(p.numel() for p in self.controlnet.parameters()) * 2
                if self.previous_controlnet:
                    mem += self.previous_controlnet.inference_memory_requirements(dtype)
                return mem
            
            def copy(self):
                c = ControlNetWrapper(self.controlnet, self.control_context, self.strength, self.ctrl_h, self.ctrl_w, self.low_vram)
                c.previous_controlnet = self.previous_controlnet
                return c
            
            def cleanup(self):
                remove_patch()
                if self.previous_controlnet:
                    self.previous_controlnet.cleanup()
        
        # Apply spatial fade if requested
        if spatial_fade != "none":
            fade_mask = self._create_spatial_fade_mask(spatial_fade, spatial_fade_strength)
            control_context = self._apply_spatial_fade(control_context, fade_mask)
        
        # Calculate dimensions for wrapper
        # 1024x1024 output = 64x64 latent (128 downscale)
        # Was hardcoded 32x32 (512x512), causing pose to only affect 25% of image
        lat_h = 64
        lat_w = 64
        
        # Create ControlNetWrapper
        # User specifies strength, wrapper uses 1-strength (inverted)
        wrapper = ControlNetWrapper(
            controlnet,
            control_context,
            strength=1.0 - strength,
            ctrl_h=lat_h,
            ctrl_w=lat_w,
            low_vram=True
        )
        
        # Apply wrapper to each conditioning entry
        modified_conditioning = []
        for cond in conditioning:
            cond_tensor, cond_metadata = cond
            new_metadata = cond_metadata.copy()
            
            # Chain with existing control if present
            existing_control = cond_metadata.get('control', None)
            if existing_control is not None:
                wrapper.previous_controlnet = existing_control
            
            new_metadata['control'] = wrapper
            modified_conditioning.append((cond_tensor, new_metadata))
        
        # Free temporary tensors before returning
        # control_context is stored in wrapper, safe to delete local ref
        del control_context
        del control_model
        
        # Force garbage collection to release VRAM
        import gc
        gc.collect()
        
        # Clear PyTorch CUDA cache
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        
        return modified_conditioning

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Existing model to modify"
                }),
                "vae": (folder_paths.get_filename_list("vae"), {
                    "default": "flux2-vae.safetensors",
                    "tooltip": "VAE model to use for encoding pose images"
                }),
                "controlnet": (folder_paths.get_filename_list("controlnet"), {
                    "default": "FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors",
                    "tooltip": "Flux2 Fun ControlNet model for pose guidance"
                }),
                "folder_name": ("STRING", {
                    "default": "",
                    "placeholder": "Pose folder name",
                    "tooltip": "Name of the pose folder containing all pose assets"
                }),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Existing conditioning to modify"
                }),
                "strength_openpose": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "ControlNet strength for OpenPose (body pose, reference_index=0)"
                }),
                "strength_openpose_hand": ("FLOAT", {
                    "default": 0.80,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "ControlNet strength for OpenPose hand (reference_index=1). 0.0 = disabled"
                }),
                "strength_openpose_full": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "ControlNet strength for OpenPose full (reference_index=2). 0.0 = disabled"
                }),
                "strength_canny": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "ControlNet strength for Canny edges (reference_index=3). 0.0 = disabled"
                }),
                "strength_depth": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "ControlNet strength for Depth (reference_index=4). 0.0 = disabled"
                }),
                "strength_normal": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "ControlNet strength for Normal (reference_index=5). 0.0 = disabled"
                }),
            },
            "optional": {
                "spatial_fade": (["none", "top", "bottom", "left", "right"], {
                    "default": "none",
                    "tooltip": "Spatial fade mode"
                }),
                "spatial_fade_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Spatial fade strength (0.0-1.0)"
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug output"
                }),
            }
        }

    def advanced_openpose_load(self, model, vae, controlnet, folder_name, conditioning,
                              strength_openpose,
                              strength_openpose_hand=0.80,
                              strength_openpose_full=0.85,
                              strength_canny=0.0,
                              strength_depth=0.0,
                              strength_normal=0.0,
                              spatial_fade="none", spatial_fade_strength=0.5, debug=False, use_reference_latents=False):
        """
        Load and apply pose guidance using the consolidated pipeline.

        Internal steps:
        1. Load pose image from specified folder/type
        2. Internally load and use VAE for encoding
        3. Create reference latents (Flux2Klein mechanism)
        4. Load and apply ControlNet (Flux2 Fun mechanism)

        Args:
            model: Existing model to modify
            folder_name: Name of the pose folder containing all pose assets
            conditioning: Existing conditioning to modify
            vae_name: VAE model file name for encoding pose images
            controlnet_name: Flux2 Fun ControlNet model file name
            strength: ControlNet strength (-1.0 = auto 0.75 for openpose, 0.0-2.0 manual)
            reference_index: Reference latent index to control
            spatial_fade: Spatial fade mode
            spatial_fade_strength: Spatial fade strength
            debug: Enable debug output

        Returns:
            (model, positive_conditioning, negative_conditioning)
        """
        if debug:
            print(f"[AdvancedOpenposeLoader] === Starting pose loading ===")
            print(f"[AdvancedOpenposeLoader] vae={vae}")
            print(f"[AdvancedOpenposeLoader] controlnet={controlnet}")
            print(f"[AdvancedOpenposeLoader] folder_name={folder_name}")
            print(f"[AdvancedOpenposeLoader] strength_openpose={strength_openpose}")
            print(f"[AdvancedOpenposeLoader] strength_openpose_hand={strength_openpose_hand}")
            print(f"[AdvancedOpenposeLoader] strength_openpose_full={strength_openpose_full}")
            print(f"[AdvancedOpenposeLoader] strength_canny={strength_canny}")
            print(f"[AdvancedOpenposeLoader] strength_depth={strength_depth}")
            print(f"[AdvancedOpenposeLoader] strength_normal={strength_normal}")
            print(f"[AdvancedOpenposeLoader] spatial_fade={spatial_fade}")
        
        if debug:
            print(f"[AdvancedOpenposeLoader] Step 1/5: Loading VAE...")
        
        # Load VAE with selected vae_name (cached for subsequent calls)
        self._load_vae(vae)
        
        # Pose types with their filenames and default strengths
        pose_types = [
            ("openpose", "openpose.png", strength_openpose),
            ("openpose_hand", "openpose_hand.png", strength_openpose_hand),
            ("openpose_full", "openpose_full.png", strength_openpose_full),
            ("canny", "canny.png", strength_canny),
            ("depth", "depth.png", strength_depth),
            ("normal", "normal.png", strength_normal),
        ]
        
        # Load and apply ControlNet for each non-zero strength pose type
        controlnet_model = self._load_controlnet(controlnet)
        
        for pose_type, filename, strength in pose_types:
            if strength > 0.0:
                if debug:
                    print(f"[AdvancedOpenposeLoader] Loading {pose_type} image...")
                
                # Load the specific pose image
                pose_image = self._load_pose_image(folder_name, pose_type)
                
                if pose_image is None:
                    if debug:
                        print(f"[AdvancedOpenposeLoader] WARNING: {pose_type} image not found, skipping (strength={strength:.2f})")
                    continue
                
                if debug:
                    print(f"[AdvancedOpenposeLoader] Encoding {pose_type} image...")
                
                # Encode to latent
                pose_latent = self._encode_pose_image(pose_image)
                
                # Store reference latents
                if use_reference_latents:
                    conditioning = self._store_reference_latents(conditioning, pose_latent)
                
                if debug:
                    print(f"[AdvancedOpenposeLoader] Applying {pose_type} control (strength={strength:.2f})")
                
                # Build control context and apply ControlNet
                control_context = self._build_control_context(pose_latent)
                conditioning = self._apply_controlnet(
                    conditioning,
                    controlnet_model,
                    control_context,
                    strength,
                    spatial_fade=spatial_fade,
                    spatial_fade_strength=spatial_fade_strength
                )
        
        if debug:
            print(f"[AdvancedOpenposeLoader] === Pose loading complete ===")
        
        # Option 2: Free all auxiliary models and tensors after conditioning is built
        # Only main model should remain in VRAM for sampling
        self._cleanup_auxiliary_resources(debug)
        
        if debug:
            print(f"[AdvancedOpenposeLoader] Auxiliary resources freed")
        
        # Return model, positive conditioning, negative conditioning
        return (model, conditioning, conditioning)


# ComfyUI node registration
    def _cleanup_auxiliary_resources(self, debug=False):
        """
        Free all auxiliary models and tensors after conditioning is built.
        Keeps only the main model for the sampling phase.
        """
        # Unload ControlNet model
        if hasattr(self, '_controlnet_model') and self._controlnet_model is not None:
            if debug:
                print(f"[AdvancedOpenposeLoader] Unloading ControlNet model")
            del self._controlnet_model
            self._controlnet_model = None
        
        # Unload VAE
        if hasattr(self, '_vae') and self._vae is not None:
            if debug:
                print(f"[AdvancedOpenposeLoader] Unloading VAE")
            del self._vae
            self._vae = None
        
        # Clear cached pose images
        if hasattr(self, '_pose_images') and self._pose_images is not None:
            if debug:
                print(f"[AdvancedOpenposeLoader] Clearing pose images")
            self._pose_images.clear()
        
        # Clear cached pose latents
        if hasattr(self, '_pose_latents') and self._pose_latents is not None:
            if debug:
                print(f"[AdvancedOpenposeLoader] Clearing pose latents")
            self._pose_latents.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Option 3: Clear PyTorch CUDA cache
        try:
            import torch
            torch.cuda.empty_cache()
            if debug:
                print(f"[AdvancedOpenposeLoader] CUDA cache cleared")
        except ImportError:
            pass

NODE_CLASS_MAPPINGS = {
    "AdvancedOpenposeLoader": AdvancedOpenposeLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedOpenposeLoader": "Advanced OpenPose Loader"
}
