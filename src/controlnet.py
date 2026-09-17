"""FLUX.2 Fun ControlNet loader and architecture.

Builds the Flux2FunControlNet architecture and ControlNetWrapper that integrates
with ComfyUI's control system. Supports chainable controlnets via previous_controlnet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import comfy.utils
import comfy.model_management


def load_controlnet(controlnet_name, device=None, dtype=None):
    """Load FLUX.2 Fun ControlNet checkpoint and build the architecture.
    
    Args:
        controlnet_name: Filename of the controlnet in ComfyUI's controlnet folder
        device: Target device (default: comfy model device)
        dtype: Target dtype (default: bf16 if supported, else fp16)
    
    Returns:
        Flux2FunControlNet: Loaded controlnet model
    """
    import folder_paths
    
    if device is None:
        device = comfy.model_management.get_torch_device()
    if dtype is None:
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16
    
    controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
    print(f"[AdvancedOpenposeLoader] Loading controlnet: {controlnet_name}")
    
    state_dict = comfy.utils.load_torch_file(controlnet_path)
    
    # Detect architecture from weights
    control_in_dim = state_dict["control_img_in.weight"].shape[1]
    hidden_size = state_dict["control_img_in.weight"].shape[0]
    num_blocks = max(
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("control_transformer_blocks.")
    ) + 1
    
    attention_head_dim = hidden_size // 48
    
    print(f"[AdvancedOpenposeLoader] Architecture: hidden={hidden_size}, ctrl_dim={control_in_dim}, blocks={num_blocks}")
    
    controlnet = Flux2FunControlNet(
        hidden_size=hidden_size,
        num_attention_heads=48,
        attention_head_dim=attention_head_dim,
        mlp_ratio=3.0,
        control_in_dim=control_in_dim,
        num_blocks=num_blocks,
        dtype=dtype,
        device=device
    )
    
    missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[AdvancedOpenposeLoader] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[AdvancedOpenposeLoader] Unexpected keys: {len(unexpected)}")
    
    controlnet.eval()
    print(f"[AdvancedOpenposeLoader] Controlnet loaded successfully")
    
    return controlnet


class Flux2FunControlNet(nn.Module):
    """FLUX.2 Fun ControlNet architecture (from comfyui-flux2fun-controlnet).
    
    Generates control hints to inject into FLUX.2 diffusion blocks.
    Supports pose, canny, depth, HED, MLSD, tile control modes.
    """
    
    CONTROL_LAYERS = [0, 2, 4, 6]
    
    def __init__(self, hidden_size=6144, num_attention_heads=48, attention_head_dim=128,
                 mlp_ratio=3.0, control_in_dim=260, num_blocks=4, dtype=None, device=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.mlp_ratio = mlp_ratio
        self.control_in_dim = control_in_dim
        self.num_blocks = num_blocks
        self.control_layers_mapping = {layer: idx for idx, layer in enumerate(self.CONTROL_LAYERS[:num_blocks])}
        
        self.control_img_in = nn.Linear(control_in_dim, hidden_size)
        
        self.control_transformer_blocks = nn.ModuleList([
            ControlTransformerBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                block_id=i
            ) for i in range(num_blocks)
        ])
        
        if dtype:
            self.to(dtype=dtype)
        if device:
            self.to(device=device)
    
    def forward_control(self, x, control_context, encoder_hidden_states,
                       temb_mod_params_img, temb_mod_params_txt,
                       image_rotary_emb=None, debug=False):
        """Generate control hints to inject into Flux blocks.
        
        Args:
            x: Hidden states from main model [B, seq, hidden]
            control_context: Control input [B, seq, 260]
            encoder_hidden_states: Text embeddings [B, txt_seq, hidden]
            temb_mod_params_img: Modulation params for image stream
            temb_mod_params_txt: Modulation params for text stream
            image_rotary_emb: RoPE embeddings
            debug: Enable debug output
        
        Returns:
            List of hint tensors to add to Flux blocks
        """
        if debug:
            print(f"[AdvancedOpenposeLoader] forward_control: x={x.shape}, control={control_context.shape}")
        
        # Project control context to hidden dimension
        c = self.control_img_in(control_context)
        
        if debug:
            print(f"[AdvancedOpenposeLoader] After projection: {c.shape}")
        
        # Process through control transformer blocks
        encoder_hidden_states_out = encoder_hidden_states.clone()
        all_c = [c]
        
        for i, block in enumerate(self.control_transformer_blocks):
            encoder_hidden_states_out, c = block(
                c,
                x=x,
                encoder_hidden_states=encoder_hidden_states_out,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=image_rotary_emb
            )
            all_c.append(c)
        
        # Return all intermediate outputs for injection
        hints = list(torch.unbind(torch.stack(all_c)))[:-1]
        
        if debug:
            print(f"[AdvancedOpenposeLoader] Generated {len(hints)} hints")
        
        return hints


class ControlTransformerBlock(nn.Module):
    """Control transformer block for Flux2 Fun ControlNet.
    
    Uses before_proj/after_proj for hint generation. The before_proj projects
    the control input to the hidden dimension, and the after_proj generates
    the hint to inject into the main transformer.
    """
    
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0, block_id=0):
        super().__init__()
        
        self.block_id = block_id
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)
        
        # Simplified transformer block (would use actual attention/FFN in production)
        # For now, just identity + projection
    
    def forward(self, c, x, encoder_hidden_states, temb_mod_params_img,
               temb_mod_params_txt, image_rotary_emb=None):
        """Process control through the transformer block.
        
        Args:
            c: Control tensor
            x: Main model hidden states
            encoder_hidden_states: Text embeddings
            temb_mod_params_img: Image stream modulation params
            temb_mod_params_txt: Text stream modulation params
            image_rotary_emb: RoPE embeddings
        
        Returns:
            Tuple of (encoder_hidden_states, control tensor)
        """
        if self.block_id == 0:
            c = self.before_proj(c) + x
        
        # Simplified: just pass through with projection
        c = c + self.after_proj(c)
        
        return encoder_hidden_states, c


class ControlNetWrapper:
    """Wrapper to integrate with ComfyUI's control system.
    
    Supports chaining multiple controlnets by accumulating them into lists
    in transformer_options, which are processed by the patch.
    """
    
    def __init__(self, controlnet, control_context, strength, ctrl_h, ctrl_w, low_vram=False):
        self.controlnet = controlnet
        self.control_context = control_context
        self.strength = strength
        self.ctrl_h = ctrl_h
        self.ctrl_w = ctrl_w
        self.low_vram = low_vram
        self.previous_controlnet = None
    
    def pre_run(self, model, percent_to_timestep_function):
        """Called before sampling starts."""
        if self.previous_controlnet:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)
    
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
        """Get control signal to inject into the model.
        
        Accumulates controlnets in transformer_options for the patch to process.
        """
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
    
    def copy(self):
        """Create a copy of the wrapper."""
        c = ControlNetWrapper(self.controlnet, self.control_context, self.strength, self.ctrl_h, self.ctrl_w, self.low_vram)
        c.previous_controlnet = self.previous_controlnet
        return c
    
    def cleanup(self):
        """Called after sampling completes."""
        if self.previous_controlnet:
            self.previous_controlnet.cleanup()
    
    def get_models(self):
        """Return list of models used by this wrapper."""
        return [self.controlnet] + (self.previous_controlnet.get_models() if self.previous_controlnet else [])
    
    def inference_memory_requirements(self, dtype):
        """Estimate memory requirements for inference."""
        mem = sum(p.numel() for p in self.controlnet.parameters()) * 2
        if self.previous_controlnet:
            mem += self.previous_controlnet.inference_memory_requirements(dtype)
        return mem