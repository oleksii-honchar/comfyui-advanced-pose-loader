"""FLUX.2 Fun Control integration via monkey-patching.

Patches the Flux model to accept and use control signals during sampling.
This is the core integration that makes pose conditioning work with FLUX.2.

Approach: Based on comfyui-flux2fun-controlnet's flux_patch.py and ControlNetWrapper.
Patches are scoped to sampling runs (applied at pre_run, removed at cleanup).
"""
from __future__ import annotations

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable

# Lazy imports — these only resolve in actual ComfyUI environment
# (not during import/test time when comfy modules aren't available)
_comfy_model_patcher = None
_comfy_sd = None
_comfy_utils = None
_comfy_model_management = None

def _get_comfy_model_patcher():
    global _comfy_model_patcher
    if _comfy_model_patcher is None:
        import comfy.model_patcher as mp
        _comfy_model_patcher = mp
    return _comfy_model_patcher

def _get_comfy_sd():
    global _comfy_sd
    if _comfy_sd is None:
        import comfy.sd as sd
        _comfy_sd = sd
    return _comfy_sd

def _get_comfy_utils():
    global _comfy_utils
    if _comfy_utils is None:
        import comfy.utils as utils
        _comfy_utils = utils
    return _comfy_utils

def _get_comfy_model_management():
    global _comfy_model_management
    if _comfy_model_management is None:
        import comfy.model_management as mm
        _comfy_model_management = mm
    return _comfy_model_management


class Flux2FunControlNet(nn.Module):
    """FLUX.2 Fun ControlNet — generates control hints for Flux diffusion.

    From Alibaba's VideoX-Fun implementation. Supports pose, canny, depth,
    HED, MLSD, tile control modes.

    Architecture: Projects control context (36 dims for FLUX.2) through a
    series of ControlTransformerBlocks to generate hints injected at specific
    double-block layers.
    """

    # Layers where control hints are injected (same as original implementation)
    CONTROL_LAYERS = [0, 2, 4, 6]

    def __init__(self, hidden_size=6144, num_attention_heads=48, attention_head_dim=128,
                 mlp_ratio=3.0, control_in_dim=36, num_blocks=4, eps=1e-6,
                 dtype=None, device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.control_in_dim = control_in_dim
        self.num_blocks = num_blocks
        self.control_layers_mapping = {layer: idx for idx, layer in enumerate(self.CONTROL_LAYERS[:num_blocks])}

        # Project control context to hidden dimension
        self.control_img_in = nn.Linear(control_in_dim, hidden_size)

        # Control transformer blocks
        self.control_transformer_blocks = nn.ModuleList([
            ControlTransformerBlock(
                dim=hidden_size, num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio,
                eps=eps, bias=False, block_id=i
            ) for i in range(num_blocks)
        ])

        if dtype:
            self.to(dtype)
        if device:
            self.to(device)

    def forward_control(self, x, control_context, encoder_hidden_states,
                        temb_mod_params_img, temb_mod_params_txt,
                        image_rotary_emb=None,
                        ctrl_h=None, ctrl_w=None, txt_seq_len=None,
                        debug=False) -> List[torch.Tensor]:
        """Generate control hints to inject into Flux blocks.

        Args:
            x: Hidden states from main model [B, seq, hidden]
            control_context: Control input [B, seq, 36]
                (16 control + 4 mask + 16 inpaint for FLUX.2)
            encoder_hidden_states: Text embeddings [B, txt_seq, hidden]
            temb_mod_params_img: Modulation params for image stream
            temb_mod_params_txt: Modulation params for text stream
            image_rotary_emb: RoPE embeddings (cos, sin)
            debug: Enable debug output

        Returns:
            List of hint tensors to add to Flux blocks
        """
        if debug:
            print(f"[Flux2 Fun] forward_control:")
            print(f"  x: {x.shape}, abs_mean={x.abs().mean():.4f}")
            print(f"  control_context: {control_context.shape}, abs_mean={control_context.abs().mean():.4f}")

        # Project control context to hidden dimension
        c = self.control_img_in(control_context)

        kwargs = dict(
            x=x,
            encoder_hidden_states=encoder_hidden_states.clone(),
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=image_rotary_emb,
        )

        for i, block in enumerate(self.control_transformer_blocks):
            encoder_hidden_states_out, c = block(c, **kwargs)
            kwargs["encoder_hidden_states"] = encoder_hidden_states_out

        hints = list(torch.unbind(c))[:-1]

        if debug:
            print(f"  Final: {len(hints)} hints generated")

        return hints


class ControlTransformerBlock(nn.Module):
    """Control block with before_proj/after_proj for hint generation."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0,
                 eps=1e-6, bias=False, block_id=0):
        super().__init__()

        self.block_id = block_id
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Simplified attention — uses the control context directly
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.norm_q = nn.RMSNorm(attention_head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=eps)
        self.to_out = nn.Linear(dim, dim, bias=bias)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio), bias=bias),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim, bias=bias)
        )

        # Learnable skip connections for hint generation
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x=None, encoder_hidden_states=None, temb_mod_params_img=None,
                temb_mod_params_txt=None, image_rotary_emb=None, **kwargs):
        """Forward pass through the control transformer block."""
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Attention
        qkv = self.qkv(c)
        q, k, v = qkv.chunk(3, dim=-1)

        batch, seq, dim = q.shape
        head_dim = dim // 8
        q = self.norm_q(q.view(batch, seq, 8, head_dim)).transpose(1, 2)
        k = self.norm_k(k.view(batch, seq, 8, head_dim)).transpose(1, 2)
        v = v.view(batch, seq, 8, head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq, dim)
        attn_out = self.to_out(attn)

        c = c + attn_out

        # FFN
        c = c + self.mlp(self.norm2(c))

        # Generate hint
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)

        return encoder_hidden_states, c


class ControlNetWrapper:
    """Wrapper to integrate with ComfyUI's control system.

    Supports chaining multiple Flux2Fun controlnets by accumulating them
    into lists in transformer_options, which are processed by the patch.
    """

    def __init__(self, controlnet, control_context, strength, ctrl_h, ctrl_w, low_vram=False):
        self.controlnet = controlnet
        self.control_context = control_context
        self.strength = strength
        self.ctrl_h = ctrl_h
        self.ctrl_w = ctrl_w
        self.low_vram = low_vram
        self.previous_controlnet = None
        self.multigpu_clones = {}

        if low_vram and control_context is not None:
            self.control_context = control_context.cpu()

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


# Monkey-patch state
_original_forward_orig = None
_patched = False


def apply_controlnet_model(model, controlnet_path, controlnet_strength=0.75):
    """Load Flux2 Fun ControlNet model from file.

    Args:
        model: ComfyUI model object (for architecture detection)
        controlnet_path: Path to controlnet safetensors file
        controlnet_strength: Default control strength

    Returns:
        Loaded Flux2FunControlNet instance
    """
    state_dict = _get_comfy_utils().load_torch_file(controlnet_path)

    # Detect architecture from weights
    control_in_dim = state_dict["control_img_in.weight"].shape[1]
    hidden_size = state_dict["control_img_in.weight"].shape[0]
    num_blocks = max(
        int(k.split(".")[1]) for k in state_dict
        if k.startswith("control_transformer_blocks.")
    ) + 1

    print(f"[Flux2 Fun] Architecture: hidden={hidden_size}, ctrl_dim={control_in_dim}, blocks={num_blocks}")

    mm = _get_comfy_model_management()
    device = mm.get_torch_device()
    dtype = torch.bfloat16 if mm.should_use_bf16() else torch.float16

    controlnet = Flux2FunControlNet(
        hidden_size=hidden_size,
        num_attention_heads=48,
        attention_head_dim=hidden_size // 48,
        mlp_ratio=3.0,
        control_in_dim=control_in_dim,
        num_blocks=num_blocks,
        dtype=dtype,
        device="cpu"
    )

    missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Flux2 Fun] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Flux2 Fun] Unexpected keys: {len(unexpected)}")

    controlnet.to(device=device, dtype=dtype)
    controlnet.eval()

    print(f"[Flux2 Fun] Loaded successfully")
    return controlnet


def _patched_forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y,
                          guidance=None, control=None, timestep_zero_index=None,
                          transformer_options={}, attn_mask=None, **kwargs):
    """Patched forward_orig with FLUX.2 Fun Control hint injection.

    Delegates to the original when no Flux2Fun controlnets are active.
    """
    global _original_forward_orig

    # Delegate to original when no Flux2Fun controlnets are active
    if not transformer_options.get('flux2_fun_controlnets'):
        return _original_forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y,
                                       guidance=guidance, control=control,
                                       timestep_zero_index=timestep_zero_index,
                                       transformer_options=transformer_options,
                                       attn_mask=attn_mask, **kwargs)

    # --- FLUX.2 Fun Control hint generation and injection ---
    from comfy.ldm.flux.layers import timestep_embedding as _timestep_embedding

    transformer_options = transformer_options.copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})

    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

    if self.params.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.txt_norm is not None:
        txt = self.txt_norm(txt)
    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    vec_orig = vec

    # Generate control hints from each active Flux2Fun controlnet
    all_controlnet_hints = {}
    flux2_fun_controlnets = transformer_options.get('flux2_fun_controlnets', [])
    flux2_fun_control_contexts = transformer_options.get('flux2_fun_control_contexts', [])
    flux2_fun_control_scales = transformer_options.get('flux2_fun_control_scales', [])
    flux2_fun_ctrl_dims = transformer_options.get('flux2_fun_ctrl_dims', [])

    for cn_idx, (controlnet, control_context, control_scale, (ctrl_h, ctrl_w)) in enumerate(
            zip(flux2_fun_controlnets, flux2_fun_control_contexts,
                flux2_fun_control_scales, flux2_fun_ctrl_dims)):
        if controlnet is None or control_context is None:
            continue

        main_img_tokens = ctrl_h * ctrl_w

        try:
            # Use block 0's img_mod/txt_mod for modulation params
            img_mod = self.double_blocks[0].img_mod(vec_orig)
            txt_mod = self.double_blocks[0].txt_mod(vec_orig)

            temb_mod_params_img = (img_mod, img_mod)
            temb_mod_params_txt = (txt_mod, txt_mod)

            img_for_control = img[:, :main_img_tokens].clone()

            controlnet_hints = controlnet.forward_control(
                x=img_for_control,
                control_context=control_context,
                encoder_hidden_states=txt.clone(),
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=None,
                ctrl_h=ctrl_h,
                ctrl_w=ctrl_w,
                txt_seq_len=txt.shape[1],
            )

            del img_for_control

            control_layers_mapping = controlnet.control_layers_mapping
            for layer_idx, hint_idx in control_layers_mapping.items():
                if hint_idx < len(controlnet_hints):
                    if layer_idx not in all_controlnet_hints:
                        all_controlnet_hints[layer_idx] = []
                    all_controlnet_hints[layer_idx].append(
                        (controlnet_hints[hint_idx], control_scale, main_img_tokens)
                    )

        except Exception as e:
            print(f"[Flux2 Fun] Error generating hints for controlnet {cn_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Run main Flux blocks with hint injection
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"

    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        img, txt = block(img=img, txt=txt, vec=vec, pe=pe,
                        attn_mask=attn_mask, transformer_options=transformer_options)

        # Apply control hints at this layer
        if i in all_controlnet_hints:
            for hint, control_scale, main_img_tokens in all_controlnet_hints[i]:
                hint = hint.to(img.device, dtype=img.dtype)
                if hint.shape[1] != main_img_tokens:
                    # Resize hint if needed
                    target_h, target_w = _find_hw(main_img_tokens)
                    hint_2d = hint.permute(0, 2, 1).reshape(hint.shape[0], hint.shape[2], -1, target_w)
                    hint_2d_up = F.interpolate(hint_2d, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    hint = hint_2d_up.reshape(hint.shape[0], hint.shape[2], -1).permute(0, 2, 1)

                # Direct multiplication — user strength = control_scale
                img[:, :main_img_tokens] = img[:, :main_img_tokens] + hint * control_scale

    # Single blocks
    img = torch.cat((txt, img), 1)
    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)

    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]

    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

    img = img[:, txt.shape[1]:, ...]
    img = self.final_layer(img, vec_orig)
    return img


def _find_hw(seq_len):
    """Find h, w from seq_len (assumes square or near-square)."""
    import math
    for h in range(int(math.sqrt(seq_len)), 0, -1):
        if seq_len % h == 0:
            return h, seq_len // h
    return 1, seq_len


def patch_transformer_for_control(model, controlnet, hint, strength=0.75):
    """Apply ControlNet via Flux class-level forward_orig patching.

    This uses the same approach as comfyui-flux2fun-controlnet: patch the
    Flux class's forward_orig method globally.

    Args:
        model: The ComfyUI Flux model object (not used, kept for API compat)
        controlnet: Loaded Flux2FunControlNet instance
        hint: Control hint tensor [B, seq, channels]
        strength: Control strength (0.0-2.0)

    Returns:
        None (patch is applied at class level)
    """
    global _original_forward_orig, _patched

    if not _patched:
        from comfy.model_base import Flux
        _original_forward_orig = Flux.forward_orig
        Flux.forward_orig = _patched_forward_orig
        _patched = True
        print("[Flux2 Fun] ControlNet patch applied (scoped to this sampling run)")
    return None


def unpatch_transformer(model, original_forward):
    """Clean up patches after sampling.

    Args:
        model: The Flux model object
        original_forward: The original forward function to restore (unused)
    """
    global _original_forward_orig, _patched

    if _patched:
        from comfy.model_base import Flux
        Flux.forward_orig = _original_forward_orig
        _original_forward_orig = None
        _patched = False
        print("[Flux2 Fun] ControlNet patch removed")


def build_control_chain(controlnets, hints, strengths):
    """Chain multiple controlnet wrappers via previous_controlnet.

    Each pose type gets its own wrapper. Chain order is maintained from
    the input list. Each wrapper's previous_controlnet points to the
    previous one in the chain.

    Args:
        controlnets: List of Flux2FunControlNet instances
        hints: List of control hint tensors
        strengths: List of control strengths

    Returns:
        List of ControlNetWrapper instances, chained via previous_controlnet
    """
    wrappers = []
    for i, (controlnet, hint, strength) in enumerate(zip(controlnets, hints, strengths)):
        ctrl_h, ctrl_w = hint.shape[2], hint.shape[3]
        wrapper = ControlNetWrapper(controlnet, hint, strength, ctrl_h, ctrl_w)
        if i > 0:
            wrapper.previous_controlnet = wrappers[i - 1]
        wrappers.append(wrapper)

    return wrappers