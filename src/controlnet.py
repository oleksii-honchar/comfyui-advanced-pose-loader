"""ControlNet model loading."""
from __future__ import annotations
import torch
import comfy.utils


def load_controlnet(controlnet_path):
    """Load a ControlNet model from the given path."""
    if not controlnet_path or not controlnet_path.strip():
        return None
    controlnet_path = controlnet_path.strip()
    if controlnet_path.endswith(".safetensors"):
        return comfy.utils.load_torch_file(controlnet_path, safe=True)
    else:
        raise ValueError(f"Unsupported ControlNet file type: {controlnet_path}. Use .safetensors.")