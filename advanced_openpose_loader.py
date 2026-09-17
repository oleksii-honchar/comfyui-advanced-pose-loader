"""
AdvancedOpenposeLoader - FLUX.2 Fun ControlNet Pose Conditioning

Loads pose images, generates ControlNet conditioning for FLUX.2 diffusion models.
Uses the same architecture and control flow as the working Flux2FunControlNetApply
node from comfyui-flux2fun-controlnet.

This is the main entry point for the ComfyUI custom node.
"""
import sys
import os

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.node import AdvancedOpenposeLoader

NODE_CLASS_MAPPINGS = {
    "AdvancedOpenposeLoader": AdvancedOpenposeLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedOpenposeLoader": "Advanced OpenPose Loader"
}