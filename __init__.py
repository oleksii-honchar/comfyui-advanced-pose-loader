import sys
import os

# Ensure the custom node directory is on sys.path
node_dir = os.path.dirname(os.path.abspath(__file__))
if node_dir not in sys.path:
    sys.path.insert(0, node_dir)

from src.node import AdvancedOpenposeLoader

NODE_CLASS_MAPPINGS = {
    "AdvancedOpenposeLoader": AdvancedOpenposeLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedOpenposeLoader": "Advanced OpenPose Loader"
}