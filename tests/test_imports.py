"""Import test - verifies all modules load and the node is registered."""
import sys
import os
from unittest.mock import MagicMock

# Add the custom node directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")
print(f"sys.path[0]: {sys.path[0]}")

# Mock comfy modules (no ComfyUI environment needed)
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.model_patcher'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()

print("=" * 60)
print("AdvancedOpenposeLoader Import Test")
print("=" * 60)

# Test 1: Import the main node module directly
print("\n[1/6] Testing advanced_openpose_loader import...")
try:
    import advanced_openpose_loader
    print("  ✓ advanced_openpose_loader module imported")
except ImportError as e:
    print(f"  ✗ Failed to import advanced_openpose_loader: {e}")
    sys.exit(1)

# Test 2: Import the node class
print("\n[2/6] Testing AdvancedOpenposeLoader class import...")
try:
    from advanced_openpose_loader import AdvancedOpenposeLoader
    print("  ✓ AdvancedOpenposeLoader class imported")
except ImportError as e:
    print(f"  ✗ Failed to import AdvancedOpenposeLoader: {e}")
    sys.exit(1)

# Test 3: Verify node registration
print("\n[3/6] Testing NODE_CLASS_MAPPINGS registration...")
try:
    from advanced_openpose_loader import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    assert "AdvancedOpenposeLoader" in NODE_CLASS_MAPPINGS, "Node not in NODE_CLASS_MAPPINGS"
    assert "AdvancedOpenposeLoader" in NODE_DISPLAY_NAME_MAPPINGS, "Node not in NODE_DISPLAY_NAME_MAPPINGS"
    print(f"  ✓ Node registered: {NODE_DISPLAY_NAME_MAPPINGS['AdvancedOpenposeLoader']}")
except (ImportError, AssertionError) as e:
    print(f"  ✗ Failed node registration: {e}")
    sys.exit(1)

# Test 4: Verify INPUT_TYPES
print("\n[4/6] Testing INPUT_TYPES schema...")
try:
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    assert "required" in input_types, "Missing 'required' in INPUT_TYPES"
    assert "optional" in input_types, "Missing 'optional' in INPUT_TYPES"
    assert "model" in input_types["required"], "Missing 'model' input"
    assert "conditioning" in input_types["required"], "Missing 'conditioning' input"
    assert "vae" in input_types["required"], "Missing 'vae' input"
    assert "control_net" in input_types["required"], "Missing 'control_net' input"
    assert "folder_name" in input_types["required"], "Missing 'folder_name' input"
    print(f"  ✓ INPUT_TYPES valid with {len(input_types['required'])} required, {len(input_types['optional'])} optional inputs")
except (AttributeError, AssertionError) as e:
    print(f"  ✗ Failed INPUT_TYPES validation: {e}")
    sys.exit(1)

# Test 5: Verify RETURN_TYPES
print("\n[5/6] Testing RETURN_TYPES schema...")
try:
    return_types = AdvancedOpenposeLoader.RETURN_TYPES
    assert isinstance(return_types, tuple), "RETURN_TYPES should be a tuple"
    assert "MODEL" in return_types, "Missing 'MODEL' output"
    assert "CONDITIONING" in return_types, "Missing 'CONDITIONING' output"
    print(f"  ✓ RETURN_TYPES valid: {return_types}")
except (AttributeError, AssertionError) as e:
    print(f"  ✗ Failed RETURN_TYPES validation: {e}")
    sys.exit(1)

# Test 6: Import all src submodules
print("\n[6/6] Testing src submodule imports...")
src_modules = [
    "src.node",
    "src.pose_loader",
    "src.controlnet",
    "src.utils",
    "src.spatial_fade",
    "src.vae_encoder",
]

failed = False
for module_name in src_modules:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
    except ImportError as e:
        print(f"  ✗ {module_name}: {e}")
        failed = True

if failed:
    sys.exit(1)

print("\n" + "=" * 60)
print("All import tests passed! ✓")
print("=" * 60)