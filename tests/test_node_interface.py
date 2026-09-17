"""Node interface tests for AdvancedOpenposeLoader.

Verifies the redesigned interface with folder input, six independent
strength sliders, and dropdowns for VAE/ControlNet models.
"""
import sys
import os
import torch
from unittest.mock import MagicMock

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Mock ComfyUI modules (not available in test environment)
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.model_patcher'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()

from advanced_openpose_loader import AdvancedOpenposeLoader


def test_input_types_required_fields():
    """Test that all required inputs are present in the correct order."""
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    required = input_types["required"]
    required_keys = list(required.keys())

    expected = [
        "vae", "model", "conditioning", "control_net",
        "folder_name", "strength_openpose", "strength_openpose_hand",
        "strength_openpose_full", "strength_canny", "strength_depth",
        "strength_normal"
    ]

    assert required_keys == expected, f"Required inputs mismatch.\nExpected: {expected}\nGot: {required_keys}"


def test_input_types_required_field_types():
    """Test the correct types for each required input."""
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    required = input_types["required"]

    assert required["vae"][0] == "COMBO", "vae should be COMBO dropdown"
    assert required["model"][0] == "MODEL", "model should be MODEL"
    assert required["conditioning"][0] == "CONDITIONING", "conditioning should be CONDITIONING"
    assert required["control_net"][0] == "COMBO", "control_net should be COMBO dropdown"
    assert required["folder_name"][0] == "STRING", "folder_name should be STRING"
    assert required["strength_openpose"][0] == "FLOAT", "strength_openpose should be FLOAT"
    assert required["strength_openpose_hand"][0] == "FLOAT", "strength_openpose_hand should be FLOAT"
    assert required["strength_openpose_full"][0] == "FLOAT", "strength_openpose_full should be FLOAT"
    assert required["strength_canny"][0] == "FLOAT", "strength_canny should be FLOAT"
    assert required["strength_depth"][0] == "FLOAT", "strength_depth should be FLOAT"
    assert required["strength_normal"][0] == "FLOAT", "strength_normal should be FLOAT"


def test_strength_sliders_ranges_and_defaults():
    """Test that all strength sliders have correct ranges and defaults."""
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    required = input_types["required"]

    default_strengths = {
        "strength_openpose": 0.75,
        "strength_openpose_hand": 0.80,
        "strength_openpose_full": 0.85,
        "strength_canny": 0.0,
        "strength_depth": 0.0,
        "strength_normal": 0.0,
    }

    for key, expected_default in default_strengths.items():
        field = required[key]
        assert field[0] == "FLOAT", f"{key} should be FLOAT"
        assert field[1]["default"] == expected_default, f"{key} default should be {expected_default}"
        assert field[1]["min"] == 0.0, f"{key} min should be 0.0"
        assert field[1]["max"] == 1.0, f"{key} max should be 1.0"
        assert field[1]["step"] == 0.01, f"{key} step should be 0.01"


def test_optional_inputs():
    """Test that optional inputs are correct."""
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    optional = input_types["optional"]
    optional_keys = list(optional.keys())

    assert optional_keys == ["spatial_fade", "spatial_fade_strength", "debug"], \
        f"Optional inputs mismatch. Expected: ['spatial_fade', 'spatial_fade_strength', 'debug'], Got: {optional_keys}"

    # spatial_fade dropdown options
    assert optional["spatial_fade"][0] == "COMBO"
    fade_options = optional["spatial_fade"][1]["options"]
    assert fade_options == ["none", "top", "bottom", "left", "right"], f"Fade options: {fade_options}"
    assert optional["spatial_fade"][1]["default"] == "none"

    # spatial_fade_strength
    assert optional["spatial_fade_strength"][0] == "FLOAT"
    assert optional["spatial_fade_strength"][1]["default"] == 0.5

    # debug
    assert optional["debug"][0] == "BOOLEAN"
    assert optional["debug"][1]["default"] == False


def test_return_types():
    """Test that return types are correct."""
    assert AdvancedOpenposeLoader.RETURN_TYPES == ("MODEL", "CONDITIONING", "CONDITIONING")
    assert AdvancedOpenposeLoader.RETURN_NAMES == ("model", "positive", "negative")


def test_function_attribute():
    """Test that the FUNCTION attribute is set."""
    assert AdvancedOpenposeLoader.FUNCTION == "apply_pose_conditioning"


def test_vae_dropdown_auto_discovers_models():
    """Test that VAE dropdown lists available VAE models."""
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    required = input_types["required"]
    vae_options = required["vae"][1]["options"]

    # Should list flux2-vae.safetensors
    assert "flux2-vae.safetensors" in vae_options, f"flux2-vae.safetensors not in VAE options: {vae_options}"


def test_controlnet_dropdown_auto_discovers_models():
    """Test that ControlNet dropdown lists available controlnet models."""
    input_types = AdvancedOpenposeLoader.INPUT_TYPES()
    required = input_types["required"]
    cn_options = required["control_net"][1]["options"]

    # Should list at least one controlnet model
    assert len(cn_options) > 0, "ControlNet dropdown should have at least one model"


def test_function_executes_complete_pipeline():
    """Test that the FUNCTION method executes the complete pipeline."""
    from unittest.mock import patch, MagicMock, call

    with patch("src.node.resolve_pose_folder") as mock_resolve, \
         patch("src.node.list_pose_images") as mock_list, \
         patch("src.node.load_pose_image") as mock_load, \
         patch("src.node.resize_to_1024") as mock_resize, \
         patch("src.node.encode_pose_image") as mock_encode, \
         patch("src.node.build_control_context") as mock_context, \
         patch("src.node.apply_controlnet_model") as mock_load_cn, \
         patch("src.node.build_control_chain") as mock_chain, \
         patch("src.node.patch_transformer_for_control") as mock_patch, \
         patch("comfy.model_management.get_torch_device") as mock_device, \
         patch("folder_paths.get_full_path") as mock_full_path:

        # Configure mocks
        mock_resolve.return_value = "/mock/poses/test"
        mock_list.return_value = {"openpose": "/mock/poses/test/openpose.png"}
        mock_load.return_value = MagicMock()
        mock_resize.return_value = MagicMock()
        mock_encode.return_value = torch.randn(1, 16, 128, 128)
        mock_context.return_value = torch.randn(1, 36, 128, 128)
        mock_load_cn.return_value = MagicMock()
        mock_chain.return_value = [MagicMock()]
        mock_device.return_value = "cpu"
        mock_full_path.side_effect = lambda kind, name: f"/mock/{kind}/{name}"

        node = AdvancedOpenposeLoader()
        model = MagicMock()
        conditioning = [("cond_tensor", {"params": {}})]

        result = node.apply_pose_conditioning(
            vae="flux2-vae.safetensors",
            model=model,
            conditioning=conditioning,
            control_net="test-controlnet.safetensors",
            folder_name="test",
            strength_openpose=0.75,
            strength_openpose_hand=0.80,
            strength_openpose_full=0.85,
            strength_canny=0.0,
            strength_depth=0.0,
            strength_normal=0.0,
            spatial_fade="none",
            spatial_fade_strength=0.5,
            debug=False,
        )

        # Verify the pipeline was executed
        mock_resolve.assert_called_once_with("test")
        mock_list.assert_called_once()
        mock_load.assert_called_once()
        mock_resize.assert_called_once()
        mock_encode.assert_called_once()
        mock_context.assert_called_once()
        mock_load_cn.assert_called_once()
        mock_chain.assert_called_once()
        mock_patch.assert_called_once()

        # Verify return values
        assert len(result) == 3, "Should return 3 values: (model, positive, negative)"
        assert result[0] == model, "First return value should be the unchanged model"
        assert result[1] == conditioning, "Second return value should be positive conditioning"
        assert result[2] == conditioning, "Third return value should be negative conditioning"


if __name__ == "__main__":
    print("Running node interface tests...")
    tests = [
        ("Input types - required fields", test_input_types_required_fields),
        ("Input types - required field types", test_input_types_required_field_types),
        ("Strength sliders - ranges and defaults", test_strength_sliders_ranges_and_defaults),
        ("Optional inputs", test_optional_inputs),
        ("Return types", test_return_types),
        ("Function attribute", test_function_attribute),
        ("VAE dropdown auto-discovers models", test_vae_dropdown_auto_discovers_models),
        ("ControlNet dropdown auto-discovers models", test_controlnet_dropdown_auto_discovers_models),
        ("Function executes complete pipeline", test_function_executes_complete_pipeline),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
