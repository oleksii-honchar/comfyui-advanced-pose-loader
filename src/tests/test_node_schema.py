"""Test the AdvancedOpenposeLoader node class and INPUT_TYPES."""

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock comfy modules
mock_comfy = MagicMock()
mock_comfy.utils = MagicMock()
mock_comfy.sd = MagicMock()
sys.modules['comfy'] = mock_comfy
sys.modules['comfy.utils'] = mock_comfy.utils
sys.modules['comfy.sd'] = mock_comfy.sd

from src.node import AdvancedOpenposeLoader


class TestNodeInputTypes:
    """Test the node's INPUT_TYPES schema."""

    def test_has_required_inputs(self):
        input_types = AdvancedOpenposeLoader.INPUT_TYPES()
        assert "required" in input_types
        assert "model" in input_types["required"]
        assert "control_net" in input_types["required"]
        assert "pose_image" in input_types["required"]

    def test_has_optional_inputs(self):
        input_types = AdvancedOpenposeLoader.INPUT_TYPES()
        assert "optional" in input_types
        assert "strength" in input_types["optional"]
        assert "use_spatial_fade" in input_types["optional"]

    def test_returns_classmethod(self):
        input_types = AdvancedOpenposeLoader.INPUT_TYPES()
        assert isinstance(input_types, dict)

    def test_function_defined(self):
        assert hasattr(AdvancedOpenposeLoader, "FUNCTION")
        assert AdvancedOpenposeLoader.FUNCTION == "apply_pose_conditioning"

    def test_category_defined(self):
        assert hasattr(AdvancedOpenposeLoader, "CATEGORY")
        assert AdvancedOpenposeLoader.CATEGORY == "conditioning/controlnet"

    def test_return_types_defined(self):
        assert hasattr(AdvancedOpenposeLoader, "RETURN_TYPES")
        assert AdvancedOpenposeLoader.RETURN_TYPES == ("CONDITIONING",)


class TestNodeInstantiation:
    """Test node can be instantiated."""

    def test_node_instantiates(self):
        node = AdvancedOpenposeLoader()
        assert node is not None