"""Pytest configuration and fixtures."""
import pytest
from unittest.mock import MagicMock, patch
import torch
import sys

# Mock comfy modules before any imports that need them
mock_comfy = MagicMock()
mock_comfy.utils = MagicMock()
mock_comfy.sd = MagicMock()
sys.modules['comfy'] = mock_comfy
sys.modules['comfy.utils'] = mock_comfy.utils
sys.modules['comfy.sd'] = mock_comfy.sd


@pytest.fixture(autouse=True)
def mock_torch_cuda():
    """Mock torch CUDA operations for tests without GPU."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.device', return_value=torch.device('cpu')):
            yield


@pytest.fixture
def mock_controlnet():
    """Create a mock ControlNet object."""
    controlnet = MagicMock()
    controlnet.strength = 0.8
    controlnet.strength_begin = 0.0
    controlnet.strength_end = 1.0
    return controlnet


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = MagicMock()
    model["conds"] = [{"text": "positive prompt"}]
    return model


@pytest.fixture
def mock_conditioning():
    """Create fake conditioning list."""
    return [
        {"text": "positive prompt", "control": None},
        {"text": "negative prompt", "control": None},
    ]