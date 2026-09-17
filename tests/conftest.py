"""Shared pytest fixtures for AdvancedOpenposeLoader tests.

Sets up ComfyUI environment mocks so tests can run without a real
ComfyUI installation.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock

# Add the custom node directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


@pytest.fixture(autouse=True)
def comfy_mock_environment():
    """Set up ComfyUI environment mocks for all tests."""
    # Mock comfy modules
    mock_comfy = MagicMock()
    mock_comfy.utils = MagicMock()
    mock_comfy.sd = MagicMock()
    mock_comfy.model_patcher = MagicMock()
    
    mock_model_management = MagicMock()
    mock_model_management.get_torch_device.return_value = 'cpu'
    mock_model_management.should_use_bf16.return_value = False
    mock_comfy.model_management = mock_model_management
    
    mock_folder_paths = MagicMock()
    mock_folder_paths.input_dir.return_value = '/tmp/comfyui/input'
    
    sys.modules['comfy'] = mock_comfy
    sys.modules['comfy.utils'] = mock_comfy.utils
    sys.modules['comfy.sd'] = mock_comfy.sd
    sys.modules['comfy.model_patcher'] = mock_comfy.model_patcher
    sys.modules['comfy.model_management'] = mock_model_management
    sys.modules['folder_paths'] = mock_folder_paths
    
    yield mock_comfy, mock_model_management, mock_folder_paths
    
    # Cleanup: remove mocked modules
    for mod in ['comfy', 'comfy.utils', 'comfy.sd', 'comfy.model_patcher', 
                'comfy.model_management', 'folder_paths']:
        if mod in sys.modules:
            del sys.modules[mod]
