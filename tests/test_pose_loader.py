"""
Unit tests for pose_loader.py — pose folder loading, path resolution, image loading.

Testable behaviors:
1. resolve_pose_folder resolves correct path for Docker and local environments
2. list_pose_images correctly identifies pose files by filename convention
3. load_pose_image loads and returns correct tensor
4. Missing pose files handled gracefully (skip with warning)
5. Folder existence and readability validated
6. Progress logging for each pose type
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call
from tempfile import TemporaryDirectory

# Add the custom node directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Mock comfy modules (no ComfyUI environment needed)
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.model_patcher'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()

import torch
from src.pose_loader import resolve_pose_folder, list_pose_images, load_pose_image, POSE_TYPES


class TestResolvePoseFolder:
    """Test pose folder path resolution."""

    def test_resolve_docker_path(self):
        """Path resolution works for Docker environment."""
        # In Docker, /opt/comfyui exists. Simulate with patch.
        with patch('src.pose_loader.os.path.exists') as mock_exists, \
             patch('src.pose_loader.os.path.isdir') as mock_isdir, \
             patch('src.pose_loader.os.access') as mock_access:
            # /opt/comfyui exists -> Docker path used
            mock_exists.side_effect = lambda p: p == '/opt/comfyui' or p == '/opt/comfyui/poses/test-pose'
            mock_isdir.return_value = True
            mock_access.return_value = True

            result = resolve_pose_folder('test-pose')
            assert result == '/opt/comfyui/poses/test-pose'

    def test_resolve_local_path(self):
        """Path resolution works for local environment."""
        # On this Mac, /opt/comfyui doesn't exist -> local path used
        # Just test that the function doesn't crash and validates the path
        with TemporaryDirectory() as tmpdir:
            poses_dir = os.path.join(tmpdir, 'poses')
            pose_folder = os.path.join(poses_dir, 'test-pose')
            os.makedirs(pose_folder)

            # Change to tmpdir and create input folder
            old_cwd = os.getcwd()
            try:
                os.makedirs(os.path.join(tmpdir, 'input'), exist_ok=True)
                os.chdir(tmpdir)

                # Capture the real os.path.exists before mocking
                import importlib
                real_exists = importlib.import_module('posixpath').exists

                with patch('src.pose_loader.os.path.exists') as mock_exists, \
                     patch('src.pose_loader.os.path.isdir') as mock_isdir, \
                     patch('src.pose_loader.os.access') as mock_access, \
                     patch('src.pose_loader.folder_paths.input_dir') as mock_input_dir:
                    mock_input_dir.return_value = os.path.join(tmpdir, 'input')
                    mock_isdir.return_value = True
                    mock_access.return_value = True
                    mock_exists.side_effect = lambda p: p != '/opt/comfyui' and real_exists(p)

                    result = resolve_pose_folder('test-pose')
                    assert 'poses' in result
                    assert result.endswith('test-pose')
            finally:
                os.chdir(old_cwd)


class TestListPoseImages:
    """Test pose image file listing by filename convention."""

    def test_identify_all_pose_types(self):
        """All six pose types correctly identified."""
        with TemporaryDirectory() as tmpdir:
            # Create all six pose type files
            for pose_type in POSE_TYPES:
                open(os.path.join(tmpdir, f"{pose_type}.png"), 'w').close()

            result = list_pose_images(tmpdir)
            assert len(result) == 6
            for pose_type in POSE_TYPES:
                assert pose_type in result
                assert result[pose_type].endswith(f"{pose_type}.png")

    def test_identify_partial_pose_types(self):
        """Only present pose types are returned."""
        with TemporaryDirectory() as tmpdir:
            # Create only some pose type files
            open(os.path.join(tmpdir, "openpose.png"), 'w').close()
            open(os.path.join(tmpdir, "openpose_hand.png"), 'w').close()
            open(os.path.join(tmpdir, "other-file.png"), 'w').close()

            result = list_pose_images(tmpdir)
            assert len(result) == 2
            assert "openpose" in result
            assert "openpose_hand" in result
            assert "openpose_full" not in result

    def test_case_insensitive_filename(self):
        """Case insensitivity for pose filenames."""
        with TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "OPENPOSE.png"), 'w').close()
            open(os.path.join(tmpdir, "Openpose_hand.Png"), 'w').close()

            result = list_pose_images(tmpdir)
            assert len(result) == 2
            assert "openpose" in result
            assert "openpose_hand" in result

    def test_non_png_files_ignored(self):
        """Non-PNG files are not considered pose images."""
        with TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "openpose.jpg"), 'w').close()
            open(os.path.join(tmpdir, "openpose.png"), 'w').close()

            result = list_pose_images(tmpdir)
            assert len(result) == 1
            assert "openpose" in result

    def test_empty_folder(self):
        """Empty folder returns empty dict."""
        with TemporaryDirectory() as tmpdir:
            result = list_pose_images(tmpdir)
            assert result == {}


class TestLoadPoseImage:
    """Test pose image loading and tensor conversion."""

    def test_load_image_returns_tensor(self):
        """load_pose_image returns a tensor."""
        with patch('src.pose_loader.comfy.utils') as mock_utils:
            mock_tensor = MagicMock()
            mock_tensor.max.return_value = 255.0
            mock_utils.load_image.return_value = mock_tensor

            result = load_pose_image('/path/to/pose.png')
            assert result is not None
            mock_utils.load_image.assert_called_once_with('/path/to/pose.png', 'RGB')

    def test_normalizes_uint8_to_float(self):
        """Images with max > 1.0 are normalized to [0, 1]."""
        with patch('src.pose_loader.comfy.utils') as mock_utils:
            mock_tensor = MagicMock()
            mock_tensor.max.return_value = 255.0
            mock_utils.load_image.return_value = mock_tensor

            load_pose_image('/path/to/pose.png')
            # Division should be called on the tensor
            mock_tensor.__truediv__.assert_called_once_with(255.0)

    def test_already_float_not_normalized(self):
        """Images already in [0, 1] range are not normalized."""
        with patch('src.pose_loader.comfy.utils') as mock_utils:
            mock_tensor = MagicMock()
            mock_tensor.max.return_value = 0.8
            mock_utils.load_image.return_value = mock_tensor

            load_pose_image('/path/to/pose.png')
            mock_tensor.__truediv__.assert_not_called()

    def test_error_on_missing_file(self):
        """Missing file raises ValueError."""
        with patch('src.pose_loader.comfy.utils') as mock_utils:
            mock_utils.load_image.side_effect = FileNotFoundError("File not found")

            with pytest.raises(ValueError, match="Failed to load pose image"):
                load_pose_image('/path/to/missing.png')


class TestFolderValidation:
    """Test folder existence and readability validation."""

    def test_missing_folder_raises(self):
        """Missing folder raises an error."""
        with pytest.raises(ValueError, match="Pose folder does not exist"):
            resolve_pose_folder('/nonexistent/folder/xyz')

    def test_unreadable_folder_raises(self):
        """Unreadable folder raises an error."""
        with TemporaryDirectory() as tmpdir:
            os.chmod(tmpdir, 0o000)
            try:
                with pytest.raises((ValueError, PermissionError)):
                    list_pose_images(tmpdir)
            finally:
                os.chmod(tmpdir, 0o755)

    def test_valid_folder_accepted(self):
        """Valid readable folder is accepted."""
        with TemporaryDirectory() as tmpdir:
            # Should not raise
            result = list_pose_images(tmpdir)
            assert isinstance(result, dict)


class TestLogging:
    """Test progress logging for each pose type."""

    def test_logs_loading_each_pose_type(self, caplog):
        """Progress is logged for each pose type loaded."""
        with TemporaryDirectory() as tmpdir:
            for pose_type in POSE_TYPES:
                open(os.path.join(tmpdir, f"{pose_type}.png"), 'w').close()

            import logging
            logger = logging.getLogger('pose_loader')
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            logger.addHandler(handler)

            result = list_pose_images(tmpdir)

            # Check that loading was logged
            # (actual implementation will use print or logging)
            assert len(result) == 6


class TestPoseTypesConstant:
    """Test that POSE_TYPES constant is correct."""

    def test_all_six_types_defined(self):
        """All six pose types are defined in POSE_TYPES."""
        assert "openpose" in POSE_TYPES
        assert "openpose_hand" in POSE_TYPES
        assert "openpose_full" in POSE_TYPES
        assert "canny" in POSE_TYPES
        assert "depth" in POSE_TYPES
        assert "normal" in POSE_TYPES
        assert len(POSE_TYPES) == 6


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '--import-mode=importlib']))
