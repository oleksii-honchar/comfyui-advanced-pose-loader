"""
Unit tests for multi_pose_encoder.py — VAE encoding and 36-dim control context construction.

Testable behaviors:
1. encode_pose_image returns 16-channel latent tensor
2. build_control_context returns 36-dim context [16, 4, 16]
3. Latent features end up in control channels (0-15)
4. Spatial fade mask applied to mask channels (16-19)
5. Inpaint channels (20-35) are zeros
6. encode_all_poses handles multiple pose types
7. Error handling for missing pose types
8. resize_to_1024 preserves aspect ratio with padding
"""
import sys
import os
import pytest
import torch
from unittest.mock import MagicMock, patch

# Add the custom node directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Mock comfy modules (no ComfyUI environment needed)
mock_model_management = MagicMock()
mock_model_management.get_torch_device.return_value = 'cpu'
mock_model_management.should_use_bf16.return_value = False

mock_comfy = MagicMock()
mock_comfy.model_management = mock_model_management
mock_comfy.utils = MagicMock()

sys.modules['comfy'] = mock_comfy
sys.modules['comfy.model_management'] = mock_model_management
sys.modules['comfy.utils'] = mock_comfy.utils
sys.modules['folder_paths'] = MagicMock()

from src.multi_pose_encoder import encode_pose_image, build_control_context, encode_all_poses
from src.pose_preprocessor import generate_spatial_fade_mask, resize_to_1024


class TestEncodePoseImage:
    """Test VAE encoding of pose images to 16-channel latent."""

    def test_returns_16_channel_latent(self):
        """encode_pose_image returns a 16-channel latent tensor."""
        # Mock VAE
        mock_vae = MagicMock()
        mock_latent = torch.randn(1, 16, 128, 128)
        mock_vae.encode.return_value = mock_latent

        # Create a test pose image (BHWC format, [0,1] range)
        pose_image = torch.rand(1, 1024, 1024, 3)

        result = encode_pose_image(mock_vae, pose_image)

        assert result.shape[1] == 16, f"Expected 16 channels, got {result.shape[1]}"
        mock_vae.encode.assert_called_once()

    def test_encodes_input_image(self):
        """encode_pose_image calls vae.encode with the pose image."""
        mock_vae = MagicMock()
        mock_latent = torch.randn(1, 16, 128, 128)
        mock_vae.encode.return_value = mock_latent

        pose_image = torch.rand(1, 1024, 1024, 3)
        encode_pose_image(mock_vae, pose_image)

        # Verify vae.encode was called
        mock_vae.encode.assert_called_once()


class TestBuildControlContext:
    """Test 36-dim control context construction."""

    def test_returns_36_channel_context(self):
        """build_control_context returns a 36-channel tensor."""
        latent = torch.randn(1, 16, 128, 128)
        context = build_control_context(latent, fade_mode="none", fade_strength=0.5)

        assert context.shape[1] == 36, f"Expected 36 channels, got {context.shape[1]}"

    def test_control_channels_contain_latent(self):
        """Latent features end up in control channels (0-15)."""
        # Use a distinctive latent value to verify it's in control channels
        latent = torch.ones(1, 16, 128, 128) * 0.7
        context = build_control_context(latent, fade_mode="none", fade_strength=0.5)

        # Control channels (0-15) should contain the latent
        control_channels = context[:, :16, :, :]
        assert torch.allclose(control_channels, latent), "Control channels don't match latent"

    def test_inpaint_channels_are_zeros(self):
        """Inpaint channels (20-35) are zeros."""
        latent = torch.randn(1, 16, 128, 128)
        context = build_control_context(latent, fade_mode="none", fade_strength=0.5)

        inpaint_channels = context[:, 20:, :, :]
        assert torch.all(inpaint_channels == 0.0), "Inpaint channels are not zeros"

    def test_no_fade_mask_is_ones(self):
        """With fade_mode='none', mask channels (16-19) are all ones."""
        latent = torch.randn(1, 16, 128, 128)
        context = build_control_context(latent, fade_mode="none", fade_strength=0.5)

        mask_channels = context[:, 16:20, :, :]
        assert torch.all(mask_channels == 1.0), "Mask channels are not all ones with no fade"

    def test_spatial_fade_applied_to_mask(self):
        """With fade_mode='top', spatial fade is applied to mask channels (16-19)."""
        latent = torch.randn(1, 16, 128, 128)
        context = build_control_context(latent, fade_mode="top", fade_strength=0.5)

        mask_channels = context[:, 16:20, :, :]

        # Mask should not be all ones (fade should be visible)
        assert not torch.all(mask_channels == 1.0), "No fade applied to mask channels"

        # Fade from top: top should have lower values, bottom should be 1.0
        top_row = mask_channels[0, 0, 0, :]
        bottom_row = mask_channels[0, 0, -1, :]

        assert bottom_row.min() > top_row.max(), "Fade direction is wrong (should fade from top)"


class TestEncodeAllPoses:
    """Test encoding multiple pose types."""

    def test_encodes_multiple_pose_types(self):
        """encode_all_poses encodes each pose type and builds contexts."""
        mock_vae = MagicMock()
        mock_latent = torch.randn(1, 16, 128, 128)
        mock_vae.encode.return_value = mock_latent

        # Create test pose images for multiple types
        pose_images = {
            "openpose": torch.rand(1, 1024, 1024, 3),
            "openpose_hand": torch.rand(1, 1024, 1024, 3),
            "canny": torch.rand(1, 1024, 1024, 3),
        }
        pose_types = list(pose_images.keys())

        result = encode_all_poses(mock_vae, pose_images, pose_types, "none", 0.5)

        # Should have one context per pose type
        assert len(result) == 3, f"Expected 3 contexts, got {len(result)}"

        # All contexts should be 36 channels
        for pose_type, context in result:
            assert context.shape[1] == 36, f"Context for {pose_type} has wrong channels"

    def test_all_pose_types_processed(self):
        """All pose types in input are processed."""
        mock_vae = MagicMock()
        mock_latent = torch.randn(1, 16, 128, 128)
        mock_vae.encode.return_value = mock_latent

        pose_images = {
            "openpose": torch.rand(1, 1024, 1024, 3),
            "openpose_hand": torch.rand(1, 1024, 1024, 3),
            "openpose_full": torch.rand(1, 1024, 1024, 3),
            "canny": torch.rand(1, 1024, 1024, 3),
            "depth": torch.rand(1, 1024, 1024, 3),
            "normal": torch.rand(1, 1024, 1024, 3),
        }

        result = encode_all_poses(mock_vae, pose_images, list(pose_images.keys()), "none", 0.5)

        # Verify all pose types are in the result
        result_types = {pose_type for pose_type, _ in result}
        assert result_types == set(pose_images.keys()), "Not all pose types processed"

    def test_missing_pose_types_handled(self):
        """Missing pose types are handled gracefully (skipped with warning)."""
        mock_vae = MagicMock()
        mock_latent = torch.randn(1, 16, 128, 128)
        mock_vae.encode.return_value = mock_latent

        # Only provide some pose types
        pose_images = {
            "openpose": torch.rand(1, 1024, 1024, 3),
        }

        # Request all types but only provide one
        requested_types = ["openpose", "openpose_hand", "canny"]

        result = encode_all_poses(mock_vae, pose_images, requested_types, "none", 0.5)

        # Should only process the provided type
        assert len(result) == 1, f"Expected 1 context for provided type, got {len(result)}"
        assert result[0][0] == "openpose"


class TestResizeTo1024:
    """Test resize with aspect ratio preservation and padding."""

    def test_wide_image_padded_top_bottom(self):
        """Wide image (200x500) should scale to fit 1024 width, pad top/bottom."""
        img = torch.zeros((1, 200, 500, 3))
        # Red square in center
        img[:, 90:110, 240:260, 0] = 1.0

        resized = resize_to_1024(img)

        assert resized.shape == (1, 1024, 1024, 3), f"Wrong shape: {resized.shape}"

        # Scale factor = 1024/500 = 2.048
        # New height = 200 * 2.048 = 409 (int)
        # Padding top/bottom = (1024 - 409) // 2 = 307
        expected_scaled_h = int(200 * (1024 / 500))
        pad_top = (1024 - expected_scaled_h) // 2

        # Red square original: y[90:110], x[240:260]
        # After scale: y[90*2.048:110*2.048] = [184:225], x[240*2.048:260*2.048] = [491:532]
        # After padding: y[184+307:225+307] = [491:532]
        red_region = resized[0, 491:532, 491:532, 0]
        assert red_region.mean() > 0.5, f"Red square not at expected location: {red_region.mean()}"

        # Verify padding is black (0.0)
        top_pad = resized[0, 0:10, :, :]
        bottom_pad = resized[0, 1014:1024, :, :]
        assert top_pad.mean() < 0.1, f"Top padding not black: {top_pad.mean()}"
        assert bottom_pad.mean() < 0.1, f"Bottom padding not black: {bottom_pad.mean()}"

    def test_tall_image_padded_left_right(self):
        """Tall image (500x200) should scale to fit 1024 height, pad left/right."""
        img = torch.zeros((1, 500, 200, 3))
        img[:, 240:260, 90:110, 0] = 1.0

        resized = resize_to_1024(img)

        assert resized.shape == (1, 1024, 1024, 3)

        # Scale factor = 1024/500 = 2.048
        # New width = 200 * 2.048 = 409 (int)
        # Padding left/right = (1024 - 409) // 2 = 307
        expected_scaled_w = int(200 * (1024 / 500))
        pad_left = (1024 - expected_scaled_w) // 2

        # Red square original: y[240:260], x[90:110]
        # After scale: y[491:532], x[184:225]
        # After padding: y[491:532], x[184+307:225+307] = [491:532]
        red_region = resized[0, 491:532, 491:532, 0]
        assert red_region.mean() > 0.5

        # Verify left/right padding
        left_pad = resized[0, :, 0:10, :]
        right_pad = resized[0, :, 1014:1024, :]
        assert left_pad.mean() < 0.1
        assert right_pad.mean() < 0.1

    def test_already_1024_no_resize(self):
        """Image already 1024x1024 should be returned unchanged."""
        img = torch.ones((1, 1024, 1024, 3))
        resized = resize_to_1024(img)
        assert torch.allclose(img, resized), "Already-1024 image was modified"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '--import-mode=importlib']))
