"""Tests for image preprocessing and spatial fade masking."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from src.pose_preprocessor import (
    resize_to_1024,
    generate_spatial_fade_mask,
    apply_spatial_fade,
)


class TestResizeTo1024:
    """Test resize_to_1024 function."""

    def test_resize_small_image_to_1024(self):
        """Image smaller than 1024 should be upscaled."""
        image = torch.rand(1, 512, 512, 3)  # (batch, h, w, channels)
        result = resize_to_1024(image)
        assert result.shape == (1, 1024, 1024, 3)

    def test_resize_large_image_to_1024(self):
        """Image larger than 1024 should be downscaled."""
        image = torch.rand(1, 2048, 2048, 3)
        result = resize_to_1024(image)
        assert result.shape == (1, 1024, 1024, 3)

    def test_resize_non_square_image(self):
        """Non-square images should be resized to 1024x1024."""
        image = torch.rand(1, 800, 1200, 3)
        result = resize_to_1024(image)
        assert result.shape == (1, 1024, 1024, 3)

    def test_resize_batched_images(self):
        """Batched images should all be resized."""
        image = torch.rand(4, 640, 640, 3)  # batch of 4
        result = resize_to_1024(image)
        assert result.shape == (4, 1024, 1024, 3)

    def test_resize_preserves_dtype(self):
        """Dtype should be preserved through resizing."""
        image = torch.rand(1, 512, 512, 3, dtype=torch.float16)
        result = resize_to_1024(image)
        assert result.dtype == torch.float16

    def test_resize_single_channel(self):
        """Single-channel images should work."""
        image = torch.rand(1, 512, 512, 1)
        result = resize_to_1024(image)
        assert result.shape == (1, 1024, 1024, 1)

    def test_resize_image_already_1024(self):
        """Image already 1024x1024 should pass through unchanged."""
        image = torch.rand(1, 1024, 1024, 3)
        result = resize_to_1024(image)
        assert result.shape == (1, 1024, 1024, 3)
        # Should be essentially the same values
        assert torch.allclose(image, result, atol=1e-5)


class TestGenerateSpatialFadeMask:
    """Test spatial fade mask generation."""

    def test_mask_none_mode(self):
        """'none' mode should return all-ones mask (full strength)."""
        mask = generate_spatial_fade_mask(1024, 1024, fade_mode="none")
        assert mask.shape == (1, 1, 1024, 1024)
        assert torch.allclose(mask, torch.ones_like(mask))

    def test_mask_top_mode_gradient(self):
        """'top' mode should have gradient from top."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="top", fade_strength=0.5)
        # Top of mask should be lower value (faded)
        top_value = mask[0, 0, 0, 50].item()
        bottom_value = mask[0, 0, 99, 50].item()
        assert top_value < 1.0
        assert bottom_value == 1.0

    def test_mask_bottom_mode_gradient(self):
        """'bottom' mode should have gradient from bottom."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="bottom", fade_strength=0.5)
        top_value = mask[0, 0, 0, 50].item()
        bottom_value = mask[0, 0, 99, 50].item()
        assert top_value == 1.0
        assert bottom_value < 1.0

    def test_mask_left_mode_gradient(self):
        """'left' mode should have gradient from left."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="left", fade_strength=0.5)
        left_value = mask[0, 0, 50, 0].item()
        right_value = mask[0, 0, 50, 99].item()
        assert left_value < 1.0
        assert right_value == 1.0

    def test_mask_right_mode_gradient(self):
        """'right' mode should have gradient from right."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="right", fade_strength=0.5)
        left_value = mask[0, 0, 50, 0].item()
        right_value = mask[0, 0, 50, 99].item()
        assert left_value == 1.0
        assert right_value < 1.0

    def test_fade_strength_controls_region(self):
        """Higher fade_strength should create larger fade region."""
        weak_mask = generate_spatial_fade_mask(100, 100, fade_mode="top", fade_strength=0.1)
        strong_mask = generate_spatial_fade_mask(100, 100, fade_mode="top", fade_strength=0.9)

        # With stronger fade, more pixels should be < 1.0
        weak_faded = (weak_mask[0, 0] < 1.0).sum().item()
        strong_faded = (strong_mask[0, 0] < 1.0).sum().item()
        assert strong_faded > weak_faded

    def test_fade_strength_zero(self):
        """fade_strength=0 should give no fade (all 1.0)."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="top", fade_strength=0.0)
        assert torch.allclose(mask, torch.ones_like(mask))

    def test_fade_strength_one(self):
        """fade_strength=1 should fade entire image."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="top", fade_strength=1.0)
        # Top should be fully faded
        assert mask[0, 0, 0, 50].item() < 0.1

    def test_mask_values_in_range(self):
        """All mask values should be in [0, 1]."""
        for mode in ["none", "top", "bottom", "left", "right"]:
            mask = generate_spatial_fade_mask(100, 100, fade_mode=mode, fade_strength=0.5)
            assert mask.min() >= 0.0
            assert mask.max() <= 1.0

    def test_mask_dtype(self):
        """Mask should be float32."""
        mask = generate_spatial_fade_mask(100, 100, fade_mode="top")
        assert mask.dtype == torch.float32


class TestApplySpatialFade:
    """Test spatial fade application to control context."""

    def test_control_context_structure(self):
        """Should work with [control(16), mask(4), inpaint(16)] structure."""
        # Build control context: 16 + 4 + 16 = 36 channels
        control = torch.rand(1, 16, 100, 100)
        mask = torch.zeros(1, 4, 100, 100)  # empty mask channels
        inpaint = torch.rand(1, 16, 100, 100)
        control_context = torch.cat([control, mask, inpaint], dim=1)

        result = apply_spatial_fade(
            control_context, fade_mode="top", fade_strength=0.3
        )

        # Should maintain same shape
        assert result.shape == control_context.shape

    def test_only_mask_channels_modified(self):
        """Only mask channels (16-19) should be modified."""
        control = torch.rand(1, 16, 50, 50)
        mask = torch.zeros(1, 4, 50, 50)
        inpaint = torch.rand(1, 16, 50, 50)
        control_context = torch.cat([control, mask, inpaint], dim=1)

        result = apply_spatial_fade(
            control_context, fade_mode="top", fade_strength=0.3
        )

        # Control channels should be unchanged
        assert torch.allclose(result[:, :16, :, :], control)
        # Inpaint channels should be unchanged
        assert torch.allclose(result[:, 20:, :, :], inpaint)
        # Mask channels should now contain the fade mask
        mask_channels = result[:, 16:20, :, :]
        assert mask_channels.min() < 1.0  # Some fade applied

    def test_no_fade_preserves_original(self):
        """fade_mode='none' should preserve original control context."""
        control = torch.rand(1, 16, 50, 50)
        mask = torch.rand(1, 4, 50, 50)
        inpaint = torch.rand(1, 16, 50, 50)
        control_context = torch.cat([control, mask, inpaint], dim=1)

        result = apply_spatial_fade(control_context, fade_mode="none")
        assert torch.allclose(result, control_context)

    def test_batched_control_context(self):
        """Should work with batched control contexts."""
        control = torch.rand(4, 16, 50, 50)
        mask = torch.zeros(4, 4, 50, 50)
        inpaint = torch.rand(4, 16, 50, 50)
        control_context = torch.cat([control, mask, inpaint], dim=1)

        result = apply_spatial_fade(
            control_context, fade_mode="bottom", fade_strength=0.2
        )
        assert result.shape == control_context.shape

    def test_fade_strength_applied_to_mask_channels(self):
        """Different fade strengths should produce different mask values."""
        control = torch.rand(1, 16, 50, 50)
        inpaint = torch.rand(1, 16, 50, 50)

        mask_weak = apply_spatial_fade(
            torch.cat([control, torch.zeros(1, 4, 50, 50), inpaint], dim=1),
            fade_mode="top", fade_strength=0.1
        )
        mask_strong = apply_spatial_fade(
            torch.cat([control, torch.zeros(1, 4, 50, 50), inpaint], dim=1),
            fade_mode="top", fade_strength=0.9
        )

        weak_mask_channels = mask_weak[:, 16:20, :, :]
        strong_mask_channels = mask_strong[:, 16:20, :, :]

        # Stronger fade should have lower values on average in fade region
        assert strong_mask_channels.mean() < weak_mask_channels.mean()