"""Test spatial fade mask creation and application."""

import pytest
import torch

from src.spatial_fade import create_spatial_fade_mask, apply_spatial_fade_to_control_signal


class TestCreateSpatialFadeMask:
    """Test fade mask generation."""

    def test_mask_shape_matches_input(self):
        mask = create_spatial_fade_mask(1024, 1024, 0.1)
        assert mask.shape == (1, 1024, 1024)

    def test_mask_is_float32(self):
        mask = create_spatial_fade_mask(512, 512, 0.1)
        assert mask.dtype == torch.float32

    def test_mask_values_in_range(self):
        mask = create_spatial_fade_mask(256, 256, 0.1)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_fade_ratio_affects_mask(self):
        mask_small = create_spatial_fade_mask(256, 256, 0.05)
        mask_large = create_spatial_fade_mask(256, 256, 0.3)

        # Different fade ratios produce different masks
        assert not torch.allclose(mask_small, mask_large)

    def test_small_fade_ratio_has_sharp_transition(self):
        mask = create_spatial_fade_mask(256, 256, 0.01)
        # Very small fade ratio means sharp transition
        # Top-left should be strong, bottom-right weak
        top_left = mask[0, 0, 0].item()
        bottom_right = mask[0, 255, 255].item()
        assert top_left > 0.5
        assert bottom_right < 0.5


class TestApplySpatialFade:
    """Test fade application to control signals."""

    def test_fade_preserves_shape(self):
        control_signal = torch.randn(1, 64, 128, 128)
        fade_mask = create_spatial_fade_mask(128, 128, 0.1)

        result = apply_spatial_fade_to_control_signal(control_signal, fade_mask)
        assert result.shape == control_signal.shape

    def test_fade_reduces_magnitude(self):
        control_signal = torch.ones(1, 1, 64, 64)
        fade_mask = create_spatial_fade_mask(64, 64, 0.5)

        result = apply_spatial_fade_to_control_signal(control_signal, fade_mask)

        # Result should be <= original (mask is 0-1)
        assert (result.abs() <= control_signal.abs() + 1e-6).all()

    def test_fade_with_zero_mask_gives_zero(self):
        control_signal = torch.ones(1, 1, 64, 64)
        zero_mask = torch.zeros(1, 64, 64)

        result = apply_spatial_fade_to_control_signal(control_signal, zero_mask)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_fade_with_full_mask_preserves(self):
        control_signal = torch.ones(1, 1, 64, 64)
        full_mask = torch.ones(1, 64, 64)

        result = apply_spatial_fade_to_control_signal(control_signal, full_mask)
        assert torch.allclose(result, control_signal)