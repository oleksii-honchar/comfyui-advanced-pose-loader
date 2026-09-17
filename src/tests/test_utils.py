"""AdvancedOpenposeLoader - cheap unit tests (no real model loading)."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils import get_strength_defaults, cleanup_pose


class TestGetStrengthDefaults:
    """Test strength default value generation."""

    def test_returns_correct_number_of_values(self):
        model = MagicMock()
        result = get_strength_defaults(model)
        assert len(result) == 6

    def test_returns_reasonable_defaults(self):
        model = MagicMock()
        strength, strength_begin, strength_end, body, face, hand = get_strength_defaults(model)

        assert 0.0 <= strength <= 1.0
        assert 0.0 <= strength_begin <= 1.0
        assert 0.0 <= strength_end <= 1.0
        assert 0.0 <= body <= 1.0
        assert 0.0 <= face <= 1.0
        assert 0.0 <= hand <= 1.0


class TestCleanupPose:
    """Test pose cleanup and restoration logic."""

    def test_restores_original_values(self):
        control_net = MagicMock()
        control_net.strength = 0.5
        control_net.strength_begin = 0.1
        control_net.strength_end = 0.9

        pose_conditioning = None

        cleanup_pose(control_net, 0.8, 0.0, 1.0, pose_conditioning)

        assert control_net.strength == 0.8
        assert control_net.strength_begin == 0.0
        assert control_net.strength_end == 1.0

    def test_clears_pose_conditioning(self):
        control_net = MagicMock()

        pose_conditioning = [
            {"control": MagicMock(), "text": "prompt1"},
            {"control": MagicMock(), "text": "prompt2"},
        ]

        cleanup_pose(control_net, 0.8, 0.0, 1.0, pose_conditioning)

        for item in pose_conditioning:
            assert item["control"] is None

    def test_handles_empty_pose_conditioning(self):
        control_net = MagicMock()
        pose_conditioning = []

        cleanup_pose(control_net, 0.8, 0.0, 1.0, pose_conditioning)
        # Should not raise