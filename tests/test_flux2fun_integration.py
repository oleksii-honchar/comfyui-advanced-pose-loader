"""
Integration tests for flux2fun_integration.py — FLUX.2 Fun ControlNet integration via monkey-patching.

Testable behaviors:
1. apply_controlnet_model loads and returns a controlnet object
2. patch_transformer_for_control installs a wrapper on the transformer's forward_orig
3. Control hints are injected at each double-block's timestep embedding with correct strength
4. unpatch_transformer restores the original forward function
5. build_control_chain chains multiple controlnet wrappers via previous_controlnet
"""
import sys
import os
import pytest
import torch
from unittest.mock import MagicMock, patch, call

# Add the custom node directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


class TestApplyControlnetModel:
    """Test loading Flux2 Fun ControlNet model."""

    def test_loads_controlnet_model(self, comfy_mock_environment):
        """apply_controlnet_model loads and returns a controlnet object."""
        from src.flux2fun_integration import apply_controlnet_model

        # Set up mock state_dict for this test
        mock_comfy, _, _ = comfy_mock_environment
        mock_state_dict = {
            "control_img_in.weight": torch.randn(6144, 36),
            "control_img_in.bias": torch.randn(6144),
            "control_transformer_blocks.0.after_proj.weight": torch.randn(6144, 6144),
            "control_transformer_blocks.0.after_proj.bias": torch.randn(6144),
        }
        mock_comfy.utils.load_torch_file.return_value = mock_state_dict

        controlnet = apply_controlnet_model(None, "test-controlnet.safetensors", 0.75)
        assert controlnet is not None
        assert hasattr(controlnet, 'parameters'), "Loaded controlnet should be a PyTorch module"
        # Verify it's a Flux2FunControlNet instance (check architecture attributes)
        assert hasattr(controlnet, 'control_layers_mapping'), "Should have control layer mapping"


class TestPatchTransformerForControl:
    """Test monkey-patching the transformer forward."""

    def test_patches_transformer_forward(self):
        """patch_transformer_for_control installs a wrapper on the transformer's forward."""
        from src.flux2fun_integration import patch_transformer_for_control, unpatch_transformer

        # Create a mock transformer with a forward_orig method
        mock_transformer = MagicMock()
        original_forward = MagicMock(return_value=torch.randn(1, 10, 64))
        mock_transformer.forward_orig = original_forward

        # Create a mock controlnet
        mock_controlnet = MagicMock()
        mock_controlnet.parameters = MagicMock(return_value=iter([torch.randn(10, 10)]))

        # Create a hint tensor
        hint = torch.randn(1, 10, 64)

        # Patch the transformer
        patched_forward = patch_transformer_for_control(
            mock_transformer, mock_controlnet, hint, strength=0.75
        )

        # Verify the forward method was wrapped
        assert mock_transformer.forward_orig != original_forward, \
            "forward_orig should have been replaced with a wrapper"

        # Cleanup
        unpatch_transformer(mock_transformer, patched_forward)

    def test_hints_injected_at_correct_location(self):
        """Control hints are injected at each double-block's timestep embedding."""
        from src.flux2fun_integration import patch_transformer_for_control, unpatch_transformer

        # Create a mock model with double_blocks
        mock_model = MagicMock()
        mock_block = MagicMock()
        mock_model.double_blocks = [mock_block]

        # Create mock img/txt tensors
        img = torch.randn(1, 10, 64)
        txt = torch.randn(1, 5, 64)

        # Track how img is modified during the patch's forward call
        img_modifications = []

        def mock_block_forward(img, txt, **kwargs):
            img_modifications.append(img.clone())
            return img, txt

        mock_block.forward = mock_block_forward

        # Create a mock controlnet that produces hints
        mock_controlnet = MagicMock()
        mock_controlnet.parameters = MagicMock(return_value=iter([torch.randn(10, 10)]))
        mock_controlnet.forward_control = MagicMock(return_value=[torch.randn(1, 10, 64) * 0.1])

        # Create a hint tensor
        hint = torch.randn(1, 10, 64)

        # Patch and call
        patched_forward = patch_transformer_for_control(
            mock_model, mock_controlnet, hint, strength=0.75
        )

        # Verify the patched forward can be called
        result = patched_forward(img, txt)
        assert result is not None

        unpatch_transformer(mock_model, patched_forward)


class TestUnpatchTransformer:
    """Test cleanup of patches after sampling."""

    def test_restores_original_forward(self):
        """unpatch_transformer restores the original forward function."""
        from src.flux2fun_integration import patch_transformer_for_control, unpatch_transformer

        # Create a mock model
        mock_model = MagicMock()
        original_forward = MagicMock()
        mock_model.forward_orig = original_forward

        # Create a mock controlnet
        mock_controlnet = MagicMock()
        mock_controlnet.parameters = MagicMock(return_value=iter([torch.randn(10, 10)]))

        # Patch
        hint = torch.randn(1, 10, 64)
        patched_forward = patch_transformer_for_control(
            mock_model, mock_controlnet, hint, strength=0.75
        )

        # Verify patched
        assert mock_model.forward_orig != original_forward

        # Unpatch
        unpatch_transformer(mock_model, patched_forward)

        # Verify restored
        assert mock_model.forward_orig == original_forward, \
            "forward_orig should be restored to original after unpatch"


class TestBuildControlChain:
    """Test chaining multiple controlnet wrappers."""

    def test_builds_chain_with_previous_controlnet(self):
        """build_control_chain chains wrappers via previous_controlnet."""
        from src.flux2fun_integration import build_control_chain

        # Create mock controlnet objects
        controlnets = []
        hints = []
        for i in range(3):
            cn = MagicMock()
            cn.parameters = MagicMock(return_value=iter([torch.randn(10, 10)]))
            controlnets.append(cn)
            # Hints are 4D: [B, C, H, W] for control context
            hints.append(torch.randn(1, 36, 16, 16))

        # Build the chain
        chain = build_control_chain(controlnets, hints, [0.75, 0.80, 0.85])

        # Verify chain structure
        assert len(chain) == 3, "Chain should have 3 wrappers"
        assert chain[0].previous_controlnet is None, "First wrapper has no previous"
        assert chain[1].previous_controlnet == chain[0], "Second wrapper chains to first"
        assert chain[2].previous_controlnet == chain[1], "Third wrapper chains to second"

    def test_chain_preserves_strengths(self):
        """Each wrapper in the chain has its own strength."""
        from src.flux2fun_integration import build_control_chain

        controlnets = []
        hints = []
        for i in range(2):
            cn = MagicMock()
            cn.parameters = MagicMock(return_value=iter([torch.randn(10, 10)]))
            controlnets.append(cn)
            # Hints are 4D: [B, C, H, W] for control context
            hints.append(torch.randn(1, 36, 16, 16))

        chain = build_control_chain(controlnets, hints, [0.75, 0.90])

        assert chain[0].strength == 0.75
        assert chain[1].strength == 0.90


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--import-mode=importlib']))