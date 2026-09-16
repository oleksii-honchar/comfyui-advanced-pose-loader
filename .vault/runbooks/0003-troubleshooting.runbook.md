---
type: runbook
title: "Troubleshooting"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [troubleshooting, errors, debugging]
see_also: ["runbooks/0001-installation-setup.runbook.md"]
---

# Runbook: Troubleshooting

## Common Issues

### VAE not found
- Ensure `flux2-vae.safetensors` is in ComfyUI `vae/` directory
- Check file name matches exactly

### ControlNet not found
- Ensure `FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors` is in `controlnet/` directory
- Verify download completed successfully

### Pose folder not found
- Ensure pose folder exists at correct base path
- Verify `folder_name` input matches folder name exactly
- Check folder contains all six required pose PNGs

### Low VRAM errors
- Node loads models to CPU by default
- If GPU memory errors occur, reduce batch size or resolution
- Consider using fp8 model variant

### Import errors on startup
- Ensure `comfyui-flux2fun-controlnet` custom node is installed
- Restart ComfyUI after installation

### Pose not adhering in output
- Increase relevant pose type strength (e.g., `strength_openpose`)
- Remember: higher strength = tighter pose lock
- Verify pose assets are 1024×1024
- Check spatial fade settings

## Debug Mode

Set `debug` input to `true` for verbose logging during execution.
