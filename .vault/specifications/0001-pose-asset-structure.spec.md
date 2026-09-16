---
type: specification
kind: standard
title: "Pose Asset Structure"
status: accepted
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [assets, pose, structure]
see_also: ["concepts/0007-pose-types-strengths.concept.md", "runbooks/0001-installation-setup.runbook.md"]
---

# Specification: Pose Asset Structure

## Goal

Define the required folder structure and file naming for pose assets consumed by the node.

## Structure

```
poses/
└── {pose_id}/
    ├── openpose.png
    ├── openpose_hand.png
    ├── openpose_full.png
    ├── canny.png
    ├── depth.png
    └── normal.png
```

## Requirements

- **Base path:** `/opt/comfyui/poses` (Docker) or `folder_paths.get_full_path("poses", "")`
- **Image size:** 1024×1024 (Flux2 Fun ControlNet standard)
- **Format:** PNG
- **Source:** OpenPose pose skeletons from [openposes.com](https://openposes.com/)
- **Pose ID:** Folder name used as the `folder_name` input to the node

## Phases

- **Phase 1:** Create pose folder with ID
- **Phase 2:** Generate all six pose type PNGs (openposes.com or custom)
- **Phase 3:** Resize to 1024×1024
- **Phase 4:** Test with node in ComfyUI
