---
type: component
c4_level: component
title: "Pose Asset Handler Component"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [assets, pose, loading, component]
see_also: ["concepts/0007-pose-types-strengths.concept.md"]
---

# Component: Pose Asset Handler

## Diagram

```mermaid
C4Component
    Container(advanced_loader, "Advanced OpenPose Loader", "Custom Node")
    Component(path_resolver, "Path Resolver", "_get_poses_base_path()")
    Component(pose_loader, "Pose Loader", "_load_pose_image()")
    Component_db(pose_folder, "Pose Assets", "poses/{pose_id}/{type}.png")
    
    Rel(path_resolver, pose_loader, "Provides base path")
    Rel(pose_loader, pose_folder, "Loads pose images from folder")
```

## Elements

| ID | Name | Type | Description |
|----|------|------|-------------|
| advanced_loader | Advanced OpenPose Loader | Container | Parent custom node |
| path_resolver | Path Resolver | Component | `_get_poses_base_path()` — resolves base path (Docker fallback: /opt/comfyui/poses) |
| pose_loader | Pose Loader | Component | `_load_pose_image()` — loads specific pose image from poses/{folder_name}/{pose_type}.png |
| pose_folder | Pose Assets | ComponentDb | Folder-per-pose structure with six pose types |

## Notes

Encapsulates pose asset loading from the structured folder system. Handles path resolution for Docker and non-Docker environments.
