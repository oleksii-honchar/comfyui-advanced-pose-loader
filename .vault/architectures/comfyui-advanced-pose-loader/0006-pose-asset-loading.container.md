---
type: container
c4_level: dataflow
title: "Pose Asset Loading Flow — Advanced OpenPose Loader"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T09:00:00Z"
updatedAt: "2026-09-16T09:00:00Z"
tags: [assets, pose, loading, flow]
see_also: ["specifications/0001-pose-asset-structure.spec.md", "components/0003-pose-asset-handler.component.md"]
---

# Pose Asset Loading Flow

## Diagram

```mermaid
flowchart TD
    subgraph Input["Node Input"]
        A[folder_name input<br/>e.g., 'my-pose-123']
    end
    
    subgraph PathResolution["Path Resolution"]
        B[_get_poses_base_path()]
        B1{Docker environment?}
        B2[/opt/comfyui/poses]
        B3[folder_paths.get_full_path<br/>('poses', '')]
    end
    
    subgraph FileLoading["File Loading"]
        C[_load_pose_image(folder_name, pose_type)]
        C1[Construct path:<br/>poses/{folder_name}/{pose_type}.png]
        C2[PIL Image.open(path)]
        C3{Image loaded?}
        C4[Return image]
        C5[Error: Pose image not found]
    end
    
    subgraph PoseTypes["Six Pose Types"]
        D1[openpose.png]
        D2[openpose_hand.png]
        D3[openpose_full.png]
        D4[canny.png]
        D5[depth.png]
        D6[normal.png]
    end
    
    subgraph Output["Loaded Images"]
        E[Six 1024x1024 PNG images]
    end
    
    A --> B
    B --> B1
    B1 -->|Yes| B2
    B1 -->|No| B3
    B2 --> C
    B3 --> C
    
    D1 --> C1
    D2 --> C1
    D3 --> C1
    D4 --> C1
    D5 --> C1
    D6 --> C1
    
    C1 --> C2
    C2 --> C3
    C3 -->|Yes| C4
    C3 -->|No| C5
    
    C4 --> E
    
    style B1 fill:#fffacd,stroke:#ccaa00
    style C3 fill:#fffacd,stroke:#ccaa00
    style D1 fill:#fff3e6,stroke:#cc6600
    style D2 fill:#fff3e6,stroke:#cc6600
    style D3 fill:#fff3e6,stroke:#cc6600
    style D4 fill:#fff3e6,stroke:#cc6600
    style D5 fill:#fff3e6,stroke:#cc6600
    style D6 fill:#fff3e6,stroke:#cc6600
```

## Notes

Flow diagram showing how pose assets are loaded from the structured folder system. Handles both Docker and non-Docker path resolution and loads all six required pose type images.
