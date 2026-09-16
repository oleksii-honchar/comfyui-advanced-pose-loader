---
type: container
c4_level: dataflow
title: "VAE Encoding Flow — Advanced OpenPose Loader"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T09:00:00Z"
updatedAt: "2026-09-16T09:00:00Z"
tags: [vae, encoding, latent, flow]
see_also: ["concepts/0003-vaes-and-latent-space.concept.md", "components/0001-vae-encoder.component.md"]
---

# VAE Encoding Flow

## Diagram

```mermaid
flowchart TD
    subgraph Input["Pose Image Input"]
        A[pose.png<br/>1024x1024 RGB]
    end
    
    subgraph Resize["Image Resize"]
        B[_resize_pose_image()]
        B1[Resize to 1024x1024<br/>if needed]
        B2[PIL Image operations]
    end
    
    subgraph VAE_Loader["VAE Loader"]
        C[_load_vae()]
        C1[_vae_cache check]
        C2{VAE in cache?}
        C3[Return cached VAE]
        C4[Load flux2-vae.safetensors]
        C5[Store in _vae_cache]
    end
    
    subgraph Encode["Latent Encoding"]
        D[_encode_pose_image()]
        D1[Preprocess image tensor]
        D2[VAE encode path]
        D3[latent = vae.encode(image)]
    end
    
    subgraph Output["Latent Output"]
        E[Latent Tensor<br/>[1,16,64,64]]
        E1[16× downsampling]
        E2[16 channels<br/>quantized]
        E3[Sharp latent rounding<br/>Flux2 format]
    end
    
    A --> B
    B --> B1
    B1 --> B2
    B2 --> C
    
    C --> C1
    C1 --> C2
    C2 -->|Yes| C3
    C2 -->|No| C4
    C4 --> C5
    C5 --> C3
    
    C3 --> D
    D --> D1
    D1 --> D2
    D2 --> D3
    
    D3 --> E
    E --> E1
    E1 --> E2
    E2 --> E3
    
    style A fill:#fff3e6,stroke:#cc6600
    style C2 fill:#fffacd,stroke:#ccaa00
    style E fill:#e6f3ff,stroke:#0066cc
    style D3 fill:#e6ffe6,stroke:#00cc00
```

## Notes

Detailed flow of the VAE encoding process from pose image input to latent tensor output. Shows the internal caching mechanism and the FLUX.2-specific latent format characteristics (16 channels, quantization, sharp rounding).
