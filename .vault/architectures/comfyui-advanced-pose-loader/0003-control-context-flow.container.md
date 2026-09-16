---
type: container
c4_level: dataflow
title: "Control Context Data Flow — Advanced OpenPose Loader"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T09:00:00Z"
updatedAt: "2026-09-16T09:00:00Z"
tags: [dataflow, control, context, latent]
see_also: ["concepts/0002-control-context-architecture.concept.md", "concepts/0003-vaes-and-latent-space.concept.md"]
---

# Control Context Data Flow

## Diagram

```mermaid
flowchart TD
    subgraph Input_Poses["Pose Assets (Folder)"]
        A1[openpose.png<br/>1024x1024]
        A2[openpose_hand.png<br/>1024x1024]
        A3[openpose_full.png<br/>1024x1024]
        A4[canny.png<br/>1024x1024]
        A5[depth.png<br/>1024x1024]
        A6[normal.png<br/>1024x1024]
    end
    
    subgraph Pose_Encoding["Pose Encoding"]
        B1[Resize to 1024x1024]
        B2[VAE Encode<br/>flux2-vae.safetensors]
        B3[Latent Tensor<br/>[1,16,64,64]]
    end
    
    subgraph Control_Context_Builder["Control Context Builder"]
        C1[Control Latent<br/>channels 0-127]
        C2[Mask Channels<br/>channels 128-131<br/>spatial fade]
        C3[Inpaint Channels<br/>channels 132-259<br/>zeros]
        C4[260-Dim Context<br/>[B,seq,260]]
    end
    
    subgraph ControlNet_Wrapper["ControlNet Wrapper"]
        D1[Strength Inversion<br/>1.0 - user_strength]
        D2[ControlNetWrapper<br/>pre_run → apply_patch]
        D3[Scoped Monkey-Patch<br/>Flux.forward_orig]
    end
    
    subgraph Sampling_Loop["Sampling Loop (KSampler)"]
        E1[Each Sampling Step<br/>4 steps for Klein]
        E2[_generate_flux2fun_hints<br/>per active controlnet]
        E3[Run ControlNet on<br/>pose context]
        E4[_apply_flux2fun_hints<br/>onto main image tokens]
        E5[Sum hints across<br/>controlnets at same layer]
    end
    
    subgraph Flux_Model["FLUX.2-Klein Model"]
        F1[Patched forward_orig<br/>version-guarded]
        F2[Main Image Tokens<br/>ctrl_h × ctrl_w]
        F3[Reference Latent Tokens<br/>excluded from hints]
        F4[Denoising in Latent Space]
    end
    
    subgraph Output["Output"]
        G1[Final Denoised Latent]
        G2[VAEDecode<br/>flux2-vae.safetensors]
        G3[Generated Image<br/>1024x1024]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B1
    A6 --> B1
    
    B1 --> B2
    B2 --> B3
    
    B3 --> C1
    C1 --> C4
    C2 --> C4
    C3 --> C4
    
    C4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> E1
    
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
    
    E5 --> F1
    F1 --> F2
    F1 --> F3
    F2 --> F4
    F4 --> G1
    
    G1 --> G2
    G2 --> G3
    
    style A1 fill:#fff3e6,stroke:#cc6600
    style A2 fill:#fff3e6,stroke:#cc6600
    style A3 fill:#fff3e6,stroke:#cc6600
    style A4 fill:#fff3e6,stroke:#cc6600
    style A5 fill:#fff3e6,stroke:#cc6600
    style A6 fill:#fff3e6,stroke:#cc6600
    style C4 fill:#e6f3ff,stroke:#0066cc
    style D3 fill:#ffe6e6,stroke:#cc0000
    style F1 fill:#ffe6e6,stroke:#cc0000
    style G3 fill:#e6ffe6,stroke:#00cc00
```

## Notes

Detailed data flow diagram showing how pose images are transformed into pose-conditioned image generation. Highlights the 260-dim control context construction, strength inversion, scoped monkey-patch, and the separation of main image tokens from reference latent tokens.
