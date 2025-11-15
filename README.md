# **ComfyUI-FlashVSR ⚡**

A powerful ComfyUI custom node based on the FlashVSR model, enabling real-time diffusion-based video super-resolution for streaming applications.

https://github.com/user-attachments/assets/1d1528c5-e3c1-487f-9c29-267ddb817809

## **Features**
**High-Quality Video Upscaling**: Utilizes the advanced FlashVSR model to upscale videos to 2x or 4x resolution.  
* **Multiple Model Versions**:  
  * **Full (Best Quality)**: Highest quality results with significant VRAM usage.  
  * **Tiny (Fast)**: Balanced quality and speed for faster processing.  
  * **Tiny Long (Low VRAM)**: Optimized for GPUs with limited VRAM, ideal for long videos.  
* **SageAttention Optimization** (Optional): Automatic ~20-30% speedup when SageAttention is installed. Falls back gracefully if not available.
* **Intelligent Tiling**: Supports `enable_tiling` to process high-resolution videos efficiently on low-VRAM GPUs.  
* **Automatic Model Download**: On the first run, the node will automatically download the required `.safetensors` models from Hugging Face ([1038lab/FlashVSR](https://huggingface.co/1038lab/FlashVSR)).  
* **Audio Passthrough**: Maintains the original audio during video frame processing, ensuring synchronization and quality preservation.

## **News & Updates**
**2025/11/15**: FlashVSR 1.1 Model Update + Frame Duplication Fix ( [update.md](https://github.com/1038lab/ComfyUI-FlashVSR/blob/main/update.md#v110-20251115) )
- Added new model: Wan2_1-T2V-1.1_3B_FlashVSR_fp32.safetensors
- Improved T2V → VSR quality, stability, details
- Applied frame duplication fix (Issue #3)
- Updated UPDATE.md accordingly
  
**2025/10/24**: Initial release of ComfyUI-FlashVSR.  
- Added **FlashVSR ⚡** and **FlashVSR Advanced ⚡** nodes.  
- Implemented automatic model download from Hugging Face (1038lab/FlashVSR).  
- Supports `.safetensors` models, audio passthrough, and tiling for low VRAM.

## **Installation**

### **Method 1: Install via ComfyUI Manager (Recommended)**

1. Start ComfyUI.  
2. Click the "Manager" button in the sidebar → "Install Custom Nodes".  
3. Search for **ComfyUI-FlashVSR**.  
4. Click the "Install" button.  
5. Restart ComfyUI.

### **Method 2: Clone the Repository**

1. Navigate to your ComfyUI `custom_nodes` directory.  
2. Run:  
```bash
   git clone https://github.com/1038lab/ComfyUI-FlashVSR.git
```

3. Restart ComfyUI.

### **Method 3: Install via Comfy CLI**

1. Ensure that `comfy-cli` is installed with:

   ```bash
   pip install comfy-cli
   ```
2. Install ComfyUI-FlashVSR using:

   ```bash
   comfy node install ComfyUI-FlashVSR
   ```
3. Restart ComfyUI.

### **Method 4: Manually Download the Models**

* The models will be automatically downloaded to `ComfyUI/models/FlashVSR/` on the first run.
* To manually download the models, visit [1038lab/FlashVSR on Hugging Face](https://huggingface.co/1038lab/FlashVSR) and download the `.safetensors` files into the `ComfyUI/models/FlashVSR/` folder.

| Model File | Purpose |
|-----------|---------|
| Wan2_1-T2V-1.1_3B_FlashVSR_fp32.safetensors | **New FlashVSR 1.1 Main Diffusion Model** |
| Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors | Previous FlashVSR 1.0 Main Model |
| Wan2.1_VAE.safetensors | Video VAE |
| Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors | Low-Quality Projection |
| Wan2_1_FlashVSR_TCDecoder_fp32.safetensors | Tiny Model Decoder |


> **📖 For optional performance optimization (~20-30% speedup), see [SageAttention Installation Guide](./SAGEATTENTION_INSTALL.md)**

## **Usage**

This node processes **image (frame) sequences**. For a complete video workflow, combine it with other nodes in ComfyUI.

* **Load**: Use a video loader (e.g., **VHS - Video Load**) to load video frames and audio.
* **Process**: Connect the frames and audio to the **FlashVSR node**.
* **Save**: Use a video combiner (e.g., **VHS - Video Combine**) to combine the output frames and audio into a final upscaled video.

### **FlashVSR Nodes**

### **Optional Settings 💡 Tips**

| Optional Setting             | Description                                                                                | Tips                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **preset** (Simple)          | Choose between: `Fast` (Tiny model), `Balanced` (Tiny model), `High Quality` (Full model). | `High Quality` requires significant VRAM. Consider using the Advanced node if you face OOM errors. |
| **model_version** (Advanced) | Options: `Tiny (Fast)`, `Tiny Long (Low VRAM)`, `Full (Best Quality)`.                     | `Full` offers the best quality, while `Tiny Long` is optimized for minimal VRAM.                   |
| **enable_tiling** (Advanced) | Breaks the video into smaller chunks to save VRAM.                                         | Enable this if you encounter OOM errors, especially with the Full model at 4x scale.               |
| **speed_optimization**       | Optimizes for processing speed. Higher values yield faster results.                        | Default is `2.0`.                                                                                  |
| **quality_boost**            | Boosts quality at the cost of VRAM usage. Higher values yield better results.              | Default is `2.0`. The Full model can handle `3.0` without crashing.                                |
| **Input Frames**             | The video frames to process.                                                               | Requires at least **21 frames** for initialization.                                                |
| **4x Upscaling**             | Optimized for 4x upscaling.                                                                | 2x upscaling is supported, but 4x generally provides better results.                               |
| **sageattention** (Advanced) | Enable/Disable SageAttention optimization.                                                 | Enabled by default. Provides ~20-30% speedup if `sageattention` package is installed.              |

## **About FlashVSR Model**

**FlashVSR** is a real-time diffusion-based video super-resolution model. It is designed to provide high-quality upscaling, particularly suited for streaming applications. The `.safetensors` versions are included for enhanced compatibility and security.

## **Requirements**

* **ComfyUI**
* **Python 3.10+**
* **Required packages**:
  * `torch >= 2.0.0`
  * `torchvision >= 0.15.0`
  * `safetensors >= 0.4.0`
  * `huggingface_hub >= 0.19.0`
  * `einops >= 0.6.0`
  * `numpy >= 1.24.0`
  * `tqdm >= 4.65.0`
  * `pillow >= 9.5.0`

* **Optional packages** (for performance boost):
  * `sageattention >= 1.0.0` - Provides ~20-30% speedup (see [Optional Performance Optimization](#optional-performance-optimization))
  * `triton >= 2.1.0` - Required by SageAttention

These packages are typically included in ComfyUI environments. If you encounter an import error, run:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 safetensors>=0.4.0 huggingface-hub>=0.19.0 einops>=0.6.0
```

### **Optional Performance Optimization**

For an automatic ~20-30% performance boost, you can install SageAttention:

```bash
pip install sageattention triton
```

**Note**: 
- SageAttention requires a CUDA-capable GPU and may conflict with some ComfyUI environments.
- **For detailed installation instructions and troubleshooting**, see [SageAttention Installation Guide](./SAGEATTENTION_INSTALL.md).
- If you encounter issues after installing SageAttention, you can:
  1. Disable it in the **FlashVSR ⚡ Advanced** node by setting `sageattention` to `disable`.
  2. Or uninstall it: `pip uninstall sageattention triton`
- The node will work perfectly fine without SageAttention installed - it will automatically fall back to standard PyTorch attention.

## **Troubleshooting**

* **FileNotFoundError: Missing `Wan2.1_VAE.safetensors`**:

  * This error usually occurs when the model download fails or is skipped.
  * **Fix**: Delete the `FlashVSR` folder in `ComfyUI/models/`, then restart ComfyUI to trigger the automatic download again.

* **Out-of-Memory (OOM) Error / CUDAMallocAsyncAllocator.cpp error**:

  * Occurs when VRAM is exhausted, especially with the High Quality preset or Full model at 4x scale.
  * **Fix**: Use the **FlashVSR Advanced ⚡** node and enable `enable_tiling` to reduce VRAM usage.

## **Credits**

* **FlashVSR**: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
* **Original HF Models**: [JunhaoZhuang/FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR)
* **Safetensors Models**: [1038lab/FlashVSR](https://huggingface.co/1038lab/FlashVSR)
* **Created by**: [AILab](https://github.com/1038lab)

## **Star History**

If this custom node helps you or if you appreciate the work, please give a ⭐ on this repo! It’s a great encouragement for my efforts!

## **License**

[GPL-3.0 License](https://github.com/1038lab/ComfyUI-FlashVSR/blob/main/LICENSE)
