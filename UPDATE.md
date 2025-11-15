# ComfyUI-FlashVSR Update Log

## V1.1.0 (2025/11/15)

### 🔧 Fixes
- Applied frame-duplication fix contributed by **chris87423** and implemented by **goofyrodent**.  
  - Removes padded frames at the end of the sequence  
  - Removes the duplicated first 2 frames produced by FlashVSR  
  - Fix applied to both Basic and Advanced nodes  
  - Resolves issue [#3](https://github.com/1038lab/ComfyUI-FlashVSR/issues/3)

### 🧩 Model Updates
- Updated FlashVSR 1.1 model:
  **[Wan2_1-T2V-1.1_3B_FlashVSR_fp32.safetensors](https://huggingface.co/1038lab/FlashVSR/blob/main/Wan2_1-T2V-1.1_3B_FlashVSR_fp32.safetensors)**

  This model improves T2V → VSR sharpness, detail preservation, and temporal stability.

