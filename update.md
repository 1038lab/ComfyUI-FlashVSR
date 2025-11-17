# ComfyUI-FlashVSR Update Log

## V1.1.1 (2025/11/17)
- Refresh FlashVSR 1.1 main model package (renamed to `FlashVSR1_1.safetensors` alongside `LQ_proj_in.safetensors` and `TCDecoder.safetensors`) and update latest HuggingFace models.
- Fix the core loader to validate the new filenames and raise clear warnings when any required asset is missing, preventing silent fallbacks.
- Fix runtime RoPE dtype patch so FlashVSR runs on Apple Silicon/MPS without float64 errors.

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


