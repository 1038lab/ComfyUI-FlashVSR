# ComfyUI-FlashVSR
# Powerful ComfyUI custom node built on the FlashVSR model,
# facilitating real-time diffusion-based video super-resolution for streaming applications.
#
# Models License Notice:
# - FlashVSR: Apache-2.0 License (https://huggingface.co/JunhaoZhuang/FlashVSR)
# - FlashVSR: Apache-2.0 License (https://github.com/OpenImagingLab/FlashVSR)
# - FlashVSR: Apache-2.0 License (https://huggingface.co/1038lab/FlashVSR)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-FlashVSR

import os
import math
import torch
import threading
import folder_paths
import comfy.utils
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from .FlashVSR import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from .FlashVSR.models.TCDecoder import build_tcdecoder
from .FlashVSR.models.utils import clean_vram, Buffer_LQ4x_Proj

def get_device_list():
    devs = ["auto"]
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except:
        pass
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except:
        pass
    return devs

DEVICE_CHOICES = get_device_list()

def log(msg, level='info'):
    colors = {'error': '\033[1;41m', 'warning': '\033[1;31m', 'success': '\033[1;32m', 'info': '\033[1;33m'}
    c = colors.get(level, '')
    print(f"{c}[FlashVSR] {msg}\033[m" if c else f"[FlashVSR] {msg}")

_flashvsr_models_checked = False
_flashvsr_check_lock = threading.Lock()
FLASHVSR_MODEL_DIR = os.path.join(folder_paths.models_dir, "FlashVSR")

def check_and_download_models():
    global _flashvsr_models_checked
    
    with _flashvsr_check_lock:

        if _flashvsr_models_checked:
            return FLASHVSR_MODEL_DIR
        
        log("First run detected, checking for models...")
        model_dir = FLASHVSR_MODEL_DIR

        required_files = [
            "Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors", 
            "Wan2.1_VAE.safetensors", 
            "Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors", 
            "Wan2_1_FlashVSR_TCDecoder_fp32.safetensors", 
            "Prompt.safetensors"
        ]

        if not os.path.exists(model_dir):
            log(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir)

        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            log(f"Missing models detected: {', '.join(missing_files)}. Downloading from HuggingFace...")
            try:
                snapshot_download(repo_id="1038lab/FlashVSR", local_dir=model_dir, 
                                local_dir_use_symlinks=False, resume_download=True)
                log("Download complete.", 'success')
            except Exception as e:
                log(f"Failed to download models: {e}", 'error')
                raise e
        else:
            log("All required models are present in cache.", 'success')

        final_missing = []
        for file_name in required_files:
            if not os.path.exists(os.path.join(model_dir, file_name)):
                final_missing.append(file_name)
        
        if final_missing:
            error_msg = f"Failed to find required models after download: {', '.join(final_missing)}. Node will not load."
            log(error_msg, 'error')
            raise FileNotFoundError(error_msg)
        
        _flashvsr_models_checked = True
        log("Model check complete.", 'success')
        return model_dir


def compute_dims(w, h, scale, align=128):
    sw, sh = w * scale, h * scale
    tw = math.ceil(sw / align) * align
    th = math.ceil(sh / align) * align
    return sw, sh, tw, th


def align_frames(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def prepare_video(frames, device, scale, dtype):
    N, H, W, C = frames.shape
    sw, sh, tw, th = compute_dims(W, H, scale)
    
    num_padded = N + 4
    aligned = align_frames(num_padded)
    if aligned == 0:
        raise ValueError(f"Need at least 21 frames, got {N}")
    
    processed = []
    for i in range(aligned):
        if i < 2:
            idx = 0
        elif i > N + 1:
            idx = N - 1
        else:
            idx = i - 2
            
        frame = frames[idx].permute(2, 0, 1).unsqueeze(0)    

        upscaled = F.interpolate(frame, size=(sh, sw), mode='bicubic', align_corners=False)
        
        pad_h, pad_w = th - sh, tw - sw
        if pad_h > 0 or pad_w > 0:
            upscaled = F.pad(upscaled, (0, pad_w, 0, pad_h), mode='replicate')
        
        normalized = upscaled * 2.0 - 1.0
        processed.append(normalized.squeeze(0).cpu().to(dtype))
    
    video = torch.stack(processed, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return video, th, tw, aligned, sh, sw


def to_frames(video):
    v = video.squeeze(0)
    v = rearrange(v, "C F H W -> F H W C")
    return (v.float() + 1.0) / 2.0


def calc_tiles(h, w, size, overlap):
    tiles = []
    stride = size - overlap
    rows = math.ceil((h - overlap) / stride)
    cols = math.ceil((w - overlap) / stride)
    
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + size, h), min(x1 + size, w)
            
            if y2 - y1 < size:
                y1 = max(0, y2 - size)
            if x2 - x1 < size:
                x1 = max(0, x2 - size)
            
            tiles.append((x1, y1, x2, y2))
    return tiles


def make_mask(h, w, overlap):
    mask = torch.ones(1, 1, h, w)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
    mask[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)
    mask[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
    mask[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
    return mask 

def init_pipe(mode, device, dtype):
    model_dir = check_and_download_models()
    
    ckpt = os.path.join(model_dir, "Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors")
    vae = os.path.join(model_dir, "Wan2.1_VAE.safetensors")
    lq = os.path.join(model_dir, "Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors")
    tcd = os.path.join(model_dir, "Wan2_1_FlashVSR_TCDecoder_fp32.safetensors")
    prompt = os.path.join(model_dir, "Prompt.safetensors")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    
    if mode == "full":
        mm.load_models([ckpt, vae])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        
        pipe.TCDecoder = build_tcdecoder([512, 256, 128, 128], device, dtype, 16 + 768)
        pipe.TCDecoder.load_state_dict(load_file(tcd, device=device), strict=False)
        pipe.TCDecoder.clean_mem()
    
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(3, 1536, 1).to(device, dtype)
    if os.path.exists(lq):
        lq_state_dict = load_file(lq, device="cpu")
        
        cleaned_state_dict = {}
        prefix_to_remove = "LQ_proj_in."
        for k, v in lq_state_dict.items():
            if k.startswith(prefix_to_remove):
                cleaned_state_dict[k[len(prefix_to_remove):]] = v
            else:
                cleaned_state_dict[k] = v
        
        pipe.denoising_model().LQ_proj_in.load_state_dict(cleaned_state_dict, strict=True)
    
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt)
    pipe.load_models_to_device(["dit", "vae"])
    
    return pipe


def _setup_device_and_dtype(device_str, dtype_str):

    dev = device_str
    if device_str == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else None
        if not dev:
            log("No 'auto' device (CUDA or MPS) available, falling back to CPU. This might be very slow.", 'warning')
            dev = "cpu"
    
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
    
    dt = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    return dev, dt


def _adjust_frame_count(result, expected_frames):

    if result.shape[0] != expected_frames:
        log(f"Warning: Output frames ({result.shape[0]}) != input frames ({expected_frames}). Truncating/padding output.", 'warning')
        if result.shape[0] > expected_frames:
            result = result[:expected_frames]
        else:
            padding = torch.zeros(expected_frames - result.shape[0], *result.shape[1:], dtype=result.dtype)
            if result.shape[0] > 0:
                padding[...] = result[-1]
            result = torch.cat((result, padding), dim=0)
    return result


class PBar:
    def __init__(self, iterable, total=None):
        self.iterable = iterable
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = 0
        self.total = total
        self.pbar = comfy.utils.ProgressBar(self.total)
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.total:
            self.current += 1
            self.pbar.update(1)
            if isinstance(self.iterable, range):
                return self.iterable[self.current - 1]
            try:
                return self.iterable[self.current - 1]
            except TypeError:
                return next(self.iterable) 
        else:
            raise StopIteration

    def update(self, n=1):
        self.current += n
        self.pbar.update(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _full(frames, pipe, scale, sr, kvr, lr, cf, ud, tv, seed, dev, dt):
    vid, th, tw, nf, sh, sw = prepare_video(frames, dev, scale, dt)
    
    if "long" not in pipe.__class__.__name__.lower():
        vid = vid.to(dev)
    
    out = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, 
                seed=seed, tiled=tv, progress_bar_cmd=lambda t: PBar(t), LQ_video=vid,
                num_frames=nf, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sr * 768 * 1280 / (th * tw), kv_ratio=kvr, local_range=lr,
                color_fix=cf, unload_dit=ud)
    
    res = to_frames(out).cpu()[:frames.shape[0], :sh, :sw, :]
    return res
    
def _tile(frames, pipe, scale, ts, to, sr, kvr, lr, cf, ud, tv, seed, dev, dt):
    N, H, W, C = frames.shape
    nf = align_frames(N + 4) - 4
    oh, ow = H * scale, W * scale
    
    canvas = torch.zeros((nf, oh, ow, C), dtype=torch.float32)
    weights = torch.zeros_like(canvas)
    tiles = calc_tiles(H, W, ts, to)
    
    with PBar(tiles) as pb:
        for i, (x1, y1, x2, y2) in enumerate(pb):
            log(f"Processing tile {i+1}/{len(tiles)}")
            
            tf = frames[:, y1:y2, x1:x2, :]
            tv_tensor, th, tw, tnf, tsh, tsw = prepare_video(tf, dev, scale, dt)
            
            if "long" not in pipe.__class__.__name__.lower():
                tv_tensor = tv_tensor.to(dev)
            
            tout = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
                        seed=seed, tiled=tv, LQ_video=tv_tensor, num_frames=tnf, height=th, width=tw,
                        is_full_block=False, if_buffer=True, topk_ratio=sr * 768 * 1280 / (th * tw),
                        kv_ratio=kvr, local_range=lr, color_fix=cf, unload_dit=ud)
            
            tres = to_frames(tout).cpu()[:nf, :tsh, :tsw, :]
            
            ah, aw = tres.shape[1], tres.shape[2]
            mask = make_mask(ah, aw, min(to * scale, ah // 4, aw // 4)).permute(0, 2, 3, 1)
            
            oy1, ox1 = y1 * scale, x1 * scale
            oy2, ox2 = min(oy1 + ah, oh), min(ox1 + aw, ow)
            ath, atw = oy2 - oy1, ox2 - ox1
            
            tres = tres[:, :ath, :atw, :]
            mask = mask[:, :ath, :atw, :]
            
            canvas[:, oy1:oy2, ox1:ox2, :] += tres * mask
            weights[:, oy1:oy2, ox1:ox2, :] += mask
            
            del tv_tensor, tout, tres
            # Clean VRAM every 4 tiles instead of every tile for better performance
            if (i + 1) % 4 == 0 or i == len(tiles) - 1:
                clean_vram()
    
    weights[weights == 0] = 1.0
    return canvas / weights


class AILab_FlashVSR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The low-resolution video frames to be upscaled."}),
                "preset": (["Fast (2x Speed)", "Balanced (2x Quality)", "Long Video (Low VRAM)", "High Quality (Best)"], {"default": "Balanced (2x Quality)", "tooltip": "Select a balance between speed and quality. 'Long Video (Low VRAM)' is best for long videos as it uses less VRAM. 'High Quality' uses the 'Full' model."}),
                "scale": ("INT", {"default": 2, "min": 2, "max": 4, "step": 2, "tooltip": "The upscaling factor. 4x is recommended for best quality."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload the main model from VRAM before decoding frames with the VAE. Saves VRAM but is slightly slower."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The random seed for the diffusion process."}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("FRAMES", "AUDIO",)
    FUNCTION = "upscale"
    CATEGORY = "🧪AILab/⚡FlashVSR"
    
    def upscale(self, frames, preset, scale, unload_model, seed, audio=None):
        # Use default values for removed UI parameters
        color_fix = True
        device = "auto"
        dtype = "bf16"
        
        frame_count = frames.shape[0]
                
        if frame_count < 21:
            raise ValueError(f"FlashVSR needs at least 21 frames to work, but got {frames.shape[0]}")

        # Duplicate the last frame 2x because VSR removes the last 2 frames
        last_frame = frames[-1:].repeat(2, *[1 for _ in range(frames.ndim - 1)])
        frames = torch.cat([frames, last_frame], dim=0)
        log(f"Duplicated last frame 2x {frame_count} >> {frames.shape[0]}", "info")
        frame_count = frames.shape[0]

        # Add padded frames
        remainder = (frame_count - 5) % 8
        added_frames = 0
        if remainder != 0:
            added_frames = 8 - remainder
            last_frame = frames[-1:]
            
            last_frames = last_frame.repeat(added_frames, *[1 for _ in range(last_frame.ndim - 1)])
            frames = torch.cat([frames, last_frames], dim=0)

            log(f"Added {added_frames} padded frame(s) to the end. {frame_count} >> {frames.shape[0]}", 'info')

        presets = {
            "Fast (2x Speed)": ("tiny", 1.5, 1.0, 9, True, True, 256, 32),
            "Balanced (2x Quality)": ("tiny", 2.0, 2.0, 11, True, True, 256, 32),
            "Long Video (Low VRAM)": ("tiny-long", 2.0, 2.0, 11, True, True, 256, 32),
            "High Quality (Best)": ("full", 2.0, 3.0, 11, True, True, 256, 32),
        }
        
        mode, sr, kvr, lr, td, tv, ts, to = presets[preset]
        ud = unload_model
        
        if scale != 4:
            log("Note: FlashVSR is optimized for 4x upscaling", 'warning')
        
        # Use helper function for device setup
        dev, dt = _setup_device_and_dtype(device, dtype)
        
        log(f"Starting: {frames.shape[0]} frames, {scale}x, {preset}")
        
        pipe = init_pipe(mode, dev, dt)
        
        if td:
            result = _tile(frames, pipe, scale, ts, to, sr, kvr, lr, color_fix, ud, tv, seed, dev, dt)
        else:
            result = _full(frames, pipe, scale, sr, kvr, lr, color_fix, ud, tv, seed, dev, dt)
        
        del pipe
        clean_vram()
        log("Done", 'success')

        # Remove padded frames
        if added_frames > 0:
            frame_count = result.shape[0]
            result = result[:-added_frames]
            log(f"Removed {added_frames} padded frame(s) from the end. {frame_count} >> {result.shape[0]}", 'info')
            frame_count = result.shape[0]

        # Remove the fist 2 frames because VSR duplicates the first frame 2x
        result = result[2:]
        log(f"Removed the first 2 frames {frame_count} >> {result.shape[0]}", 'info')

        # Optional remove 1px black border
        result = result[:, 1:-1, 1:-1, :]

        return (result, audio,)
    

class AILab_FlashVSR_Advanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "model_version": (["Tiny (Fast)", "Tiny Long (Low VRAM)", "Full (Best Quality)"], {"default": "Tiny (Fast)", "tooltip": "Tiny: Fast, good quality. Tiny Long: Slower, uses much less VRAM for long videos. Full: Best quality, highest VRAM."}),
                "scale": ("INT", {"default": 2, "min": 2, "max": 4, "step": 2, "tooltip": "The upscaling factor. 4x is recommended for best quality."}),
                "enable_tiling": ("BOOLEAN", {"default": True, "tooltip": "Process the video in tiles. Slower, but saves a large amount of VRAM."}),
                "tile_size": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 32, "tooltip": "The height/width of each tile (before upscaling). Larger tiles are faster but use more VRAM. Recommended: 384-512 for HD videos."}),
                "tile_overlap": ("INT", {"default": 24, "min": 8, "max": 256, "step": 8, "tooltip": "The amount of overlap between tiles to prevent visible seams. Recommended: 48-64 for smooth blending. (Must be < Tile Size / 2)"}),
                "speed_optimization": ("FLOAT", {"default": 2.0, "min": 1.5, "max": 2.0, "step": 0.1, "tooltip": "Higher value (e.g., 2.0) is faster but may capture less fine detail. Lower (e.g., 1.5) is slower but more detailed."}),
                "quality_boost": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Increases quality and detail preservation at the cost of more VRAM. (Higher = Better Quality)"}),
                "stability_level": ([9, 11], {"default": 11, "tooltip": "Controls temporal stability. 11 is more stable (default). 9 can be sharper but may have less temporal consistency."}),
                "color_fix": ("BOOLEAN", {"default": True, "tooltip": "Applies a color correction step to prevent potential color shifts from the original video."}),
                "vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Enables tiling for the VAE decoding step. Further reduces VRAM usage, especially for high-resolution output."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload the main model from VRAM before decoding frames with the VAE. Saves VRAM but is slightly slower."}),
                "sageattention": (["enable", "disable"], {"default": "enable", "tooltip": "Enable SageAttention optimization for faster inference (~20-30% speedup). Disable if encountering compatibility issues."}),
                "device": (DEVICE_CHOICES, {"default": DEVICE_CHOICES[0], "tooltip": "Select the processing device. 'auto' will try to pick the best one available (CUDA > MPS > CPU)."}),
                "precision": (["bf16", "fp16"], {"default": "bf16", "tooltip": "Processing precision. BF16 is often faster on modern GPUs. FP16 is a good alternative."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The random seed for the diffusion process."}), 
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("frames", "audio",)
    FUNCTION = "upscale"
    CATEGORY = "🧪AILab/⚡FlashVSR"
    
    def upscale(self, frames, model_version, scale, enable_tiling, tile_size, tile_overlap,
                speed_optimization, quality_boost, stability_level, color_fix, vae_tiling,
                unload_model, sageattention, device, precision, seed, audio=None):
        
        # Convert sageattention string to boolean for internal use
        use_sage = (sageattention == "enable")
        dtype = precision  # Map precision parameter to dtype variable
        
        frame_count = frames.shape[0]
                
        if frame_count < 21:
            raise ValueError(f"FlashVSR needs at least 21 frames to work, but got {frames.shape[0]}")

        if enable_tiling and tile_overlap >= tile_size / 2:
            raise ValueError("tile_overlap must be less than half of tile_size")

        # Duplicate the last frame 2x because VSR removes the last 2 frames
        last_frame = frames[-1:].repeat(2, *[1 for _ in range(frames.ndim - 1)])
        frames = torch.cat([frames, last_frame], dim=0)
        log(f"Duplicated last frame 2x {frame_count} >> {frames.shape[0]}", "info")
        frame_count = frames.shape[0]

        # Add padded frames
        remainder = (frame_count - 5) % 8
        added_frames = 0
        if remainder != 0:
            added_frames = 8 - remainder
            last_frame = frames[-1:]
            
            last_frames = last_frame.repeat(added_frames, *[1 for _ in range(last_frame.ndim - 1)])
            frames = torch.cat([frames, last_frames], dim=0)

            log(f"Added {added_frames} padded frame(s) to the end. {frame_count} >> {frames.shape[0]}", 'info')
        
        mode_map = {
            "Tiny (Fast)": "tiny",
            "Tiny Long (Low VRAM)": "tiny-long",
            "Full (Best Quality)": "full"
        }
        mode = mode_map[model_version]
        
        if scale != 4:
            log("Note: FlashVSR is optimized for 4x upscaling", 'warning')
        
        # Use helper function for device setup
        dev, dt = _setup_device_and_dtype(device, dtype)
        
        log(f"Starting: {frames.shape[0]} frames, {scale}x, {model_version}")
        
        pipe = init_pipe(mode, dev, dt)
        
        if enable_tiling:
            result = _tile(frames, pipe, scale, tile_size, tile_overlap,
                           speed_optimization, quality_boost, stability_level,
                           color_fix, unload_model, vae_tiling, seed, dev, dt)
        else:
            result = _full(frames, pipe, scale, speed_optimization,
                           quality_boost, stability_level, color_fix,
                           unload_model, vae_tiling, seed, dev, dt)
        
        del pipe
        clean_vram()
        log("Done", 'success')

        # Remove padded frames
        if added_frames > 0:
            frame_count = result.shape[0]
            result = result[:-added_frames]
            log(f"Removed {added_frames} padded frame(s) from the end. {frame_count} >> {result.shape[0]}", 'info')
            frame_count = result.shape[0]

        # Remove the fist 2 frames because VSR duplicates the first frame 2x
        result = result[2:]
        log(f"Removed the first 2 frames {frame_count} >> {result.shape[0]}", 'info')

        # Optional remove 1px black border
        result = result[:, 1:-1, 1:-1, :]

        return (result, audio,)


NODE_CLASS_MAPPINGS = {
    "AILab_FlashVSR": AILab_FlashVSR,
    "AILab_FlashVSR_Advanced": AILab_FlashVSR_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_FlashVSR": "FlashVSR ⚡",
    "AILab_FlashVSR_Advanced": "FlashVSR ⚡ Advanced",
}

