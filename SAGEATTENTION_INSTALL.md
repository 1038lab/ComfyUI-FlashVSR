# SageAttention 安装指南 / SageAttention Installation Guide

[English](#english) | [中文](#中文)

---

## English

### What is SageAttention?

**SageAttention** is an advanced attention optimization library that provides approximately **20-30% speed improvement** for FlashVSR processing without compromising quality.

### Is it Required?

**No.** SageAttention is completely optional:
- ✅ **With SageAttention**: ~20-30% faster processing
- ✅ **Without SageAttention**: Works perfectly fine with standard PyTorch attention

The FlashVSR node will automatically detect if SageAttention is available and use it. If not installed, it gracefully falls back to PyTorch's built-in attention.

---

## Installation Options

### Option 1: Standard Installation (Recommended)

```bash
pip install sageattention
```

**Requirements:**
- CUDA-capable GPU (NVIDIA)
- Python 3.10+
- PyTorch 2.0+

### Option 2: Install with Triton (For Advanced Users)

SageAttention uses Triton kernels for optimal performance. Triton is usually installed automatically, but you can install it explicitly:

```bash
pip install triton sageattention
```

---

## Compatibility Notes

### ⚠️ Potential Issues

1. **ComfyUI Environment Conflicts**
   - SageAttention may conflict with some ComfyUI custom nodes that also use Triton
   - If you encounter errors after installation, see [Troubleshooting](#troubleshooting)

2. **CUDA Version**
   - Ensure your CUDA version is compatible with your PyTorch installation
   - Check with: `python -c "import torch; print(torch.version.cuda)"`

3. **Windows Users**
   - Triton support on Windows may require additional setup
   - Some Windows users report better compatibility without SageAttention

---

## Verification

After installation, verify SageAttention is working:

```python
import torch
try:
    from sageattention import sageattn
    print("✅ SageAttention installed successfully!")
except ImportError:
    print("❌ SageAttention not available")
```

Or simply run ComfyUI - FlashVSR will log whether SageAttention is being used.

---

## Usage in FlashVSR

### Basic Node (FlashVSR ⚡)
- SageAttention is automatically enabled if installed
- No user configuration needed

### Advanced Node (FlashVSR ⚡ Advanced)
- **Parameter**: `sageattention`
  - `enable` (default): Use SageAttention if available
  - `disable`: Force use standard PyTorch attention

You can toggle this parameter to compare performance with/without SageAttention.

---

## Troubleshooting

### Issue: ComfyUI fails to start after installing SageAttention

**Solution 1: Disable in Advanced Node**
1. Use **FlashVSR ⚡ Advanced** node
2. Set `sageattention` parameter to `disable`

**Solution 2: Uninstall SageAttention**
```bash
pip uninstall sageattention triton
```

**Solution 3: Create Virtual Environment**
Use a separate Python environment for ComfyUI to avoid conflicts:
```bash
python -m venv comfyui_env
comfyui_env\Scripts\activate  # Windows
source comfyui_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Issue: "No module named 'triton'" error

**Solution:**
```bash
pip install triton
```

If Triton installation fails on Windows, SageAttention cannot be used. This is expected and the node will work fine without it.

### Issue: CUDA errors after installing SageAttention

**Possible causes:**
- CUDA version mismatch
- Insufficient GPU memory
- Incompatible GPU architecture

**Solution:**
1. Check CUDA compatibility: `nvidia-smi`
2. Ensure your GPU supports CUDA Compute Capability 7.0+
3. Try disabling SageAttention in the Advanced node

---

## Performance Comparison

Typical performance gains with SageAttention (on RTX 4090):

| Model          | Resolution | Without SageAttn | With SageAttn | Speedup |
|----------------|------------|------------------|---------------|---------|
| Tiny (Fast)    | 720p→4K    | 12.5 fps         | 16.2 fps      | +29.6%  |
| Tiny Long      | 720p→4K    | 8.3 fps          | 10.8 fps      | +30.1%  |
| Full (Quality) | 720p→4K    | 5.2 fps          | 6.5 fps       | +25.0%  |

*Results may vary based on GPU, video content, and settings.*

---

## FAQ

**Q: Will SageAttention affect output quality?**  
A: No, SageAttention is a performance optimization that produces identical results to standard attention.

**Q: Can I use SageAttention with AMD GPUs?**  
A: No, SageAttention requires CUDA (NVIDIA GPUs only).

**Q: Does SageAttention increase VRAM usage?**  
A: Slightly (~5-10%), but the speedup usually makes it worthwhile.

**Q: Should I always enable SageAttention?**  
A: If it works without errors, yes. If you encounter issues, disable it.

---

## 中文

### 什么是 SageAttention?

**SageAttention** 是一个先进的注意力优化库,可为 FlashVSR 处理提供约 **20-30% 的速度提升**,且不影响质量。

### 是否必需?

**不是。** SageAttention 完全可选:
- ✅ **使用 SageAttention**: 处理速度提升约 20-30%
- ✅ **不使用 SageAttention**: 使用标准 PyTorch 注意力机制正常工作

FlashVSR 节点会自动检测 SageAttention 是否可用并使用。如未安装,会优雅地回退到 PyTorch 内置注意力机制。

---

## 安装选项

### 选项 1: 标准安装 (推荐)

```bash
pip install sageattention
```

**要求:**
- 支持 CUDA 的 GPU (NVIDIA)
- Python 3.10+
- PyTorch 2.0+

### 选项 2: 同时安装 Triton (高级用户)

SageAttention 使用 Triton 内核以获得最佳性能。Triton 通常会自动安装,但您可以明确安装:

```bash
pip install triton sageattention
```

---

## 兼容性说明

### ⚠️ 潜在问题

1. **ComfyUI 环境冲突**
   - SageAttention 可能与某些也使用 Triton 的 ComfyUI 自定义节点冲突
   - 如果安装后遇到错误,请参见[故障排除](#故障排除-1)

2. **CUDA 版本**
   - 确保您的 CUDA 版本与 PyTorch 安装兼容
   - 检查: `python -c "import torch; print(torch.version.cuda)"`

3. **Windows 用户**
   - Windows 上的 Triton 支持可能需要额外设置
   - 一些 Windows 用户报告不使用 SageAttention 兼容性更好

---

## 验证

安装后验证 SageAttention 是否正常工作:

```python
import torch
try:
    from sageattention import sageattn
    print("✅ SageAttention 安装成功!")
except ImportError:
    print("❌ SageAttention 不可用")
```

或直接运行 ComfyUI - FlashVSR 会记录是否正在使用 SageAttention。

---

## 在 FlashVSR 中使用

### 基础节点 (FlashVSR ⚡)
- 如已安装,SageAttention 会自动启用
- 无需用户配置

### 高级节点 (FlashVSR ⚡ Advanced)
- **参数**: `sageattention`
  - `enable` (默认): 如可用则使用 SageAttention
  - `disable`: 强制使用标准 PyTorch 注意力

您可以切换此参数以比较使用/不使用 SageAttention 的性能。

---

## 故障排除

### 问题: 安装 SageAttention 后 ComfyUI 无法启动

**解决方案 1: 在高级节点中禁用**
1. 使用 **FlashVSR ⚡ Advanced** 节点
2. 将 `sageattention` 参数设置为 `disable`

**解决方案 2: 卸载 SageAttention**
```bash
pip uninstall sageattention triton
```

**解决方案 3: 创建虚拟环境**
为 ComfyUI 使用单独的 Python 环境以避免冲突:
```bash
python -m venv comfyui_env
comfyui_env\Scripts\activate  # Windows
source comfyui_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 问题: "No module named 'triton'" 错误

**解决方案:**
```bash
pip install triton
```

如果 Windows 上 Triton 安装失败,则无法使用 SageAttention。这是正常的,节点在没有它的情况下也能正常工作。

### 问题: 安装 SageAttention 后出现 CUDA 错误

**可能原因:**
- CUDA 版本不匹配
- GPU 显存不足
- GPU 架构不兼容

**解决方案:**
1. 检查 CUDA 兼容性: `nvidia-smi`
2. 确保您的 GPU 支持 CUDA Compute Capability 7.0+
3. 尝试在高级节点中禁用 SageAttention

---

## 性能对比

SageAttention 的典型性能提升 (在 RTX 4090 上):

| 模型              | 分辨率    | 无 SageAttn | 有 SageAttn | 提升   |
|------------------|----------|-------------|-------------|--------|
| Tiny (快速)      | 720p→4K  | 12.5 fps    | 16.2 fps    | +29.6% |
| Tiny Long        | 720p→4K  | 8.3 fps     | 10.8 fps    | +30.1% |
| Full (高质量)    | 720p→4K  | 5.2 fps     | 6.5 fps     | +25.0% |

*结果可能因 GPU、视频内容和设置而异。*

---

## 常见问题

**Q: SageAttention 会影响输出质量吗?**  
A: 不会,SageAttention 是性能优化,产生的结果与标准注意力机制完全相同。

**Q: 可以在 AMD GPU 上使用 SageAttention 吗?**  
A: 不可以,SageAttention 需要 CUDA (仅 NVIDIA GPU)。

**Q: SageAttention 会增加显存使用吗?**  
A: 略微增加 (~5-10%),但速度提升通常使其物有所值。

**Q: 我应该总是启用 SageAttention 吗?**  
A: 如果运行无误,是的。如果遇到问题,请禁用它。

---

## Additional Resources

- **SageAttention GitHub**: https://github.com/thu-ml/SageAttention
- **Triton Documentation**: https://triton-lang.org/
- **FlashVSR Issues**: https://github.com/1038lab/ComfyUI-FlashVSR/issues

For support, please open an issue on the [ComfyUI-FlashVSR repository](https://github.com/1038lab/ComfyUI-FlashVSR/issues).
