"""ComfyUI-FlashVSR - Real-time Video Super-Resolution Node"""

print("\n[FlashVSR] Loading nodes...")

from .AILab_FlashVSR import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Check for optional SageAttention optimization
try:
    from sageattention import sageattn
    print("[FlashVSR] ✓ SageAttention detected (~20-30% speedup enabled)")
except ImportError:
    print("[FlashVSR] ○ SageAttention not installed (optional speedup available)")

print(f"[FlashVSR] ✓ Loaded {len(NODE_CLASS_MAPPINGS)} node(s)\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]