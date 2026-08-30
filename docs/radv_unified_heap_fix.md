# radv Unified Heap Fix (Strix Halo / APU)

## Problem

Mesa radv on Strix Halo (gfx1151, UMA) reports a **split heap** by default:

- `visible_vram = 2/3 * total` → **82 GB DEVICE_LOCAL** heap
- `gtt = 1/3 * total` → **41 GB** host-visible heap

The Chonk pool selects the 82 GB DEVICE_LOCAL heap, which caps the pool at
~82 GB. A 30B model (INT4 base 15.5 GB + bf16 KV 32 GB + LoRA r=128 ~15 GB +
AdamW + activations) at full 131K context exceeds this during the first
forward+backward → `VK_ERROR_OUT_OF_DEVICE_MEMORY` → HSA memory fault →
hipErrorLaunchFailure → hard freeze/reboot. Physical RAM (129 GB) is never
the limit; radv's DEVICE_LOCAL heap accounting is.

## Fix

Mesa has a driconf option `radv_enable_unified_heap_on_apu` (default `false`).
When `true`, radv exposes ONE unified DEVICE_LOCAL heap = full GTT (~130 GB)
instead of the 82 GB + 41 GB split. It is a **runtime driconf option** — no
Mesa rebuild required (verified present in `libvulkan_radeon.so` 26.0.8, and
in `radv_physical_device.c` `radv_physical_device_init_mem_types`).

Enable it for the trainer (python process) via a user driconf file:

`~/.drirc`:

```xml
<driconf>
  <device>
    <application name="python" executable="python3.12">
      <option name="radv_enable_unified_heap_on_apu" value="true" />
    </application>
    <application name="python3" executable="python3">
      <option name="radv_enable_unified_heap_on_apu" value="true" />
    </application>
    <application name="python" executable="python">
      <option name="radv_enable_unified_heap_on_apu" value="true" />
    </application>
  </device>
</driconf>
```

## Verification

After the fix, the training log should report:

- Before: `Selected DEVICE_LOCAL memory type 3 (heap budget: 83727 MB)`
- After:  `... heap budget: ~126000 MB`

(`vulkaninfo` won't show the change because it runs as `vulkaninfo`, not
`python`; the fix is matched by executable basename `python3.12`.)

## Non-Mesa alternative

AMDVLK (AMD's proprietary Vulkan driver) also exposes a different heap layout
and can be selected via `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/amdvlk.json`.
This requires adding AMD's repo (not published for Ubuntu 26.04 yet) and is
riskier alongside the ROCm 7.1 install, so the radv unified-heap driconf is
preferred.

## Related commits

- (this doc)
- ChunkedPoolBuffer (split 14.43 GB q buffer into <=1 GB blocks)
- per-layer KV cache
- best-fit slab + aggressive empty-block release
- full graduated bucket ladder (1 MB .. 512 GB)