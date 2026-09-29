# XPU (Intel Arc Pro B70, Windows) training notes: failure modes + mitigations

Field report from bf16 LoRA fine-tuning (Gemma4-12B, r32, 4K blocks) on torch-XPU,
for anyone taking the Chonk/torch path to Intel discrete GPUs. All below is
measured, not theorized. Happy to provide logs.

## Environment

- GPU: Intel Arc Pro B70 32 GB (Battlemage G31, 0xE223), driver 32.0.101.8974
- CPU: Ryzen 9 7900X, 64 GB DDR5, Windows 11
- torch 2.11.0+xpu, transformers 5.18.0.dev0 (gemma4_unified), PEFT 0.19.1
- Python 3.14, MSVC-built extensions
- Sibling GPU on box: RX 7900 XTX (torch-blind on Windows; used as Chonk pool host)

## 1. Fused foreach optimizer kernels wedge the engine (UR_DEVICE_LOST)

`torch.optim.AdamW(..., foreach=True)` and `clip_grad_norm_(..., foreach=True)`:
first optimizer phase passes, second hangs the compute engine indefinitely —
no TDR, no driver event, just a wedged queue. Surfaces later as
`UR_RESULT_ERROR_DEVICE_LOST` at whatever op syncs next (we saw it in an
attention-mask comparison). Per-tensor path (`foreach=False`) ran 6000+ steps
clean. If you touch XPU training, default foreach off until the fused path
matures.

## 2. Explicit `torch.xpu.synchronize()` degrades per call

Measured 23.6s → 37.0s and growing across consecutive calls, in a loop doing
nothing else different. Implicit syncs (`.item()`, D2H copies) are sufficient
and stable. Do not add a per-step explicit synchronize on this stack.

## 3. The 4 GB single-allocation cap bites the caching allocator

Intel documents the 4 GB stateful-addressing cap (compute-runtime
ALLOCATIONS_GREATER_THAN_4GB.md). torch's caching allocator requests large
blocks; on a 27.8/32 GB footprint, block requests that cross the cap fail as
UR 39/40 even with `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. What worked:
layer-granular `.to('xpu')` moves (~0.5–2.5 GB each, all under the cap) instead
of one bulk `model.to()` (which hangs), plus `max_split_size_mb`-style
discipline on block growth. Small allocations (4 KB mask tensors) failing
while gigabytes report free is the signature symptom — it means fence/block
bookkeeping, not capacity.

## 4. Kill -9 accumulates driver-side debris; clean exits don't (as much)

Every `Stop-Process -Force` on a mid-kernel trainer measurably worsens the
next run (earlier deaths, smaller failing allocs). A GPU disable/enable cycle
(`Disable-PnpDevice`/`Enable-PnpDevice` on the B70) reliably restores health;
verified with 8 GB + 28 GB probe allocations post-reset. Operational rule we
adopted: STOP-file clean exits only, kills as last resort, reset on repeated
UR-40s. Watchdog design note: a native event-poll loop can hold the GIL while
wedged, starving an in-process watchdog thread — enforce liveness from a
separate supervisor process (we watch heartbeat.txt freshness, 13 min limit).

## 5. Headless-launch + Intel MKL: forrtl error 200

`forrtl: error (200): program aborting due to window-CLOSE event` kills
headless-launched trainers (WMI/scheduler, no console). Fix:
`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` in the launcher environment (inherited).

## 6. Env knobs that helped (all inherited process-wide)

- `SYCL_UR_USE_LEVEL_ZERO_V2=1`
- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
- `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`
- B70 drivers: 8974 verified for this workload (8861 was 24x broken for
  compute per repo docs; 8626 broke training per Intel forums — we stayed put)

## 7. Thermal ruled out

Sustained 73 C max, fans normal, on a training run that still degraded —
this family of failures is driver/queue state, not heat. (Pace still degrades
~36 s/step fresh to ~400 s/step over ~1 h of uptime; mitigation: clean-exit
recycle every 50 steps + resume, ~7% relaunch tax, fully automated.)

## Offer

Full step logs with per-phase timings (data/fetch/H2D/fw/bw/clip/adam),
allocator reserved-vs-allocated traces across the failure boundary, and the
supervisor/heartbeat harness are available on request. If any of this belongs
in HARDWARE_SUPPORT.md or a Windows-training doc, say which section and I'll
PR it in repo style.
