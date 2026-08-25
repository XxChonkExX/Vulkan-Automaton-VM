# Self-Hosted GPU Runners Setup

Turn the two hardware boxes into permanent CI gates: every push to `main`
runs the real GPU suites (pool smoke, llama-server boot + completion,
throughput floor, autograd numerics).

```
push to main ──► ci.yml        (cloud: compile matrix + CPU tests)
             └──► gpu-ci.yml   (self-hosted: REAL GPU gates)
                    ├── windows-gpu  [this box: XTX + B70]
                    └── linux-gpu    [X2: Strix Halo / ROCm]
```

## SECURITY (read once)

- `gpu-ci.yml` triggers on **push to main + manual dispatch ONLY** — never
  `pull_request`. A fork PR would otherwise execute arbitrary code on your
  machine. Keep that trigger list unchanged.
- Runners must be registered against this repo (not as org-wide runners)
  so the machine is bound to a repo you control.
- The runner runs as YOUR user (it needs GPU access). That is expected.

## This box (Windows, dual-GPU)

1. **Test the suite manually first** (no runner needed):
   ```powershell
   pwsh -NoProfile -ExecutionPolicy Bypass -File D:\VulkanVM\ci\gpu_windows.ps1
   ```
   All gates should PASS (the throughput floor is 8 t/s at 8K ctx — the
   VRAM-spill cliff from docs/VRAM_OVERFLOW_FINDINGS.md fails it at ~half).

2. **Register the runner** (GitHub → repo → Settings → Actions → Runners →
   New self-hosted runner; pick Windows x64, then run the displayed
   commands). When configuring:
   ```powershell
   ./config.cmd --url https://github.com/XxChonkExX/Vulkan-Automaton-VM `
       --labels gpu --runasservice
   ```
   The `gpu` label is what `gpu-ci.yml` targets. `--runasservice` keeps it
   alive across reboots (installs as a Windows service).

3. **Verify**: push anything to main → the `GPU gate (Windows)` job should
   pick up within seconds and pass in ~3-4 minutes.

## X2 (Linux, Strix Halo / ROCm)

1. **Test the kit manually** (after the LoRA run frees the GPU):
   ```bash
   cd ~/VulkanVM   # or wherever the repo lives
   git pull
   bash ci/gpu_linux.sh                    # incremental
   FRESH=1 bash ci/gpu_linux.sh            # fresh-clone mode (0.3 gate)
   CHONK_TEST_MODEL=/path/to/some.gguf bash ci/gpu_linux.sh   # + server smoke
   ```

2. **Register the runner**:
   ```bash
   ./config.sh --url https://github.com/XxChonkExX/Vulkan-Automaton-VM \
       --labels gpu --runasservice
   sudo ./svc.sh install && sudo ./svc.sh start
   ```

## What each gate catches

| Gate | Catches |
|---|---|
| CPU suites | allocator invariants, placement logic, slab fuzz |
| Device enumeration | driver updates breaking Vulkan visibility |
| Server boot + pool stats | pool regressions, device-filter breakage, Chonk pool activation |
| Throughput floor | the VRAM-spill cliff class (docs/VRAM_OVERFLOW_FINDINGS.md) — silent 2x decode loss |
| Autograd numerics (X2) | gradient-math regressions vs ATen (M4) |
| Fresh-clone build (X2, `FRESH=1`) | "works on my machine" — missing files, bad defaults |

## Notes

- Runners offline? `gpu-ci.yml` jobs queue until the machine is back
  (bounded by `timeout-minutes`). Cloud `ci.yml` is unaffected.
- The Windows suite reuses `D:\VulkanVM\build_infer` incrementally (fast);
  wipe it occasionally to prove clean builds, or run the X2 kit with
  `FRESH=1`.
- Keep `gpu-ci.yml`'s trigger list push-only. Seriously.
