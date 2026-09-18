# VulkanVM for Dummies — The Relay Race Tour

*Or: what this project actually does, explained the way I wish someone had
explained it to me.*

---

## The relay race

Training or running a big model on a GPU is a **relay race**. The runners are
your GPU, your CPU, your second GPU, maybe a second computer. The baton is
your data: model weights, activations, the KV cache, optimizer states.

On a normal system, the race is rigged. Every runner has their own private
track, and the tracks don't connect. So every handoff looks like this:

```
GPU memory  →  copy to staging  →  system RAM  →  copy again  →  next runner
```

Each copy is a **penalty lap**. Your GPU — the fastest runner you own — stands
at the fence waiting while the baton gets photocopied, notarized, and carried
over by hand. Do that every layer, every token, every step, and your monster
GPU spends its life waiting at fences.

Multi-vendor makes it crueler. AMD hands you a baton shaped like a DMA-BUF.
Intel wants a Level Zero handle. NVIDIA wants CUDA. Android wants an
AHardwareBuffer. The batons don't fit each other's hands, so everybody
translates, and translating means copying.

**VulkanVM's whole job is to get rid of the penalty laps.**

---

## The Chonk Buffer: one track

The core of VulkanVM is the **Chonk Buffer** — one giant contiguous memory
arena on the GPU, carved up by a buddy allocator that never fragments. Not
"mostly never." *Never*: the allocator hands out exact-fit chunks and returns
the slack, so after thousands of allocations and frees the free space is still
one clean run of track.

Everything runs on this one track:

- model weights live here
- activations here
- the KV cache here
- optimizer states here
- the PyTorch allocator draws from here

No fences. No penalty laps. The training logs in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) show what that buys: a 27B model
with 131K context of KV cache, weights, optimizer and activations all living
in one pool, flat memory for hundreds of steps.

## External memory: the baton never changes hands

"But PyTorch speaks CUDA/HIP, not Vulkan. Don't you still have to copy?"

No — and this is the trick the whole project is built on. Vulkan can **export**
a piece of its memory as an OS handle (a DMA-BUF on Linux, a Win32 handle on
Windows, an AHardwareBuffer on Android). HIP — AMD's CUDA — can **import** that
same handle. The result: the same physical bytes are now *simultaneously* a
Vulkan buffer and a HIP allocation.

```
Chonk Buffer (Vulkan)
       │  export (a handle, not a copy)
       ▼
HIP import  ──►  PyTorch tensor  ──►  your training loop
```

The GPU vendor's own driver does the handshake. We copy **nothing**. The
PyTorch tensor you train on *is* the Chonk Buffer memory, wearing a HIP name
tag. That's why the docs keep saying things like "zero-copy" and
"single-copy" — and why [docs/LIFETIME_CONTRACT.md](docs/LIFETIME_CONTRACT.md)
is very strict about who owns the baton at every moment.

## The coach: who carries which baton

A modern model isn't one race — it's forty races at once with wildly different
batons: attention, dense weights, 48×512 experts, the KV cache, one giant
embedding table. Somebody has to decide which runner carries which baton, and
"give the GPU everything" is the wrong answer more often than you'd think.

The **auto-placement planner** is the coach. Before the race starts it reads
the model file, weighs every baton, checks every runner's hands (VRAM, RAM,
bandwidth), and hands out assignments. On the flagship workload it
rediscovered — from first principles, zero hand-tuning — the exact expert
split a human had previously tuned by trial and error. One flag:

```bash
--vvm-split 'ffn_.*_exps.=auto,per_layer_token_embd=CPU'
```

No plan is ever a guess about your machine; it's computed from your model and
your hardware, every load.

## Multi-GPU: same track, next lane

A second GPU gets its own Chonk Buffer pool — same surface, next lane. When
runner A needs runner B's baton, there are three ways to pass it, and the
system always tells you which one you got:

- **The handoff** — GPU A exports the memory, GPU B imports it directly. One
  DMA, zero copies, the way the vendors intended. Works whenever the drivers
  allow it: same-vendor always, cross-vendor on Linux via DMA-BUF.
- **The exchange zone** — Windows drivers refuse cross-vendor handoffs (AMD →
  Intel doesn't just fail, it *crashes the machine* — measured, not rumored).
  So both runners pass the baton through a marked stretch of shared ground:
  one allocation in system RAM that *both* GPUs map into their own address
  space. GPU A DMAs the baton into the zone, GPU B DMAs it out. Zero CPU
  copies, 5–6 GiB/s, byte-verified in both directions. The Vulkan extension
  is even literally named for this — `VK_EXT_external_memory_host`.
- **The courier** — when nothing else is possible, an official ferries the
  baton through RAM in honest chunks. Portable, works everywhere, 3.5 GiB/s,
  and never pretends to be something faster. (`vvm-info` prints the policy;
  `VVM_P2P_POLICY=host` forces it.)

## The supply van: races bigger than your pockets

Here's the party trick: serving a **90 GB model on a single 24 GB card**.
That race has more batons than any runner can hold, so the team recruits
equipment:

- **The crate by the track** — system RAM. The OS page cache holds the hot
  experts and serves them at ~17 GB/s. The surprise finding of the whole
  campaign: on this box, the *CPU* runner pulling batons from the crate beat
  both GPUs at expert math. Measured twice. The coach noticed.
- **The supply van** — NVMe. The storage stream (five layers: pack format →
  software cache → request queue → IoRing backend → tiering scheduler) keeps
  the crate stocked at 3.3–3.5 GB/s — 96% of what the disk's own benchmark
  says the wire can physically deliver.

So: dense layers and KV cache live in VRAM, all 24,576 experts stream from
the crate, and the race runs at 16–17 tokens/s — still holding ~12–13
tokens/s at the full 262K native context. Full story:
[docs/inference_benchmarks.md](docs/inference_benchmarks.md) and
[docs/STORAGE_STREAM_ARCHITECTURE.md](docs/STORAGE_STREAM_ARCHITECTURE.md).

## No collisions: the baton-return official

Every young memory system makes the same mistake: the CPU hands back a buffer
the microsecond *it* is done — while the GPU runner is still mid-stride on
that same memory. The next allocation grabs the identical bytes, two runners
sprint through one lane, and you get corruption that looks like bad luck.

Version 0.4 hired a **baton-return official**. Handing memory back now means
handing it over with a lap number attached — "this baton is dead after lap
42." The official files it in the retirement queue and returns it to the bin
only once the GPU certifies lap 42 complete. Nobody waits. Nobody collides.
The cross-GPU copy path uses the same official to become genuinely
asynchronous: pass the baton, keep running, it re-enters circulation only
when safe. The fine print is contract rule R11:
[docs/RETIREMENT.md](docs/RETIREMENT.md).

## The network: recruiting the next team

When one computer isn't enough, the transport layer recruits other machines
into the relay — same track, next building:

- **TCP** for the honest, works-everywhere path
- **RDMA** when the NIC can DMA straight out of your pool
- **UCX** when you have opinions about your fabric

The placement planner decides which shards live at which building — the same
way the coach assigns batons, one level up.

## The announcer booth

You can hear everything. `vvm-info` reads out every runner's hands and the
exchange-zone policy before the race starts. `GET /vvm/stats` streams every
pool's fill level live, mid-race. `VVM_LOG_LEVEL=warn` turns down the crowd
noise when you're trying to think.

## Compute: new moves, same track

The compute layer adds tensor operations, collectives, and layout conversion
as Vulkan compute shaders — so data that's already on the track gets processed
*on* the track instead of taking a penalty lap into some other math library.

## Integrations: your frameworks, unchanged

The whole point is that you don't change how you work:

- **PyTorch**: the pluggable allocator makes `torch.cuda` draw from the Chonk
  Buffer. Your training loop doesn't know. Your optimizer doesn't know. The
  memory system knows, and that's who needed to.
- **ONNX Runtime**: same story, execution provider flavor.
- **llama.cpp**: serves through the pool on a fork branch — parity verified
  against stock across two GPU vendors.
- **Android**: the same pool, imported as an AHardwareBuffer — verified on a
  Galaxy S24+.

---

## The honest part

This is experimental, homebrewed, and proud of it. Some things are verified
end-to-end on real hardware: AMD RDNA3, Strix Halo, Intel B70 (Vulkan *and*
Level Zero), NVIDIA GTX 1080 Ti (CUDA path, small-model duty), Adreno
Android — see [docs/HARDWARE_SUPPORT.md](docs/HARDWARE_SUPPORT.md). Some
things are designed but thinly tested (RDMA on real NICs, DStorage import,
GDeflate). The autograd ops currently run through ATen with Vulkan dispatch
landing; the numbers are validated against PyTorch in
[test_autograd_numerics.py](python/vulkanvm_torch/test_autograd_numerics.py).

What we claim, we test. What we haven't tested, we say.

---

## TL;DR

| You want | You use |
|---|---|
| One GPU, zero fragmentation, no penalty laps | Core: Chonk Buffer |
| PyTorch training with everything in GPU memory | Core + PyTorch integration |
| Two GPUs, different vendors, one model | Multi-GPU lanes + exchange zone |
| 90 GB model on a 24 GB card | The coach + the crate + the supply van |
| Two computers, one relay team | + Transport: TCP/RDMA/UCX |
| Tensor math without leaving the track | + Compute layer |
| All of it on your phone | Core: Android AHardwareBuffer |
| To hear the race | `vvm-info`, `GET /vvm/stats`, `VVM_LOG_LEVEL=warn` |
| To finish every lap without collisions | The baton-return official (R11) |

The race is the same. The track is finally connected.

*— the relay race analogy is Mike/ChonkE's; it cuts to the heart of what
Automaton/Chonk Buffer does.*
