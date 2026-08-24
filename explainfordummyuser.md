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
at the fence waiting while the baton gets photocopyed, notarized, and carried
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

## Multi-GPU: same track, next lane

Got a second GPU from a *different vendor*? In most stacks, cross-vendor means
"copy through system RAM and pray." Here, each GPU gets its own Chonk Buffer
pool and the pools talk:

- **Direct path**: export a block from GPU A, import into GPU B — one DMA when
  the drivers allow it.
- **Staged path**: when they don't, the transport layer moves the bytes
  efficiently and tells you honestly that it did.

The llama.cpp integration runs one 40B model pooled across an AMD 7900 XTX and
an Intel Arc Pro B70 — different vendors, one model, verified at parity with
stock ([docs/inference_benchmarks.md](docs/inference_benchmarks.md)).

## The network: recruiting the next team

When one computer isn't enough, the transport layer recruits other machines
into the relay — same track, next building:

- **TCP** for the honest, works-everywhere path
- **RDMA** when the NIC can DMA straight out of your pool
- **UCX** when you have opinions about your fabric

The [ModelHub](README.md) and placement planner decide which shards live
where, by capacity and bandwidth — the same way the pool decides which block
holds which tensor, one level up.

## Compute: teaching the runners new moves

The compute layer adds tensor operations, collectives, and layout conversion
as Vulkan compute shaders — so data that's already on the track gets processed
*on* the track instead of taking a penalty lap into some other math library.

## Integrations: your frameworks, unchanged

The whole point is that you don't change how you work:

- **PyTorch**: the pluggable allocator makes `torch.cuda` draw from the Chonk
  Buffer. Your training loop doesn't know. Your optimizer doesn't know. The
  memory system knows, and that's who needed to.
- **ONNX Runtime**: same story, execution provider flavor.
- **Android**: the same pool, imported as an AHardwareBuffer — verified on a
  Galaxy S24+.

---

## The honest part

This is experimental, homebrewed, and proud of it. Some things are verified
end-to-end on real hardware (AMD RDNA3, Strix Halo, Intel B70, Adreno
Android — see [docs/HARDWARE_SUPPORT.md](docs/HARDWARE_SUPPORT.md)). Some
things are designed but never touched real silicon (NVIDIA, we're looking at
you — [help wanted](README.md)). The autograd ops currently run through ATen
with Vulkan dispatch landing; the numbers are validated against PyTorch in
[test_autograd_numerics.py](python/vulkanvm_torch/test_autograd_numerics.py).

What we claim, we test. What we haven't tested, we say.

---

## TL;DR

| You want | You use |
|---|---|
| One GPU, zero fragmentation, no penalty laps | Core: Chonk Buffer |
| PyTorch training with everything in GPU memory | Core + PyTorch integration |
| Two GPUs, different vendors, one model | Core: cross-GPU sharing |
| Two computers, one relay team | + Transport: TCP/RDMA/UCX |
| Tensor math without leaving the track | + Compute layer |
| All of it on your phone | Core: Android AHardwareBuffer |

The race is the same. The track is finally connected.

*— the relay race analogy is Mike/ChonkE's; it cuts to the heart of what
Automaton/Chonk Buffer does.*
