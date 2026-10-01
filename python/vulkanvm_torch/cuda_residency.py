"""cuda_residency.py -- ghost/double detector for CUDA residency.

Counts BYTES BY UNIQUE STORAGE, not by parameter: two Parameters sharing
one storage (tied lm_head, views) count once; two live copies of the same
weights (unified + text, stale wraps, duplicated optimizer shadows) count
twice and get flagged. One-line audit at lifecycle checkpoints:

    from cuda_residency import audit
    audit(model, expected_gb=24.0, tag="post-load")     # prints, warns

Ghost taxonomy (observed in the wild, Gemma-4 campaign):
  - unified-then-text double load (27GB + 24GB resident pre-train)
  - PEFT wrap transient base copies during get_peft_model
  - optimizer .state_dict() copies alongside live states
  - host shadow buffers retained after pool migration
  - (LEGIT) EMA shadow copies -- allowlist via expected_gb or tag
"Racing ourselves": parallel launches stacking identical models --
check process single-flight alongside residency.
"""
import torch


def storage_bytes(module, include_buffers=True):
    seen = {}
    params = list(module.named_parameters())
    if include_buffers:
        try:
            params += [(n, b) for n, b in module.named_buffers()
                       if isinstance(b, torch.Tensor)]
        except Exception:
            pass
    for n, t in params:
        if not isinstance(t, torch.Tensor) or t.device.type == "meta":
            continue
        st = t.untyped_storage() if hasattr(t, "untyped_storage") \
            else t.storage()
        key = st.data_ptr()
        if key not in seen:
            seen[key] = (st.nbytes(), n)
    return seen


def audit(module, expected_gb=None, tag="", warn_ratio=1.25,
          allowlist=()):
    seen = storage_bytes(module)
    total = sum(b for b, _ in seen.values()) / 1e9
    cuda_only = sum(b for (b, _n) in seen.values()) / 1e9
    line = (f"[residency:{tag}] {total:.2f}GB across "
            f"{len(seen)} storages")
    if expected_gb is not None:
        ratio = total / max(expected_gb, 1e-9)
        line += f" (expect ~{expected_gb:.1f}GB, x{ratio:.2f})"
        if ratio > warn_ratio and tag not in allowlist:
            line += "  *** GHOSTS: possible double residency ***"
    print(line, flush=True)
    return total, seen


def single_flight(name, lock_dir="/tmp"):
    """Refuse duplicate training processes (the 'racing ourselves' guard).
    Returns True if this process holds the lock."""
    import fcntl
    path = f"{lock_dir}/.{name}.single_flight.lock"
    try:
        fh = open(path, "w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        single_flight._fh = fh  # hold for process lifetime
        return True
    except OSError:
        return False
