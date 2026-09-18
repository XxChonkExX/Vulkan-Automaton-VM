# experiments/ — one-off probes, quarantined from the repo root

These scripts were single-session experiments, kept for archaeology. They are
NOT part of the build, the test suite, or any documented workflow. Nothing
references them; do not add new dependencies on them.

| Script | What it was |
|---|---|
| `edge_sweep.sh` / `edge_test.py` | Edge-case sweeps from an early hardening pass |
| `probe_mem.py` | Manual memory-probing helper (superseded by `igpu_probe_test`) |
| `test_allocator.py` | Early allocator smoke (superseded by `buddy_test` / `chonk_slab_test`) |
| `test_foreign_ptr.py` | Foreign-pointer import probe (superseded by `external_handle_test`) |

If you need one, run it from here. If it becomes load-bearing, promote it
to `tests/` with assertions instead of extending it in place.
