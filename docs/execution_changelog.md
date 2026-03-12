# Execution Changelog

## 2026-03-11

- Started a dedicated branch `codex/flashattention1-swap` to replace the unusable FlashAttention-3 assumption with a Turing-capable FlashAttention-1 path.
- Verified that the remote execution host is an RTX 2080 Ti (`sm_75`), where PyTorch efficient attention is available but PyTorch Flash SDPA is not.
- Added an execution plan to document the dependency migration, remote `uv` update flow, and validation criteria.
- Replaced the dead FA3 import path with an explicit FlashAttention-1-or-SDPA runtime selector and documented the Linux-only dependency in `pyproject.toml`.
- Generated a new `uv.lock` on the remote Linux host and copied it back to the local branch because Windows cannot build `flash-attn` 1.x while locking.
- Confirmed that `flash-attn==1.0.9` builds during `uv sync` on the remote host but fails to import at runtime against Torch 2.9.1, so the branch currently falls back to PyTorch SDPA.
- Prototyped a Torch 2.2.2 + CUDA 12.1 environment on the remote host, confirmed that FlashAttention-1 imports there, and updated the branch to use that compatibility line.
- Pinned `numpy<2` for the Torch 2.2 line, made the training code Torch-2.2-compatible, reduced `HEAD_DIM` to 64 to satisfy the Turing backward constraint, and validated a full FlashAttention-1 run.
- Recorded the first working FA1 result on the remote host: commit `1394c1e`, `Attention backend: flash-attn-v1`, `val_bpb = 1.850956`.

## 2026-03-12

- Continued the FA1 branch on the same remote `uv` environment and found that much smaller effective batches dominate under the fixed 5-minute budget on the RTX 2080 Ti.
- Advanced the frontier from `1394c1e` (`1.850956`) through repeated `TOTAL_BATCH_SIZE` reductions down to `9e333f2` (`1.308654`) at `2**13`.
- Confirmed that going below the minimum legal batch with `DEVICE_BATCH_SIZE = 4` is invalid, and that lowering both device batch and total batch (`8d95b31`) regresses despite more steps.
- Retuned the Muon small-batch neighborhood and bracketed a local optimum around `MATRIX_LR = 0.016`.
- Retuned weight decay on the small-batch frontier and found the current best at commit `affdc07`, `val_bpb = 1.305262`, with `WEIGHT_DECAY = 0.075`.
- Verified that a 5-layer model (`8399c67`) is close but still worse than the 6-layer configuration on this FA1 frontier.
- Documented that `WINDOW_PATTERN` is still a no-op on the active attention paths, so the most credible future gain requires making local attention real rather than continuing blind window-pattern sweeps.
