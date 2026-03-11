# Release Notes

## Unreleased

### Changed

- Replaced the dead FlashAttention-3 assumption with a Linux-only FlashAttention-1 dependency path for Turing-class execution hosts.
- Updated the training runtime to prefer FlashAttention-1 when available and fall back safely to PyTorch SDPA.
- Moved the FA1 branch to a compatible runtime stack: Torch 2.2.2 on the CUDA 12.1 wheel index, plus NumPy 1.26.x.
- Adjusted the FA1 branch to use 64-dim attention heads so FlashAttention-1 backward works on Turing GPUs.

### Notes

- This migration does not require rerunning `prepare.py` because dataset and tokenizer artifacts are independent of the attention kernel.
- The original Torch 2.9.1 stack only validated the fallback path because `flash-attn==1.0.9` failed to import at runtime there.
- On the compatible FA1 stack, the remote RTX 2080 Ti run completed with `Attention backend: flash-attn-v1` and `val_bpb = 1.850956`.
