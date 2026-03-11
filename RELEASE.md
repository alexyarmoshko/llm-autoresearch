# Release Notes

## Unreleased

### Changed

- Replaced the dead FlashAttention-3 assumption with a Linux-only FlashAttention-1 dependency path for Turing-class execution hosts.
- Updated the training runtime to prefer FlashAttention-1 when available and fall back safely to PyTorch SDPA.

### Notes

- This migration does not require rerunning `prepare.py` because dataset and tokenizer artifacts are independent of the attention kernel.
- On the current remote Torch 2.9.1 stack, `flash-attn==1.0.9` builds but fails to import at runtime, so this branch currently validates the fallback path rather than activating FlashAttention-1.
