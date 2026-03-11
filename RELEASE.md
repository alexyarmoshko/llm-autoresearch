# Release Notes

## Unreleased

### Changed

- Replaced the dead FlashAttention-3 assumption with a Linux-only FlashAttention-1 dependency path for Turing-class execution hosts.
- Updated the training runtime to prefer FlashAttention-1 when available and fall back safely to PyTorch SDPA.

### Notes

- This migration does not require rerunning `prepare.py` because dataset and tokenizer artifacts are independent of the attention kernel.
