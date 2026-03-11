# Execution Changelog

## 2026-03-11

- Started a dedicated branch `codex/flashattention1-swap` to replace the unusable FlashAttention-3 assumption with a Turing-capable FlashAttention-1 path.
- Verified that the remote execution host is an RTX 2080 Ti (`sm_75`), where PyTorch efficient attention is available but PyTorch Flash SDPA is not.
- Added an execution plan to document the dependency migration, remote `uv` update flow, and validation criteria.
- Replaced the dead FA3 import path with an explicit FlashAttention-1-or-SDPA runtime selector and documented the Linux-only dependency in `pyproject.toml`.
- Generated a new `uv.lock` on the remote Linux host and copied it back to the local branch because Windows cannot build `flash-attn` 1.x while locking.
- Confirmed that `flash-attn==1.0.9` builds during `uv sync` on the remote host but fails to import at runtime against Torch 2.9.1, so the branch currently falls back to PyTorch SDPA.
