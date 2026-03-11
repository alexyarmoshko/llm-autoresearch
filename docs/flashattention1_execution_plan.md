# Replace Dead FA3 Assumptions with FlashAttention-1 on Turing

This Execution Plan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository follows [PLANS.md](C:/Users/alex/.codex/PLANS.md). This document must be maintained in accordance with that file.

## Purpose / Big Picture

After this change, the training code can use a real FlashAttention-family kernel on the remote RTX 2080 Ti instead of carrying a Hopper-oriented FlashAttention-3 path that is never exercised on this hardware. A user should be able to run `uv sync` on the Linux execution host, then `uv run train.py`, and observe that training starts with an explicit attention backend selection that prefers FlashAttention-1 when available and otherwise falls back safely to PyTorch SDPA.

## Progress

- [x] (2026-03-11 18:12Z) Created a dedicated branch `codex/flashattention1-swap` for the dependency and runtime change.
- [x] (2026-03-11 18:14Z) Confirmed via remote probing that the execution host is an RTX 2080 Ti (`sm_75`), PyTorch Flash SDPA is unavailable there, and PyTorch efficient attention is available.
- [x] (2026-03-11 20:05Z) Patched `pyproject.toml`, `train.py`, and supporting docs so Linux installs can build FlashAttention-1 and the runtime prints a backend choice with safe SDPA fallback.
- [x] (2026-03-11 20:15Z) Refreshed `uv.lock` on the remote Linux host and copied the resulting lockfile back into the local branch.
- [x] (2026-03-11 20:24Z) Updated the remote Linux environment with `uv sync` and ran a full training validation.
- [x] (2026-03-11 22:03Z) Prototyped and adopted a Torch/NumPy compatibility matrix that actually runs FlashAttention-1 on the remote RTX 2080 Ti.

## Surprises & Discoveries

- Observation: The active code path in `train.py` computes `window_size` but ignores it because `torch.nn.functional.scaled_dot_product_attention` is called without any sliding-window argument.
  Evidence: local inspection of `train.py` showed the `fa3.flash_attn_func(..., window_size=window_size)` line commented out while the SDPA fallback remains active.

- Observation: The remote RTX 2080 Ti cannot use PyTorch Flash SDPA even though the global backend flag is enabled.
  Evidence: remote probe returned `can_use_flash_attention False` with a warning that flash attention only supports `sm80+`.

- Observation: FlashAttention-1 version `1.0.9` builds on the remote Linux host but fails to import at runtime against the current Torch stack.
  Evidence: validation log reported `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` from `flash_attn_cuda...so`, after `uv sync` had already built and installed `flash-attn==1.0.9`.

- Observation: FlashAttention-1 does run on the RTX 2080 Ti when the environment is moved to Torch 2.2.2 + CUDA 12.1 wheels + NumPy 1.26, and the model head dimension is reduced to 64.
  Evidence: remote validation run `1394c1e_run.log` printed `Attention backend: flash-attn-v1` and completed with `val_bpb: 1.850956`.

## Decision Log

- Decision: Treat this task as a dependency-and-runtime migration instead of another hyperparameter experiment.
  Rationale: The current FlashAttention-3 assumption is dead code on `sm_75`, so replacing it is infrastructure work that changes what experiments are even possible.
  Date/Author: 2026-03-11 / Codex

- Decision: Perform the work on `codex/flashattention1-swap` rather than the experiment branch.
  Rationale: The user explicitly requested a new branch, and isolating the dependency migration reduces risk to the current tuned training frontier.
  Date/Author: 2026-03-11 / Codex

- Decision: Move the branch from Torch 2.9.1 + CUDA 12.8 wheels to Torch 2.2.2 + CUDA 12.1 wheels, and pin NumPy below 2.
  Rationale: FlashAttention-1 imported successfully only on the older Torch line, and Torch 2.2 warned about NumPy 2.x ABI compatibility.
  Date/Author: 2026-03-11 / Codex

- Decision: Reduce `HEAD_DIM` from 128 to 64 on the FA1 branch.
  Rationale: FlashAttention-1 on Turing requires backward head dimension at most 64; larger heads fail during backpropagation.
  Date/Author: 2026-03-11 / Codex

## Outcomes & Retrospective

The migration first produced a safe fallback branch on the original Torch stack, then a working FlashAttention-1 branch after the compatibility matrix was adjusted. The final working recipe on the remote RTX 2080 Ti is: Torch 2.2.2 from the CUDA 12.1 wheel index, NumPy 1.26.4, FlashAttention-1 version 1.0.9, and model head dimension 64. With that stack, `uv sync` succeeds, `uv run train.py` prints `Attention backend: flash-attn-v1`, and the validation run improves to `val_bpb: 1.850956`.

## Context and Orientation

This repository has three main files relevant to this change. `prepare.py` handles dataset download, tokenizer training, and evaluation helpers; it does not depend on the attention kernel choice and should not need to be rerun for this migration. `train.py` contains the full model and currently imports `kernels.get_kernel` to fetch a FlashAttention-3 interface, but the actual attention call is a PyTorch SDPA fallback. `pyproject.toml` defines the project dependencies and is the right place to express a Linux-only FlashAttention-1 dependency for the remote host.

The remote execution host is a Linux machine with an NVIDIA GeForce RTX 2080 Ti. In NVIDIA naming, this is a Turing GPU with compute capability `sm_75`. FlashAttention-3 is aimed at newer Hopper-class GPUs and is not usable here. FlashAttention-1 is the FlashAttention family variant that remains relevant for Turing hardware.

## Plan of Work

First, update `pyproject.toml` so Linux environments can install FlashAttention-1 while Windows development on the local machine does not attempt to build it. At the same time, remove the unused `kernels` integration if no live code path requires it. Then patch `train.py` near the top-level imports and inside `CausalSelfAttention.forward` to choose an attention backend explicitly: prefer FlashAttention-1 when importable and the tensor layout is compatible, otherwise fall back to the existing SDPA path. The code must print which backend is selected so a novice can verify behavior from the training log.

After the code changes, refresh `uv.lock` from the local repository so the local repo remains the source of truth. Then copy the changed files to the remote execution environment and run `uv sync` there to update only that remote project environment. Finally, run `uv run train.py` on the remote host and inspect the log for both the backend selection message and the training summary.

## Concrete Steps

From `C:\Users\alex\repos\llm-autoresearch` on the local machine:

    git checkout -b codex/flashattention1-swap
    uv lock

From the remote Linux execution directory copied from this repo:

    uv sync
    uv run train.py > run.log 2>&1
    grep -E "Attention backend|val_bpb|peak_vram_mb" run.log

Expected signs of success:

    Attention backend: flash-attn-v1
    val_bpb:          ...
    peak_vram_mb:     ...

If FlashAttention-1 cannot be imported or built, the log must still show a safe fallback:

    Attention backend: torch-sdpa

## Validation and Acceptance

Acceptance requires all of the following:

1. `uv sync` completes on the remote Linux host without rerunning `prepare.py`.
2. `uv run train.py` starts successfully and prints the selected attention backend.
3. The training job completes the usual 5-minute budget and prints a normal summary containing `val_bpb`.
4. The local repository contains updated dependency metadata, the execution plan, the changelog, and release notes for this migration.

## Idempotence and Recovery

These steps are safe to repeat. Re-running `uv sync` should only reconcile the environment to the checked-in lockfile. If FlashAttention-1 fails to build or import, the code keeps a PyTorch SDPA fallback so training remains runnable. Recovery means reverting the dependency and backend-selection patches while leaving `prepare.py` data and tokenizer artifacts untouched.

## Artifacts and Notes

Remote hardware probe before implementation:

    device NVIDIA GeForce RTX 2080 Ti
    capability (7, 5)
    can_use_flash_attention False
    can_use_efficient_attention True

The official FlashAttention guidance relevant here is: Turing GPUs such as RTX 2080 should use FlashAttention 1.x; newer FlashAttention paths target Ampere or newer GPUs.

## Interfaces and Dependencies

`train.py` must end this work with a small backend-selection interface that can answer two questions: whether FlashAttention-1 is importable, and which attention backend is being used for the current run. The dependency surface should be limited to `flash-attn` on Linux and PyTorch SDPA as the fallback. The runtime should continue to call a single attention helper from `CausalSelfAttention.forward` so future experiments do not need to know which backend is active.

Revision note: created the initial self-contained plan before code changes so the FA1 migration can be resumed from this file alone.

Revision note: updated after remote validation to record that FlashAttention-1 builds successfully but fails to import at runtime against Torch 2.9.1, leaving the branch on the safe SDPA fallback path.

Revision note: updated after the Torch 2.2 / NumPy 1.26 / head-dim-64 compatibility pass to record the first end-to-end FlashAttention-1 run on the remote Turing host.
