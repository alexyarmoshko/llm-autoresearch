# FlashAttention-1 Frontier Handoff

## Status

- Branch: `codex/flashattention1-swap`
- Current best commit: `affdc07`
- Current best `val_bpb`: `1.305262`
- Best log: `results/affdc07_run.log`
- Remote execution environment: `alex@r2d2.dahaus:~/runs/llm-autoresearch-fa1-setup`
- Local repo remains the source of truth. Only copy `train.py` and pull logs/results back.

## Best Known Configuration

The active best configuration in `train.py` is:

```python
HEAD_DIM = 64
TOTAL_BATCH_SIZE = 2**13
MATRIX_LR = 0.016
WEIGHT_DECAY = 0.075
DEPTH = 6
DEVICE_BATCH_SIZE = 4
```

Runtime assumptions for this branch:

- `torch==2.2.2`
- `numpy==1.26.4`
- `flash-attn==1.0.9`
- FlashAttention-1 active on the RTX 2080 Ti
- `torch.compile` disabled on this compatibility line

## What Changed

The biggest win was not a subtle hyperparameter tweak. The major improvement came from moving to the working Turing FA1 stack and then lowering `TOTAL_BATCH_SIZE` aggressively under the fixed 5-minute budget:

- `1394c1e`: first working FA1 run, `1.850956`
- `a50c1b8`: `TOTAL_BATCH_SIZE = 2**16`, `1.689385`
- `6ebe1b5`: `TOTAL_BATCH_SIZE = 2**15`, `1.607238`
- `f7a28d6`: `TOTAL_BATCH_SIZE = 2**14`, `1.568847`
- `9e333f2`: `TOTAL_BATCH_SIZE = 2**13`, `1.308654`
- `c617902`: `MATRIX_LR = 0.016`, `1.306282`
- `affdc07`: `WEIGHT_DECAY = 0.075`, `1.305262`

## What Looks Exhausted

These nearby probes did not improve the frontier:

- Lowering microbatch and effective batch together (`8d95b31`) regressed.
- Reducing depth to 5 (`8399c67`) regressed.
- `MATRIX_LR = 0.0125` and `0.017` both regressed.
- `WEIGHT_DECAY = 0.05` and midpoint `0.085` both regressed.

This means the local FA1 small-batch neighborhood is bracketed reasonably well around:

- `TOTAL_BATCH_SIZE = 2**13`
- `DEVICE_BATCH_SIZE = 4`
- `MATRIX_LR = 0.016`
- `WEIGHT_DECAY = 0.075`

## Important Caveat

`WINDOW_PATTERN` is currently a misleading knob. The code computes window sizes, but the active attention implementations do not apply them:

- FA1 path ignores `window_size`
- SDPA fallback also ignores `window_size`

So any search over `"SSSL"` versus `"L"` is currently a no-op unless the attention code is changed.

## Best Next Steps

If continuing from here, the highest-value directions are:

1. Implement real local/sliding attention on the FA1 or fallback path so `WINDOW_PATTERN` actually affects computation.
2. Retune Adam-side hyperparameters on the small-batch frontier, especially `EMBEDDING_LR`, `UNEMBEDDING_LR`, and possibly Adam betas.
3. Re-test simplicity ablations on the FA1 frontier, especially value embeddings and `x0` mixing, because the old negative results were from a different regime.
4. Revisit width/aspect-ratio changes only after one of the above, since the old width tests were on materially different settings.

## Resume Checklist

When resuming:

1. Confirm branch is still `codex/flashattention1-swap`.
2. Confirm `train.py` still matches the best settings above.
3. Sync only `train.py` to `~/runs/llm-autoresearch-fa1-setup/`.
4. Run `uv run train.py` remotely via `bash -lc`.
5. Copy `run.log` back into `results/<commit>_run.log`.
6. Update `results.tsv` locally only.
