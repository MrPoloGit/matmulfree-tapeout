# MatMul-free LM tapeout

Getting [`ridger/MMfreeLM`](https://huggingface.co/ridger/MMfreeLM-370M) —
the HGRN-bit architecture from "Scalable MatMul-free Language Modeling": a
linear-RNN mixer (f/i/g/o_proj) replacing self-attention entirely, no KV
cache, with BitLinear (ternary) weights throughout — through
[ankhdjet](https://github.com/mpai17/ankhdjet)'s weights-to-masks compiler,
and eventually onto SKY130 silicon.

This is the sibling of [`fbi-llm-tapeout`](../fbi-llm-tapeout), same split
of concerns: the *generic* HGRN-bit architecture support (config parsing,
IR construction) lives in ankhdjet itself, on the `matmulfree-support`
branch; this project tracks the real `ridger/MMfreeLM` checkpoints
specifically — the tests that decode and compile them for real, and,
eventually, the chip-level work.

## Why this one is different from FBI-LLM

FBI-LLM needed a genuinely new quantization scheme in ankhdjet (affine
binary, `wscale*sign(weight)+wbias`, new IR fields, new RTL). HGRN-bit
does not: every BitLinear weight here is a standard absmean-ternary QAT
master weight, the *same* scheme BitNet b1.58 already uses, just under
different module names (`f_proj`/`i_proj`/`g_proj`/`o_proj` instead of
`q_proj`/`k_proj`/`v_proj`/`o_proj`). The real architectural difference is
entirely in the *dataflow* (a linear-RNN mixer, no attention, no KV
cache) — not in how the weights are stored or decoded. Confirmed by
checking the real checkpoint directly, not assumed from the paper.

## Status

**Done — real checkpoint, real pipeline, bit-exact (ported from
ankhdjet's own investigation on the `matmulfree-support` branch):**

- `ridger/MMfreeLM-370M` and `-1.3B` configs parsed directly from their
  real `config.json`; `intermediate_size`'s real formula
  (`int(h*r*2/3)` truncated, then rounded **up** to the next multiple of
  256 — not round-to-nearest) found and fixed after it silently gave the
  wrong answer for 1.3B specifically (5376 vs the real checkpoint's
  5632) — the one size where the two formulas disagree, confirmed
  against the real `down_proj` tensor shape, not just the config.
- Every BitLinear weight — `f`/`i`/`g`/`o_proj`, `gate_proj`,
  `down_proj`, `embed`, and `lm_head` — decodes as `TERNARY` via
  ankhdjet's existing absmean ladder, bit-exact against an independent
  direct computation on the real safetensors tensors, at both released
  sizes checked (370M and 1.3B).
- The recurrent state's area cost scoped correctly:
  `HGRNBitArch.recurrent_state_values` — one `(num_heads, head_dim)`
  vector per block, confirmed against the training repo's `init_state`
  — not a matrix the way gated-delta-net's linear attention keeps.
- **RTL validation needed zero new files.** Because this decodes to
  plain `QuantScheme.TERNARY`, ankhdjet's existing, unmodified
  `emit_layer_nor` + `cirom_nor_tile.sv` compile its real matmuls
  correctly with no changes at all. All 6 of block 0's real projections
  from the real 370M checkpoint, sliced to 128×64 (Verilator's
  per-literal width limit; same convention as ankhdjet's own BitNet RTL
  tests): **6/6 bit-exact** (`tests/test_mmfreelm_block0_rtl.py`).

**Not done:**

1. `emit_model`/`verify_model` (the full-model compile + independent
   disk audit FBI-LLM went through) has not been attempted for this
   architecture at all — likely to "just work" the same way the
   single-layer RTL did, given everything above, but unverified.
2. The recurrent state's area isn't wired into `area_model.py`'s actual
   estimator pipeline yet — `recurrent_state_values` exists and is
   tested in isolation (on the `matmulfree-support` branch), but
   nothing calls it the way KV-cache fields are costed for attention
   architectures.
3. No chip-level/LibreLane work at all. No chip-top RTL, no LibreLane
   config, no signoff run — everything FBI-LLM has at the physical
   level (`cirom_chip_digital_affine`, a full clean DRC 0 / LVS 0 /
   timing-clean run) has no HGRN-bit counterpart yet. Given HGRN-bit
   reuses the *unmodified* ternary tile, the physical story should
   actually be simpler here than FBI-LLM's (no new controller RTL to
   sign off) — the real gap is the linear-RNN mixer's dataflow
   (sequencing f/i/g/o_proj and the elementwise gating across the
   recurrence), not the per-tile hardware.
4. No multi-layer/full-model dataflow, same caveat FBI-LLM's README
   states: everything validated so far is per-layer, not the real
   block's actual recurrent composition.

## Setup

```bash
# from this directory, with the matmulfree-support ankhdjet worktree
# checked out as a sibling (not the main ankhdjet checkout -- this is
# where the generic HGRN-bit architecture support actually lives):
uv sync

# tests download real checkpoints (370M, 1.3B) via huggingface_hub on
# first run and cache them normally; no conversion step needed (unlike
# FBI-LLM, these ship as safetensors already)
uv run pytest tests/ -q
```

If your `ankhdjet` (matmulfree-support) checkout isn't at
`../ankhdjet-matmulfree`, set `ANKHDJET_REPO` to its path — the tests and
`uv.sources` path dependency both honor it (update `pyproject.toml`'s
`[tool.uv.sources]` path too, since `uv sync` doesn't read env vars for
that).

Also requires `verilator` on `PATH` for the RTL test
(`tests/test_mmfreelm_block0_rtl.py`) — skips cleanly if missing, same as
the checkpoint gating.

## Layout

```
tests/test_mmfreelm_real_checkpoint.py   real config + weight decode,
                                          bit-exact against direct computation
tests/test_mmfreelm_block0_rtl.py        real block-0 projections -> the
                                          EXISTING unmodified ternary NOR
                                          tile RTL, bit-exact, 6/6
tests/_verilator_runner.py               shared Verilator build/run helper
                                          (copied from fbi-llm-tapeout, no
                                          ankhdjet import dependency)
```
