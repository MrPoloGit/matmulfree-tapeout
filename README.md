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
- **The complete real 370M model compiles and audits clean.**
  `emit_model` against all 146 real layers (embed + 6 projections ×
  24 blocks + lm_head): **0 off-fabric** — unlike FBI-LLM, `embed` and
  `lm_head` are both genuinely ternary here, nothing needs
  `skip_layers`. The independent bit-for-bit disk audit (`verify_model`)
  passes all 146/146 (`tests/test_mmfreelm_full_model_compile.py`) —
  the same rigor ankhdjet's own docs report for
  `microsoft/bitnet-b1.58-2B-4T`, now confirmed for this architecture
  too, not just projected from the single-layer RTL result above.

**Not done:**

1. The recurrent state's area isn't wired into `area_model.py`'s actual
   estimator pipeline yet — `recurrent_state_values` exists and is
   tested in isolation (on the `matmulfree-support` branch), but
   nothing calls it the way KV-cache fields are costed for attention
   architectures.
2. No chip-level/LibreLane work at all. No chip-top RTL, no LibreLane
   config, no signoff run — everything FBI-LLM has at the physical
   level (`cirom_chip_digital_affine`, a full clean DRC 0 / LVS 0 /
   timing-clean run) has no HGRN-bit counterpart yet. Given HGRN-bit
   reuses the *unmodified* ternary tile, the physical story should
   actually be simpler here than FBI-LLM's (no new controller RTL to
   sign off) — the real gap is the linear-RNN mixer's dataflow
   (sequencing f/i/g/o_proj and the elementwise gating across the
   recurrence), not the per-tile hardware.
3. No multi-layer/full-model dataflow, same caveat FBI-LLM's README
   states: everything validated so far is per-layer, not the real
   block's actual recurrent composition.

## Full-model tapeout area estimate

Unlike FBI-LLM (whose binary scheme the area estimator doesn't support
yet, forcing a hand derivation from the raw cell density alone),
HGRN-bit decodes to plain `QuantScheme.TERNARY` — the project's real,
calibrated estimator (`ankhdjet/estimate/area_model.py`,
`sky130_v4.yaml`) runs directly against the full, already-verified
370M model, no approximation needed.

**Full 370M model, digital readout** (the tier actually signed off in
silicon for this project's other vehicles), recurrent state included
as `state_bytes` (bf16, kept in the same SRAM a KV cache would use):

| | cells | periphery | die overhead | **total** |
|---|---|---|---|---|
| mm² | 826 | 1,942 | 415 | **3,183** |

Periphery is 61% of the die — not the cells. This is the per-column
decoder/sampler/accumulator cost repeated across all 146 layers, not
just the two vocab-sized ones: excluding `embed`+`lm_head` entirely
(65.5M params) only brings the total down to **2,712 mm²**. Analog
readout instead of digital is worse (**6,134 mm²** — confirms digital,
the tier this project actually built, is the right call at this scale).

**This also means the earlier FBI-LLM estimate (~214 mm²) was too
optimistic** and should be treated as a rough lower bound, not a real
number — it only applied raw cell density plus a flat +14% overhead
borrowed from a 22M-weight reference shape. Given periphery alone
came in at over 2x the cell area here, FBI-LLM's true full-model
number is almost certainly much larger. Worth re-deriving properly
the same way once there's time (see `fbi-llm-tapeout`'s README).

Either number is **far beyond any single reticle** (max ~600-858 mm²
depending on process) — reticle size is the hard wall, not just
fabrication cost.

### Comparison: Taalas

[Taalas](https://wavect.io/blog/taalas-hc1-llm-asic-review/) is a real,
shipped company doing the same fundamental thing this project does —
put a specific model's weights directly into silicon instead of
reading them from external memory — at a completely different level
of engineering maturity. Their HC1 chip: TSMC 6 nm, 815 mm² die, 53
billion transistors, a quantized (mixed 3-bit/6-bit) Llama 3.1 8B
**on one chip**, ~17,000 tokens/sec per user, AMD acquired the company
in early 2026.

| | process | die area | params | area per Mparam |
|---|---|---|---|---|
| Taalas HC1 | 6 nm | 815 mm² | 8,000M | 0.10 mm² |
| This estimate (370M, digital) | 130 nm (SKY130) | 3,183 mm² | 370M | 8.60 mm² |

Raw, that's **~84x worse area efficiency per parameter**. Almost all
of that gap is the process node, not the architecture: 130 nm vs 6 nm
is a ~21.7x linear, ~469x area difference for equivalent structures
under naive node² scaling. Applying that factor the other way —
shrinking this estimate's 370M number to a hypothetical 6 nm process —
gives **~6.8 mm²**, actually *better* than Taalas's real 37.7 mm²
(their own density applied to 370M params). Take that with real
caveats, not as a claim this architecture beats Taalas's: node²
scaling is a crude rule of thumb that undercounts real advanced-node
costs (parasitics, reliability margins, SRAM/analog structures scale
worse than logic), Taalas ships a real, validated product at full
engineering maturity while this is a calibrated-but-unbuilt estimate,
and Taalas's 3/6-bit mixed precision needs more bits per weight than
this project's ternary cells, which should cost *them* more area per
parameter, not less — so the comparison is genuinely informative
about where the gap actually lives (mostly node, not concept), but not
a result to oversell.

Sources:
- [Taalas HC1 Review: Hardwired LLM ASIC](https://wavect.io/blog/taalas-hc1-llm-asic-review/)
- [AMD Acquires Taalas, Startup Etching AI Weights Into Silicon](https://aiweekly.co/alerts/amd-acquires-taalas-startup-etching-ai-weights-into-silicon)

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
tests/test_mmfreelm_full_model_compile.py  full 146-layer compile + audit,
                                          0 off-fabric, 146/146 verified
tests/_verilator_runner.py               shared Verilator build/run helper
                                          (copied from fbi-llm-tapeout, no
                                          ankhdjet import dependency)
```
