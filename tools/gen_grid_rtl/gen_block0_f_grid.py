"""Bazel genrule script: emit the full-size grid_rtl top + per-chunk
memh for HGRN-bit's real block-0 f_proj matrix (1024x1024, 512 chunks
at the GT2N-proven 64x32 tile size), for the Nautilus GT2N physical
design flow.

Reads the real checkpoint's f_proj tensor via safetensors'
framework="pt" loader (the stored dtype is bfloat16, which plain numpy
has no native dtype for -- safetensors.numpy's loader fails on it
outright; torch is the one dependency this script cannot dodge, same
as ankhdjet.frontend.hf's own loader) and replicates ankhdjet's
documented absmean-ternary transform in plain numpy (`scale =
1/mean(abs(w))`, `w_tern = clip(round(w/scale), -1, 1)`) rather than
importing ankhdjet.frontend.hf's general checkpoint loader.
ankhdjet.backend.grid_rtl (the RTL emitter) IS imported directly from
source -- it and its one dependency (idents.py) are pure Python with
no heavy deps, passed in as plain srcs rather than an installed
package.

f_proj is stored (out, in) = (1024, 1024) in the checkpoint, like
every non-embedding projection; ankhdjet's IR convention is
(input_dim, output_dim), so the quantized tensor is transposed before
emission -- matching exactly what
test_mmfreelm_full_size_grid_rtl.py's cocotb validation already proved
bit-exact for this layer via the real ankhdjet loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from grid_rtl import emit_grid

REPO_ID = "ridger/MMfreeLM-370M"
TENSOR_NAME = "model.layers.0.attn.f_proj.weight"
MACRO_ROWS, MACRO_COLS = 64, 32
ACT_W, ACC_W = 4, 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    shard = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
    with safe_open(shard, framework="pt") as f:
        raw = f.get_tensor(TENSOR_NAME).float().numpy()

    scale = max(float(np.abs(raw).mean()), 1e-5)
    q_direct = np.clip(np.round(raw / scale), -1, 1).astype(np.int8)
    W = q_direct.T  # (out, in) -> ankhdjet IR's (input_dim, output_dim)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = emit_grid(
        W, out_dir=out_dir, layer_name="b0_f", top_name="ankhdjet_grid_b0_f",
        macro_rows=MACRO_ROWS, macro_cols=MACRO_COLS,
        act_w=ACT_W, acc_w=ACC_W,
    )
    print(f"emitted b0_f grid: N={info['N']} M={info['M']} "
          f"grid_r={info['grid_r']} grid_c={info['grid_c']}")


if __name__ == "__main__":
    main()
