"""Bit-exact validation that HGRN-bit's real decoded weights compile
through ankhdjet's EXISTING ternary NOR-array RTL (emit_layer_nor +
cirom_nor_tile.sv) completely unmodified -- no new RTL file, no new
tile controller, unlike FBI-LLM's affine-binary case. This is possible
because HGRN-bit decodes to plain QuantScheme.TERNARY (see ankhdjet's
docs/matmulfree_hgrn_bit_investigation.md on the matmulfree-support
branch): every BitLinear weight is a standard absmean-ternary QAT
master weight, the same scheme BitNet b1.58 already uses, just under
different module names.

Mirrors ankhdjet's tests/test_bitnet_block0_all_projections.py: slice
each real block-0 matrix to 128x64 (Verilator's per-literal width
limit rejects the full matrix even as a parameter override), compile
through emit_layer_nor + cirom_nor_tile, run Verilator, compare
bit-exact against ternary_matmul_nor.

Depends on a sibling `ankhdjet` checkout, specifically the
matmulfree-support worktree (see README.md), for the Python package
and for rtl/column/cirom_nor_tile.sv (not part of ankhdjet's
pip-packaged `_rtl` bundle).

Gated on a local HF cache of ridger/MMfreeLM-370M, skipping cleanly
if absent.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANKHDJET_ROOT = Path(os.environ.get(
    "ANKHDJET_REPO", str(THIS_PROJECT_ROOT.parent / "ankhdjet-matmulfree")))
sys.path.insert(0, str(ANKHDJET_ROOT))
sys.path.insert(0, str(THIS_PROJECT_ROOT))

from tests._verilator_runner import build_and_run

pytestmark = pytest.mark.package

TILE_SV = ANKHDJET_ROOT / "rtl" / "column" / "cirom_nor_tile.sv"

# The 6 real BitLinear matmuls of HGRN-bit block 0 (see
# hgrn_bit_block_projections in ankhdjet): no q/k/v/o, no up_proj.
BLOCK0_MATRICES = ["b0_f", "b0_i", "b0_g", "b0_o", "b0_gate", "b0_down"]

SLICE_N = 128
SLICE_M = 64
K_BITS = 8
SUBCOL = 64

TB = r"""
`timescale 1ns/1ps
module tb_hgrn_proj;
    localparam int N = {N};
    localparam int M = {M};
    localparam int K = {K};
    localparam int WACC = {WACC};
    logic clk, rst_n, start, valid;
    logic [K*N-1:0]              act_flat;
    logic signed [WACC*M-1:0]    acc_flat;
    logic [K-1:0]                act_byte [0:N-1];

    ankhdjet_layer_{NAME} dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act_flat(act_flat),
        .acc_flat(acc_flat), .valid(valid)
    );

    always #5 clk = ~clk;
    int i, j, cycles, acc_val;

    initial begin
        $readmemh("act.hex", act_byte);
        for (i = 0; i < N; i = i + 1) begin
            act_flat[K*(i+1)-1 -: K] = act_byte[i];
        end
        clk = 0; rst_n = 0; start = 0;
        #23 rst_n = 1;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cycles = 0;
        while (!valid && cycles < 200000) begin
            @(negedge clk);
            cycles++;
        end
        if (!valid) begin $display("TIMEOUT"); $finish; end
        for (j = 0; j < M; j = j + 1) begin
            acc_val = $signed(acc_flat[WACC*(j+1)-1 -: WACC]);
            $display("OUT[%0d]=%0d", j, acc_val);
        end
        $display("CYCLES=%0d", cycles);
        $finish;
    end
endmodule
"""


def _hf_cache_has_repo(repo_id: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe = repo_id.replace("/", "--")
    return any(cache_dir.glob(f"models--{safe}*"))


@pytest.fixture(scope="module")
def mmfreelm_370m_model():
    if not TILE_SV.exists():
        pytest.skip(f"no ankhdjet checkout found at {ANKHDJET_ROOT} "
                    f"(set ANKHDJET_REPO to override)")
    if not _hf_cache_has_repo("ridger/MMfreeLM-370M"):
        pytest.skip("ridger/MMfreeLM-370M not in local HF cache")
    from ankhdjet.frontend.hf import load_weights
    model, _, _ = load_weights("ridger/MMfreeLM-370M", progress=False, quantize="absmean")
    return model


@pytest.mark.parametrize("mat_name", BLOCK0_MATRICES)
def test_block0_matrix_bit_exact_through_nor_tile(mmfreelm_370m_model, mat_name, tmp_path):
    from ankhdjet.backend.verilog import _column_widths, emit_layer_nor
    from ankhdjet.frontend.ir import Layer, LayerType, QuantScheme, WeightTensor
    from ankhdjet.reference.nor import ternary_matmul_nor

    layer_full = next(L for L in mmfreelm_370m_model.layers if L.name == mat_name)
    full = layer_full.weights["weight"].data
    assert layer_full.weights["weight"].scheme == QuantScheme.TERNARY
    full_n, full_m = full.shape
    n = min(SLICE_N, full_n)
    m = min(SLICE_M, full_m)
    W = full[:n, :m].astype(np.int8)

    rng = np.random.default_rng(seed=0xA2A4A + hash(mat_name) % (2**31))
    act = rng.integers(0, 2**K_BITS, size=n, dtype=np.int64)

    expected = ternary_matmul_nor(W, act, k_bits=K_BITS, subcol_rows=SUBCOL).astype(np.int64)

    layer = Layer(
        name=f"{mat_name}_slice", layer_type=LayerType.LINEAR,
        input_dim=n, output_dim=m,
        weights={"weight": WeightTensor(name="weight", data=W, scheme=QuantScheme.TERNARY)},
    )

    layer_v = tmp_path / f"ankhdjet_layer_{layer.name}.sv"
    layer_v.write_text(emit_layer_nor(layer, k_bits=K_BITS, subcol_rows=SUBCOL))

    _, wacc = _column_widths(n, K_BITS)
    tb_src = TB.format(N=n, M=m, K=K_BITS, WACC=wacc, NAME=layer.name)
    (tmp_path / "tb.sv").write_text(tb_src)
    (tmp_path / "act.hex").write_text("\n".join(f"{int(a):x}" for a in act) + "\n")

    stdout = build_and_run(
        workdir=tmp_path,
        sources=[TILE_SV, layer_v, tmp_path / "tb.sv"],
        top="tb_hgrn_proj",
        build_timeout=600.0, run_timeout=600.0,
    )
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")

    measured = np.zeros(m, dtype=np.int64)
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))

    assert np.array_equal(measured, expected), (
        f"{mat_name}: mismatch on {int((measured != expected).sum())}/{m} columns "
        f"(+1/0/-1 = {(W==1).sum()}/{(W==0).sum()}/{(W==-1).sum()})")
