"""Full-size (not sliced), real-weight RTL validation of HGRN-bit's 6
real BitLinear matmuls via the grid_rtl/cirom_grid_ctrl tiling
mechanism -- the same mechanism behind the macro-grid work, generalized
to multi-chunk layers, at the GT2N-proven tile size (64x32, matching
cirom_dig_ctrl's real silicon dimensions) -- rather than the 128x64
*slice* through the simpler emit_layer_nor generator that
test_mmfreelm_block0_rtl.py uses. This is the direct full-size upgrade
of that validation: real layer-0 weight matrices at their actual
shapes (1024x1024 for f/i/g/o_proj, 1024x5632 for gate_proj, 2816x1024
for down_proj), tiled into real macro chunks, simulated via cocotb +
Verilator, and checked bit-exact against a plain ternary matmul
reference (expected[m] = sum_n act[n] * W[n, m], W in {-1, 0, +1}).

ACT_W is picked per matrix so the controller's fixed 16-bit signed
accumulator (cirom_grid_ctrl hardcodes 2 result bytes/column; see its
acc_hi_b expression, which requires ACC_W <= 16) cannot overflow even
in the worst case where every row's bitline agrees in sign: max |acc|
= N * (2**ACT_W - 1) must stay under 2**15. 1024-row matrices (f/i/g/o,
gate) fit ACT_W=4 (matching cirom_dig_ctrl's own real value) with
margin; down_proj's N=2816 needs ACT_W=3.

Depends on a sibling `ankhdjet` checkout (matmulfree-support worktree)
for ankhdjet.backend.grid_rtl and rtl/grid/{cirom_grid_ctrl,
cirom_array_beh}.sv. Gated on a local HF cache of
ridger/MMfreeLM-370M and on cocotb + Verilator, skipping cleanly if
either is absent.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANKHDJET_ROOT = Path(os.environ.get(
    "ANKHDJET_REPO", str(THIS_PROJECT_ROOT.parent / "ankhdjet-matmulfree")))
sys.path.insert(0, str(ANKHDJET_ROOT))
sys.path.insert(0, str(THIS_PROJECT_ROOT))

pytestmark = pytest.mark.package

GRID_CTRL_SV = ANKHDJET_ROOT / "rtl" / "grid" / "cirom_grid_ctrl.sv"
ARRAY_BEH_SV = ANKHDJET_ROOT / "rtl" / "grid" / "cirom_array_beh.sv"

MACRO_ROWS, MACRO_COLS = 64, 32  # the GT2N-proven cirom_dig_ctrl tile size
ACC_W = 16


def _safe_act_w(n: int, acc_w: int = ACC_W) -> int:
    acc_max = 2 ** (acc_w - 1) - 1
    w = 1
    while n * (2 ** (w + 1) - 1) <= acc_max:
        w += 1
    return w


BLOCK0_MATRICES = ["b0_f", "b0_i", "b0_g", "b0_o", "b0_gate", "b0_down"]

# All 24 layers' 6 real BitLinear projections (144 total) -- embed/lm_head
# (32000x1024, ~16,000 chunks each at the GT2N tile size) are excluded:
# full RTL simulation at that chunk count isn't practical the way the
# 24 per-block matrices are, and they're already bit-audited at the
# mask-program level by emit_model/verify_model.
ALL_BLOCK_MATRICES = [f"b{i}_{p}" for i in range(24)
                      for p in ("f", "i", "g", "o", "gate", "down")]


def _hf_cache_has_repo(repo_id: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe = repo_id.replace("/", "--")
    return any(cache_dir.glob(f"models--{safe}*"))


def _validate_projection(mat_name: str, W, out_root: str,
                         act=None, act_w: int | None = None,
                         expected=None) -> tuple[str, bool, str]:
    """Emit, build, and cocotb-simulate one real projection's full-size
    grid RTL; return (mat_name, passed, detail). Module-level and
    picklable so it can run under a process pool.

    By default (act=None) generates a deterministic random activation
    and checks against the plain ternary matmul. Passing `act` (and
    `expected`) overrides this -- used for `embed`'s one-hot/row-lookup
    check, where the "activation" is a one-hot selector rather than a
    real quantized input."""
    import shutil as _shutil
    from pathlib import Path as _Path

    import numpy as _np
    from cocotb_tools.runner import get_runner as _get_runner

    from ankhdjet.backend.grid_rtl import emit_grid as _emit_grid

    out_dir = _Path(out_root) / mat_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n, m = W.shape
    if act is None:
        act_w = _safe_act_w(n)
        rng = _np.random.default_rng(seed=0xA2A4A + hash(mat_name) % (2**31))
        act = rng.integers(0, 2**act_w, size=n, dtype=_np.int64)
        expected = (act.astype(_np.int64) @ W.astype(_np.int64)).astype(_np.int64)

    top_name = f"ankhdjet_grid_{mat_name}"
    info = _emit_grid(
        W, out_dir=out_dir, layer_name=mat_name, top_name=top_name,
        macro_rows=MACRO_ROWS, macro_cols=MACRO_COLS,
        act_w=act_w, acc_w=ACC_W,
    )
    if info["N"] < n or info["M"] < m:
        return mat_name, False, "grid padding shrank the matrix"

    result_path = out_dir / "result.csv"
    build_dir = out_dir / "sim_build"
    try:
        runner = _get_runner("verilator")
        runner.build(
            sources=[str(GRID_CTRL_SV), str(ARRAY_BEH_SV),
                    str(info["afe"]), str(info["top"])],
            hdl_toplevel=top_name,
            build_args=["--timing", "-Wno-fatal"],
            build_dir=str(build_dir),
            waves=False,
        )
        _shutil.copy(_Path(__file__).parent / "cocotb_grid" / "driver.py",
                    out_dir / "driver.py")
        runner.test(
            hdl_toplevel=top_name,
            test_module="driver",
            test_dir=str(out_dir),
            build_dir=str(build_dir),
            extra_env={
                "ANKHDJET_N": str(n),
                "ANKHDJET_M": str(m),
                "ANKHDJET_ACT_W": str(act_w),
                "ANKHDJET_ACC_W": str(ACC_W),
                "ANKHDJET_ACT_CSV": ",".join(str(int(a)) for a in act),
                "ANKHDJET_RESULT_PATH": str(result_path),
            },
        )
    except SystemExit as e:
        return mat_name, False, f"cocotb/verilator failed (exit {e.code})"

    if not result_path.exists():
        return mat_name, False, "cocotb test did not write a result file"
    measured = _np.array([int(x) for x in result_path.read_text().split(",")],
                         dtype=_np.int64)
    if not _np.array_equal(measured, expected):
        bad = int((measured != expected).sum())
        return mat_name, False, f"mismatch on {bad}/{m} columns"
    return mat_name, True, "ok"


@pytest.fixture(scope="module")
def mmfreelm_370m_model():
    if not GRID_CTRL_SV.exists():
        pytest.skip(f"no ankhdjet checkout found at {ANKHDJET_ROOT} "
                    f"(set ANKHDJET_REPO to override)")
    if not _hf_cache_has_repo("ridger/MMfreeLM-370M"):
        pytest.skip("ridger/MMfreeLM-370M not in local HF cache")
    from ankhdjet.frontend.hf import load_weights
    model, _, _ = load_weights("ridger/MMfreeLM-370M", progress=False, quantize="absmean")
    return model


@pytest.mark.parametrize("mat_name", BLOCK0_MATRICES)
def test_full_size_matrix_bit_exact_through_grid_rtl(mmfreelm_370m_model, mat_name, tmp_path):
    if shutil.which("verilator") is None:
        pytest.skip("verilator not found on PATH")
    try:
        import cocotb_tools.runner  # noqa: F401
    except ImportError:
        pytest.skip("cocotb not installed")

    layer = next(L for L in mmfreelm_370m_model.layers if L.name == mat_name)
    W = layer.weights["weight"].data.astype(np.int8)
    name, ok, detail = _validate_projection(mat_name, W, str(tmp_path))
    assert ok, f"{name} ({W.shape[0]}x{W.shape[1]}): {detail}"


@pytest.mark.slow
def test_full_model_all_block_matrices_bit_exact_through_grid_rtl(
        mmfreelm_370m_model, tmp_path):
    """The full upgrade: all 24 layers' 6 real BitLinear projections
    (144 total, every one of them at real size, real weights), fanned
    out across a process pool the same way macro_grid's emit_model
    parallelizes per-layer work. embed/lm_head excluded -- see
    ALL_BLOCK_MATRICES' comment."""
    if shutil.which("verilator") is None:
        pytest.skip("verilator not found on PATH")
    try:
        import cocotb_tools.runner  # noqa: F401
    except ImportError:
        pytest.skip("cocotb not installed")

    by_name = {L.name: L for L in mmfreelm_370m_model.layers}
    calls = []
    for mat_name in ALL_BLOCK_MATRICES:
        W = by_name[mat_name].weights["weight"].data.astype(np.int8)
        calls.append((mat_name, _validate_projection,
                      (mat_name, W, str(tmp_path)), {}))

    from ankhdjet.backend._pool import run_ordered

    def _progress(stage, done, total, label):
        print(f"[{done}/{total}] {label}", flush=True)

    results = run_ordered(calls, jobs=2, progress=_progress, stage="grid_rtl")

    failures = [(name, detail) for name, ok, detail in results if not ok]
    print(f"\n{len(results) - len(failures)}/{len(results)} projections "
          f"bit-exact through full-size grid RTL")
    assert not failures, "failures:\n" + "\n".join(
        f"  {name}: {detail}" for name, detail in failures)


@pytest.mark.slow
def test_lm_head_full_size_bit_exact_through_grid_rtl(mmfreelm_370m_model, tmp_path):
    """lm_head (1024x32000, 16,000 chunks at the GT2N tile size) -- the
    same real matmul structure as the 144 block projections, just the
    largest one in the model. ~16,000 behavioral chunk instances pushes
    Verilator compile time to ~25-30 minutes (confirmed feasible, not
    assumed); the cocotb sim itself is fast (~1.5 min) once built."""
    if shutil.which("verilator") is None:
        pytest.skip("verilator not found on PATH")
    try:
        import cocotb_tools.runner  # noqa: F401
    except ImportError:
        pytest.skip("cocotb not installed")

    layer = next(L for L in mmfreelm_370m_model.layers if L.name == "lm_head")
    W = layer.weights["weight"].data.astype(np.int8)
    name, ok, detail = _validate_projection("lm_head", W, str(tmp_path))
    assert ok, f"{name} ({W.shape[0]}x{W.shape[1]}): {detail}"


@pytest.mark.slow
def test_embed_full_size_bit_exact_through_grid_rtl(mmfreelm_370m_model, tmp_path):
    """embed (32000x1024, also 16,000 chunks) is a lookup table, not a
    reduction -- the real access pattern is "read row token_id", not a
    dot product. Validated via a one-hot activation (a single row's
    act=1, every other row act=0), which is mathematically equivalent
    to the row lookup and exercises the exact same grid_rtl/
    cirom_grid_ctrl machinery as every other real projection: expected
    is the real embedding row read directly, not a matmul reference."""
    if shutil.which("verilator") is None:
        pytest.skip("verilator not found on PATH")
    try:
        import cocotb_tools.runner  # noqa: F401
    except ImportError:
        pytest.skip("cocotb not installed")

    layer = next(L for L in mmfreelm_370m_model.layers if L.name == "embed")
    W = layer.weights["weight"].data.astype(np.int8)
    n, m = W.shape

    rng = np.random.default_rng(0xE4BED)
    token_id = int(rng.integers(0, n))
    act = np.zeros(n, dtype=np.int64)
    act[token_id] = 1
    expected = W[token_id].astype(np.int64)

    name, ok, detail = _validate_projection(
        "embed", W, str(tmp_path), act=act, act_w=1, expected=expected)
    assert ok, f"{name} ({n}x{m}, token_id={token_id}): {detail}"
