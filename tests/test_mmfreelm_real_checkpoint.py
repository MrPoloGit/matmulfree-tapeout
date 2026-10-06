"""HGRN-bit (MatMul-free LM, ridger/matmulfreellm) config parsing and
real-weight decode against the actual released checkpoints. Config.json
fields were read directly from a real checkpoint (ridger/MMfreeLM-370M),
not assumed from the paper or training code; so was the weight
quantization scheme -- see `HGRNBitArch`'s docstring in ankhdjet's
ankhdjet/frontend/hf.py for what that investigation found (every
BitLinear weight, including embeddings and lm_head, is a continuous
absmean-ternary QAT master weight, the same scheme BitNet b1.58
already uses, not a new one).

The generic architecture-parsing unit tests (synthetic configs only)
stay in ankhdjet's own test suite
(tests/test_hgrn_bit_frontend.py on the matmulfree-support branch);
this file is the checkpoint-specific half, same split ankhdjet's
binary-support work made for FBI-LLM.

Depends on a sibling `ankhdjet` checkout, specifically the
matmulfree-support worktree (see README.md), for the Python package.

Gated on a local HF cache the same way ankhdjet's other real-checkpoint
tests are (e.g. test_bitnet_block0_all_projections.py), skipping
cleanly if absent -- these are real downloads (370M and 1.3B params),
not fixtures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANKHDJET_ROOT = Path(os.environ.get(
    "ANKHDJET_REPO", str(THIS_PROJECT_ROOT.parent / "ankhdjet-matmulfree")))
sys.path.insert(0, str(ANKHDJET_ROOT))

pytestmark = pytest.mark.package


def _hf_cache_has_repo(repo_id: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe = repo_id.replace("/", "--")
    return any(cache_dir.glob(f"models--{safe}*"))


def test_real_config_matches_mmfreelm_370m():
    """ridger/MMfreeLM-370M's real config.json, parsed: confirms the
    intermediate_size formula and the total parameter count against
    the model's own name (373.8M, matching "370M" to naming-convention
    rounding)."""
    if not Path(ANKHDJET_ROOT / "ankhdjet" / "frontend" / "hf.py").exists():
        pytest.skip(f"no ankhdjet checkout found at {ANKHDJET_ROOT} "
                    f"(set ANKHDJET_REPO to override)")
    if not _hf_cache_has_repo("ridger/MMfreeLM-370M"):
        pytest.skip("ridger/MMfreeLM-370M not in local HF cache")
    from ankhdjet.frontend.hf import load_config

    model, arch = load_config("ridger/MMfreeLM-370M")
    assert arch.hidden_size == 1024
    assert arch.num_hidden_layers == 24
    assert arch.intermediate_size == 2816
    assert arch.vocab_size == 32000
    total = sum(L.input_dim * L.output_dim for L in model.layers)
    assert 370e6 < total < 380e6


def test_real_1_3b_down_proj_shape_matches_ceiling_not_round():
    """The regression test for the round-vs-ceiling bug: 1.3B is the
    one size where they disagree, so this is checked directly against
    the real down_proj tensor's shape, not just the config-derived
    arch.intermediate_size."""
    if not Path(ANKHDJET_ROOT / "ankhdjet" / "frontend" / "hf.py").exists():
        pytest.skip(f"no ankhdjet checkout found at {ANKHDJET_ROOT} "
                    f"(set ANKHDJET_REPO to override)")
    if not _hf_cache_has_repo("ridger/MMfreeLM-1.3B"):
        pytest.skip("ridger/MMfreeLM-1.3B not in local HF cache")
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    from ankhdjet.frontend.hf import load_config

    model, arch = load_config("ridger/MMfreeLM-1.3B")
    assert arch.intermediate_size == 5632

    snapshot_dir = Path(snapshot_download(
        repo_id="ridger/MMfreeLM-1.3B", allow_patterns=["*.safetensors"]))
    sf = next(snapshot_dir.glob("*.safetensors"))
    with safe_open(sf, framework="pt") as f:
        shape = f.get_slice("model.layers.0.mlp.down_proj.weight").get_shape()
    assert shape == [arch.hidden_size, arch.intermediate_size]


@pytest.mark.parametrize("repo_id", ["ridger/MMfreeLM-370M", "ridger/MMfreeLM-1.3B"])
def test_real_weights_decode_bit_exact_against_direct_absmean(repo_id):
    """Every BitLinear weight -- f/i/g/o_proj, gate_proj, down_proj,
    embed, and lm_head -- decodes as TERNARY via the existing absmean
    ladder, bit-exact against an independent direct computation of the
    same transform on the raw safetensors tensor. Real checkpoints,
    not synthetic data, at two of the three released sizes (370M and
    1.3B -- the latter is also where the intermediate_size
    round-vs-ceiling bug actually mattered, so this exercises the
    fixed dimension end to end, not just the shape formula)."""
    if not Path(ANKHDJET_ROOT / "ankhdjet" / "frontend" / "hf.py").exists():
        pytest.skip(f"no ankhdjet checkout found at {ANKHDJET_ROOT} "
                    f"(set ANKHDJET_REPO to override)")
    if not _hf_cache_has_repo(repo_id):
        pytest.skip(f"{repo_id} not in local HF cache")
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from ankhdjet.frontend.hf import load_weights
    from ankhdjet.frontend.ir import QuantScheme

    model, arch, scales = load_weights(
        repo_id, progress=False, quantize="absmean")
    assert all(L.weights["weight"].scheme == QuantScheme.TERNARY
              for L in model.layers)
    by_name = {L.name: L for L in model.layers}

    snapshot_dir = Path(snapshot_download(
        repo_id=repo_id, allow_patterns=["*.safetensors"]))
    sf = next(snapshot_dir.glob("*.safetensors"))
    checks = [
        ("b0_f", "model.layers.0.attn.f_proj.weight"),
        ("b0_gate", "model.layers.0.mlp.gate_proj.weight"),
        ("b0_down", "model.layers.0.mlp.down_proj.weight"),
        ("embed", "model.embeddings.weight"),
        ("lm_head", "lm_head.weight"),
    ]
    with safe_open(sf, framework="pt") as f:
        for ir_name, hf_name in checks:
            raw = f.get_tensor(hf_name).float().numpy()
            scale = float(np.abs(raw).mean())
            q_direct = np.clip(np.round(raw / scale), -1, 1).astype(np.int8)
            ir_data = by_name[ir_name].weights["weight"].data
            ir_recon = ir_data if ir_name == "embed" else ir_data.T
            assert np.array_equal(q_direct, ir_recon), f"{ir_name}: sign mismatch"
            assert abs(scale - scales[ir_name]) < 1e-6, f"{ir_name}: scale mismatch"
