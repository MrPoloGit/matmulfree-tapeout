"""Full-model mask-program compile + independent audit against the
real ridger/MMfreeLM-370M checkpoint -- the same end-to-end rigor
ankhdjet's docs/results.md applies to microsoft/bitnet-b1.58-2B-4T
(emit every layer's mask program, then independently reassemble from
disk and compare bit-for-bit), and the same rigor
fbi-llm-tapeout/tests/test_fbi_llm_full_model_compile.py applies to
LiqunMa/FBI-LLM_130M. Unlike that FBI-LLM test, no `skip_layers` is
needed here: HGRN-bit's `embed` and `lm_head` are both genuinely
ternary in the real checkpoint (confirmed in
test_mmfreelm_real_checkpoint.py), not placeholder/tied data, so the
full 146-layer model (144 block projections + embed + lm_head) goes
through the fabric with nothing off to the side.

Depends on a sibling `ankhdjet` checkout, specifically the
matmulfree-support worktree (see README.md), for the Python package.

Gated on a local HF cache of ridger/MMfreeLM-370M, skipping cleanly
if absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def test_full_model_emission_and_audit(tmp_path: Path) -> None:
    if not (ANKHDJET_ROOT / "ankhdjet").is_dir():
        pytest.skip(f"no ankhdjet checkout found at {ANKHDJET_ROOT} "
                    f"(set ANKHDJET_REPO to override)")
    if not _hf_cache_has_repo("ridger/MMfreeLM-370M"):
        pytest.skip("ridger/MMfreeLM-370M not in local HF cache")

    from ankhdjet.backend.macro_grid import emit_model
    from ankhdjet.backend.verify import verify_model
    from ankhdjet.frontend.hf import load_weights
    from ankhdjet.frontend.ir import QuantScheme

    model, arch, scales = load_weights(
        "ridger/MMfreeLM-370M", progress=False, quantize="absmean")
    assert all(L.weights["weight"].scheme == QuantScheme.TERNARY for L in model.layers)

    man = emit_model(model, tmp_path, macro_rows=64, macro_cols=256, write_wmat=True)

    assert man["off_fabric"] == []
    layers = man["layers"]
    # embed + 6 projections/block * num_hidden_layers + lm_head
    assert len(layers) == 1 + 6 * arch.num_hidden_layers + 1
    # No `scheme`/`scale_axis` fields on this branch's GridManifest --
    # those were added as part of binary-support's work on a separate
    # branch; matmulfree-support's emit_layer_grid is ternary-only
    # (raises ValueError on anything else), so the manifest doesn't
    # need to record it. Check the real fields it does have instead.
    assert {L["layer"] for L in layers} == {
        "embed", "lm_head",
        *(f"b{i}_{p}" for i in range(arch.num_hidden_layers)
          for p in ("f", "i", "g", "o", "gate", "down")),
    }
    assert man["totals"]["weights"] == sum(
        L.weights["weight"].data.size for L in model.layers)

    audit = verify_model(model, tmp_path, jobs=8)
    assert audit["ok"]
    assert audit["n_ok"] == audit["n_layers"] == len(layers)
    assert audit["failures"] == []
