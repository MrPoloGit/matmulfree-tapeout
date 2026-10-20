"""Macro wrapping gen_block0_f_grid as a genrule with the exact output
list the real b0_f geometry produces (16x32 chunks of 64x32 at the
GT2N-proven tile size -- see gen_block0_f_grid.py), so BUILD.bazel
doesn't hand-enumerate 1,026 filenames. The genrule's action is cached
by Bazel's own disk_cache like any other action (keyed on the
deterministic command + tool inputs): once it has run, the Nautilus
job's persistent bazel-cache PVC means later flow stages reuse the
output instead of re-downloading the checkpoint and re-emitting it.
"""

def b0_f_grid_files(layer_name = "b0_f", grid_r = 16, grid_c = 32):
    """The exact output paths gen_block0_f_grid.py writes for one
    GRID_R x GRID_C grid (top/afe .sv plus per-chunk pos/neg memh)."""
    memh = []
    for i in range(grid_r):
        for j in range(grid_c):
            memh.append("{}/r{}_c{}.wpos.memh".format(layer_name, i, j))
            memh.append("{}/r{}_c{}.wneg.memh".format(layer_name, i, j))
    return [
        "ankhdjet_grid_{}.sv".format(layer_name),
        "cirom_grid_afe.sv",
    ] + memh

def gen_b0_f_grid(name, out_dir = "gen"):
    """Runs gen_block0_f_grid.py as a genrule, declaring its full,
    exact output list (see b0_f_grid_files) under out_dir/."""
    outs = ["{}/{}".format(out_dir, f) for f in b0_f_grid_files()]
    native.genrule(
        name = name,
        outs = outs,
        tools = ["//tools/gen_grid_rtl:gen_block0_f_grid"],
        cmd = "$(location //tools/gen_grid_rtl:gen_block0_f_grid) --out-dir $(RULEDIR)/{}".format(out_dir),
        # Real checkpoint download (huggingface.co); not hermetic by
        # Bazel's usual standard, but no different in kind from the
        # orfs/openroad archive fetches already in this build graph --
        # cached on disk once materialized, see module docstring.
        tags = ["requires-network"],
    )
