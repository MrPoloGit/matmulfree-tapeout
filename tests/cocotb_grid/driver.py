"""cocotb driver for a grid_rtl-emitted top (cirom_grid_ctrl + behavioral
per-chunk arrays): full-size, real-weight ternary matmul validation.

Protocol matches the real silicon-proven tile's (cirom_dig_ctrl) and the
earlier FBI-LLM chip-top cocotb driver's byte-serial convention, minus
the cfg_mode/cfg_in bit-serial config load -- cirom_grid_ctrl has no
scale/bias config, it is plain ternary: load N activation bytes over
ui/act_wr, pulse start, collect NB = 2*M result bytes (low byte then
high byte per output column, ascending column, per the controller's own
comment), reconstruct signed ACC_W-bit values.
"""

from __future__ import annotations

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def test_grid(dut) -> None:
    n = int(os.environ["ANKHDJET_N"])
    m = int(os.environ["ANKHDJET_M"])
    act_w = int(os.environ["ANKHDJET_ACT_W"])
    acc_w = int(os.environ["ANKHDJET_ACC_W"])
    out_w = 8
    grpb = 2  # cirom_grid_ctrl always streams exactly 2 bytes/column
    nbytes = m * grpb

    act = [int(x) for x in os.environ["ANKHDJET_ACT_CSV"].split(",")]
    assert len(act) == n

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.ui.value = 0
    dut.act_wr.value = 0
    dut.start.value = 0

    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)

    for i in range(n):
        dut.ui.value = act[i] & ((1 << act_w) - 1)
        dut.act_wr.value = 1
        await RisingEdge(dut.clk)
    dut.act_wr.value = 0
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    rx: list[int] = []
    cycles = 0
    max_cycles = n * 20 + nbytes * 2 + 5000
    while len(rx) < nbytes and cycles < max_cycles:
        await RisingEdge(dut.clk)
        if dut.result_valid.value == 1:
            rx.append(int(dut.result.value))
        cycles += 1
    assert len(rx) == nbytes, f"timeout: got {len(rx)}/{nbytes} result bytes"

    while dut.done.value != 1:
        await RisingEdge(dut.clk)

    out = []
    for i in range(m):
        v = rx[i * grpb] | (rx[i * grpb + 1] << out_w)
        if v >= (1 << (acc_w - 1)):
            v -= 1 << acc_w
        out.append(v)

    result_path = os.environ["ANKHDJET_RESULT_PATH"]
    with open(result_path, "w") as f:
        f.write(",".join(str(v) for v in out))
