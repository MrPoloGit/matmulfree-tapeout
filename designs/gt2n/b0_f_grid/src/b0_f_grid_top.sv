// b0_f_grid_top: cirom_grid_ctrl parameterized at HGRN-bit's real
// block-0 f_proj geometry (1024x1024, tiled 16x32 chunks of 64x32 at
// the GT2N-proven cirom_dig_ctrl tile size) -- the physical-design
// scale-up of the single tile already synthesized/placed/routed on
// GT2N. Matches cirom_dig_ctrl's own scope exactly: blp/bln are
// top-level ports (raw bitlines in, sampled here), the mask-programmed
// array itself is a separate hardened block stitched in later, not
// part of this GDS. See cirom_grid_ctrl.sv's own header for the
// controller's sweep/accumulate protocol.
`default_nettype none

module b0_f_grid_top (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [7:0]            ui,
    input  logic                  act_wr,
    input  logic                  start,
    output logic [1024-1:0]       wl,
    output logic                  pre_n,
    input  logic [16384-1:0]      blp,
    input  logic [16384-1:0]      bln,
    output logic [7:0]            result,
    output logic                  result_valid,
    output logic                  busy,
    output logic                  done
);
    cirom_grid_ctrl #(
        .GRID_R(16), .GRID_C(32), .MR(64), .MC(32),
        .ACT_W(4), .ACC_W(16)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n), .ui(ui), .act_wr(act_wr), .start(start),
        .wl(wl), .pre_n(pre_n), .blp(blp), .bln(bln),
        .result(result), .result_valid(result_valid),
        .busy(busy), .done(done)
    );
endmodule

`default_nettype wire
