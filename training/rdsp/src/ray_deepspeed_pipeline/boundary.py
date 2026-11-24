# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Stage-boundary layout conversion.

Every stage lays one global microbatch out over its ranks as a grid:

    rows      (dim 0) split evenly into `dp` shards  — data parallel
    sequence  (dim 1) split evenly into `sp` shards  — sequence parallel
    replicated over `tp` ranks                         — tensor parallel

A rank's *cell* is its (dp_index, sp_index); its TP peers share the cell and
hold identical boundary tensors, so one representative per cell (tp_index 0)
is the source of truth. When adjacent stages use different grids, each
destination rank assembles its own cell from exactly the source cells that
overlap it, never materializing the whole microbatch anywhere. Gradients
use the same routine in reverse.

Grid geometry compares fractions of the global extent by integer
cross-multiplication, so routing works before tensor sizes are known.
"""

from dataclasses import dataclass

import torch

from ray_deepspeed_pipeline.errors import StepFailed


@dataclass(frozen=True)
class Grid:
    dp: int = 1
    sp: int = 1
    tp: int = 1

    @property
    def world(self) -> int:
        return self.dp * self.sp * self.tp


@dataclass(frozen=True)
class Cell:
    dp_index: int
    dp: int
    sp_index: int = 0
    sp: int = 1


def rank_coords(grid: Grid, rank: int) -> tuple[int, int, int]:
    """(dp, sp, tp) indices of a rank. TP varies fastest, then SP, then DP, so
    consecutive ranks form a TP group (the DeepSpeed/Megatron convention)."""
    t = rank % grid.tp
    q = (rank // grid.tp) % grid.sp
    d = rank // (grid.tp * grid.sp)
    return d, q, t


def rank_cell(grid: Grid, rank: int) -> Cell:
    d, q, _ = rank_coords(grid, rank)
    return Cell(d, grid.dp, q, grid.sp)


def representatives(grid: Grid) -> list[int]:
    """One rank per cell (tp_index 0), in cell order."""
    return [r for r in range(grid.world) if rank_coords(grid, r)[2] == 0]


def _overlap(i: int, n: int, j: int, m: int) -> bool:
    # [i/n, (i+1)/n) intersects [j/m, (j+1)/m)
    return i * m < (j + 1) * n and j * n < (i + 1) * m


def overlapping_cells(src: Grid, dest: Cell) -> list[Cell]:
    return [
        Cell(d, src.dp, q, src.sp)
        for d in range(src.dp)
        if _overlap(d, src.dp, dest.dp_index, dest.dp)
        for q in range(src.sp)
        if _overlap(q, src.sp, dest.sp_index, dest.sp)
    ]


def _span(index: int, parts: int, total: int, what: str) -> tuple[int, int]:
    if total % parts:
        raise StepFailed(f"{what} extent {total} is not divisible by {parts} shards")
    size = total // parts
    return index * size, (index + 1) * size


def cell_slice(tensor: torch.Tensor, cell: Cell) -> torch.Tensor:
    """The cell's region of a full (global-microbatch) tensor."""
    r0, r1 = _span(cell.dp_index, cell.dp, tensor.shape[0], "row")
    out = tensor[r0:r1]
    if cell.sp > 1:
        s0, s1 = _span(cell.sp_index, cell.sp, tensor.shape[1], "sequence")
        out = out[:, s0:s1]
    return out


def slice_inputs(inputs, cell: Cell):
    """The cell's share of a first-stage input: a tensor, or a dict whose
    values are row-shaped tensors or per-row lists. A per-row list holds
    values that are not row-shaped (a row's packed image patches); the cell's
    rows are concatenated along dim 0."""
    if not isinstance(inputs, dict):
        return cell_slice(inputs, cell)
    out = {}
    for name, value in inputs.items():
        if not isinstance(value, list):
            out[name] = cell_slice(value, cell)
            continue
        if cell.sp > 1:
            raise StepFailed(
                f"input {name!r} is given per row and cannot be split across sequence shards"
            )
        r0, r1 = _span(cell.dp_index, cell.dp, len(value), "row")
        out[name] = torch.cat(value[r0:r1])
    return out


def assemble(pieces: list[tuple[Cell, torch.Tensor]], dest: Cell) -> torch.Tensor:
    """Build dest's region from source pieces that together cover it."""
    if not pieces:
        raise StepFailed("no source pieces for a boundary transfer")
    if len(pieces) == 1 and pieces[0][0] == dest:
        return pieces[0][1]  # identity boundary: no copy
    first_cell, first = pieces[0]
    rows = first.shape[0] * first_cell.dp
    uses_seq = first_cell.sp > 1 or dest.sp > 1
    seq = first.shape[1] * first_cell.sp if uses_seq else None

    r0, r1 = _span(dest.dp_index, dest.dp, rows, "row")
    shape = list(first.shape)
    shape[0] = r1 - r0
    if uses_seq:
        s0, s1 = _span(dest.sp_index, dest.sp, seq, "sequence")
        shape[1] = s1 - s0
    out = torch.empty(shape, dtype=first.dtype, device=first.device)

    covered = 0
    for cell, piece in pieces:
        pr0, pr1 = _span(cell.dp_index, cell.dp, rows, "row")
        lo, hi = max(pr0, r0), min(pr1, r1)
        if lo >= hi:
            continue
        src = piece[lo - pr0 : hi - pr0]
        if uses_seq:
            ps0, ps1 = _span(cell.sp_index, cell.sp, seq, "sequence")
            slo, shi = max(ps0, s0), min(ps1, s1)
            if slo >= shi:
                continue
            out[lo - r0 : hi - r0, slo - s0 : shi - s0] = src[:, slo - ps0 : shi - ps0]
            covered += (hi - lo) * (shi - slo)
        else:
            out[lo - r0 : hi - r0] = src
            covered += hi - lo
    expected = shape[0] * (shape[1] if uses_seq else 1)
    if covered != expected:
        raise StepFailed(
            f"boundary pieces cover {covered} of {expected} elements of the destination cell {dest}"
        )
    return out


def assemble_boundary(pieces: list[tuple[Cell, tuple]], dest: Cell) -> tuple:
    """dest's (hidden state, block arguments) from the source cells'
    (hidden, extras). Under sequence parallelism every block argument must be
    laid out rows x sequence like the hidden state; one that is not (a 4D
    attention mask) fails the step."""
    hidden = assemble([(cell, x) for cell, (x, _) in pieces], dest)
    first_cell, (first, extras) = pieces[0]
    if first_cell.sp > 1 or dest.sp > 1:
        for name, t in extras.items():
            if name.startswith("@"):  # stage-to-stage settings, not block arguments
                continue
            if t.dim() < 2 or t.shape[1] != first.shape[1]:
                raise StepFailed(
                    f"block argument {name!r} of shape {tuple(t.shape)} is not laid out "
                    f"rows x sequence, so it cannot be split across sequence shards; "
                    f"sequence parallelism needs blocks that get no 4D attention mask "
                    f"(sdpa attention, no padding mask)"
                )
    out = {
        name: assemble([(cell, got[name]) for cell, (_, got) in pieces], dest) for name in extras
    }
    return hidden, out


def conversion_name(src: Grid, dest: Grid) -> str:
    """Declarative name of a boundary's layout conversion."""
    s, d = (src.dp, src.sp), (dest.dp, dest.sp)
    if s == d:
        return "identity"
    if s == (1, 1):
        return "replicate-to-shard"
    if d == (1, 1):
        return "shard-to-replicate"
    return "shard-to-shard"


def gradient_scale(src_avg_degree: int, dest_avg_degree: int) -> float:
    """Factor for a gradient passed from the stage averaging over
    `src_avg_degree` ranks to the one averaging over `dest_avg_degree`.

    DeepSpeed averages gradients over data-parallel ranks and sums them over
    sequence shards, so the gradient entering stage s must be dp_s x
    dL/d(output of s); crossing a boundary rescales by dp_s / dp_{s+1}."""
    return dest_avg_degree / src_avg_degree


# --- point-to-point routing: the same overlap geometry, seen from each end ---


def p2p_sources(src: Grid, dest: Grid, dest_rank: int) -> list[tuple[int, Cell]]:
    """(source rank, source cell) pairs a destination rank receives from, one
    representative per overlapping source cell."""
    cell = rank_cell(dest, dest_rank)
    rep_of = {rank_cell(src, r): r for r in representatives(src)}
    return [(rep_of[c], c) for c in overlapping_cells(src, cell)]


def p2p_destinations(src: Grid, dest: Grid, src_rank: int) -> list[int]:
    """Destination ranks a source rank sends to. Only TP rank 0 sends (its TP
    peers hold identical tensors); every TP peer of an overlapping
    destination cell receives its own copy."""
    if rank_coords(src, src_rank)[2] != 0:
        return []
    mine = rank_cell(src, src_rank)
    return [r for r in range(dest.world) if mine in overlapping_cells(src, rank_cell(dest, r))]
