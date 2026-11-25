# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Static boundaries: gather-and-resplit between differing stage grids."""

import itertools

import pytest
import torch

from ray_deepspeed_pipeline.boundary import (
    Cell,
    Grid,
    assemble,
    cell_slice,
    conversion_name,
    overlapping_cells,
    rank_cell,
    rank_coords,
    representatives,
    slice_inputs,
)
from ray_deepspeed_pipeline.errors import StepFailed

GRIDS = [
    Grid(1, 1),
    Grid(2, 1),
    Grid(4, 1),
    Grid(8, 1),
    Grid(2, 2),
    Grid(4, 2),
    Grid(1, 2),
    Grid(2, 1, tp=4),
    Grid(4, 1, tp=2),
]


@pytest.mark.parametrize("src,dest", list(itertools.product(GRIDS, GRIDS)))
def test_static_boundary_reshard_equals_slice_of_whole(src, dest):
    """Every destination cell rebuilt from overlapping source cells equals
    the same region cut from the full tensor — for every grid pair."""
    torch.manual_seed(0)
    full = torch.randn(8, 16, 3)
    pieces = {
        Cell(d, src.dp, q, src.sp): cell_slice(full, Cell(d, src.dp, q, src.sp))
        for d in range(src.dp)
        for q in range(src.sp)
    }
    for r in representatives(dest):
        cell = rank_cell(dest, r)
        needed = overlapping_cells(src, cell)
        got = assemble([(c, pieces[c]) for c in needed], cell)
        assert torch.equal(got, cell_slice(full, cell)), (src, dest, cell)
        # it never needs more than the overlapping sources
        assert len(needed) <= max(1, src.dp // dest.dp + 1) * max(1, src.sp // dest.sp + 1)


def test_identity_boundary_does_not_copy():
    t = torch.randn(4, 5)
    c = Cell(1, 2)
    assert assemble([(c, t)], c) is t


def test_integer_token_ids_reshard():
    ids = torch.arange(8 * 6).reshape(8, 6)
    src, dest = Grid(2, 1), Grid(4, 1)
    pieces = {c: cell_slice(ids, c) for c in (Cell(0, 2), Cell(1, 2))}
    cell = rank_cell(dest, 3)
    assert cell == Cell(3, 4)
    needed = overlapping_cells(src, cell)
    assert needed == [Cell(1, 2)], "the last quarter lies inside the second half"
    out = assemble([(c, pieces[c]) for c in needed], cell)
    assert torch.equal(out, ids[6:8])
    assert out.dtype == ids.dtype


def test_canonical_rank_order_tp_fastest():
    g = Grid(dp=2, sp=1, tp=4)
    assert [rank_coords(g, r) for r in range(8)] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
    ]
    assert representatives(g) == [0, 4]


def test_indivisible_rows_fail_loudly():
    with pytest.raises(StepFailed, match="divisible"):
        cell_slice(torch.randn(6, 2), Cell(0, 4))


def test_missing_piece_detected():
    full = torch.randn(4, 2)
    with pytest.raises(StepFailed, match="cover"):
        assemble([(Cell(0, 2), cell_slice(full, Cell(0, 2)))], Cell(0, 1))


def test_conversion_names():
    assert conversion_name(Grid(1, 1), Grid(1, 1, tp=4)) == "identity"
    assert conversion_name(Grid(1, 1), Grid(4, 1)) == "replicate-to-shard"
    assert conversion_name(Grid(4, 1), Grid(1, 1, tp=2)) == "shard-to-replicate"
    assert conversion_name(Grid(4, 2), Grid(2, 1)) == "shard-to-shard"


@pytest.mark.parametrize("src,dest", list(itertools.product(GRIDS, GRIDS)))
def test_p2p_routing_is_symmetric_and_complete(src, dest):
    """Every (source -> destination) message one side expects, the other side
    sends — and each destination rank gets exactly the cells it needs."""
    from ray_deepspeed_pipeline.boundary import p2p_destinations, p2p_sources

    sends = {(s, d) for s in range(src.world) for d in p2p_destinations(src, dest, s)}
    recvs = {(s, d) for d in range(dest.world) for s, _ in p2p_sources(src, dest, d)}
    assert sends == recvs
    for d in range(dest.world):
        cells = [c for _, c in p2p_sources(src, dest, d)]
        assert cells == overlapping_cells(src, rank_cell(dest, d))


def test_stage_inputs_dict_split_per_rank():
    """Row-shaped values are sliced like any boundary tensor; per-row lists
    (images packed per row) are cut by row and concatenated per rank."""
    ids = torch.arange(12).reshape(4, 3)
    pixels = [torch.full((n, 2), float(r)) for r, n in enumerate((1, 0, 2, 3))]
    inputs = {"input_ids": ids, "pixel_values": pixels}

    first = slice_inputs(inputs, Cell(0, 2))
    second = slice_inputs(inputs, Cell(1, 2))

    assert torch.equal(first["input_ids"], ids[:2])
    assert torch.equal(second["input_ids"], ids[2:])
    assert first["pixel_values"].tolist() == [[0.0, 0.0]]
    assert second["pixel_values"].tolist() == [[2.0, 2.0]] * 2 + [[3.0, 3.0]] * 3


def test_stage_inputs_tensor_unchanged_behaviour():
    x = torch.arange(8).reshape(4, 2)
    assert torch.equal(slice_inputs(x, Cell(1, 2)), cell_slice(x, Cell(1, 2)))


def test_per_row_lists_cannot_be_sequence_split():
    inputs = {"input_ids": torch.zeros(2, 4), "pixel_values": [torch.zeros(1, 2)] * 2}
    with pytest.raises(StepFailed, match="sequence"):
        slice_inputs(inputs, Cell(0, 1, 0, 2))


def test_boundary_block_arguments_follow_the_hidden_state():
    from ray_deepspeed_pipeline.boundary import assemble_boundary

    src, dest = Grid(1, 1), Grid(1, 2)
    hidden = torch.randn(2, 8, 4)
    cos = torch.randn(2, 8, 3)
    pieces = [(rank_cell(src, 0), (hidden, {"0/cos": cos}))]
    for rank in range(2):
        cell = rank_cell(dest, rank)
        x, extras = assemble_boundary(pieces, cell)
        assert torch.equal(x, cell_slice(hidden, cell))
        assert torch.equal(extras["0/cos"], cell_slice(cos, cell))


def test_a_4d_attention_mask_cannot_be_split_across_sequence_shards():
    from ray_deepspeed_pipeline.boundary import assemble_boundary

    hidden, mask = torch.randn(2, 8, 4), torch.zeros(2, 1, 8, 8)
    pieces = [(rank_cell(Grid(1, 1), 0), (hidden, {"0/attention_mask": mask}))]
    with pytest.raises(StepFailed, match="attention_mask.*sequence"):
        assemble_boundary(pieces, rank_cell(Grid(1, 2), 0))
    # without sequence parallelism the mask is split by rows only
    _, extras = assemble_boundary(pieces, rank_cell(Grid(2, 1), 1))
    assert torch.equal(extras["0/attention_mask"], mask[1:])
