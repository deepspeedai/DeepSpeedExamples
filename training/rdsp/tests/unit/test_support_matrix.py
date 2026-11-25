# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The support matrix is default-deny and internally consistent."""

from ray_deepspeed_pipeline.support_matrix import BASELINE_ROW, HETEROGENEOUS_ROWS, ROWS


def test_default_deny():
    # a row claims support only with a recorded passing acceptance run; the
    # first row has passed its own
    assert BASELINE_ROW.supported
    for row in HETEROGENEOUS_ROWS:
        assert not row.supported or row.evidence, row.row_id


def test_first_row_definition():
    assert BASELINE_ROW.row_id == "two-stage-baseline"
    assert (BASELINE_ROW.stages, BASELINE_ROW.gpus_per_stage) == (2, 1)
    assert BASELINE_ROW.dtype == "bf16" and BASELINE_ROW.zero_stage == 0
    assert BASELINE_ROW.schedule == "1f1b" and BASELINE_ROW.boundaries == "identity"
    assert (BASELINE_ROW.dp, BASELINE_ROW.tp, BASELINE_ROW.ep, BASELINE_ROW.sp) == (1, 1, 1, 1)


def test_row_ids_unique():
    ids = [r.row_id for r in ROWS]
    assert len(ids) == len(set(ids))


def test_composed_row_requires_every_prerequisite():
    by_id = {r.row_id: r for r in HETEROGENEOUS_ROWS}
    for row in HETEROGENEOUS_ROWS:
        assert all(req in by_id for req in row.requires), row.row_id
        if row.supported:
            assert all(by_id[req].supported for req in row.requires), (
                f"{row.row_id} cannot be supported before {row.requires}"
            )
    assert set(by_id["four-stage-mixed"].requires) == set(by_id) - {"four-stage-mixed"}


def test_rows_carry_no_project_history():
    """Row ids and evidence name what was validated, not when or in which phase."""
    import re

    for row in ROWS:
        text = " ".join(str(v) for v in vars(row).values())
        assert not re.search(r"\bp\d+-|20\d\d-\d\d-\d\d|§|rerun after", text), row.row_id
