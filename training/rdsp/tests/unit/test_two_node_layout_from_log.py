# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""bench/two_node_35b/scripts/layout_from_log.py: Bridge's cut and the vision stage's
width for the two-node runs, read from rdsp's run 1 log rather than picked by hand."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench", "two_node_35b", "scripts"))

import layout_from_log as lfl  # noqa: E402

LOG = (
    "stages (blocks): ((0, 5), (5, 10), (10, 15), (15, 20), "
    "(20, 25), (25, 30), (30, 35), (35, 40))\n"
    "measured: MeasuredStages(cuts=(5, 10, 15, 20, 25, 30, 35), "
    "seconds=(9.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 5.0), peak_gib=(40.0,), "
    "bytes_per_param=(4.0,), limit_gib=74.6)\n"
    "step 1 loss 1.0\n"
    "stages (blocks): ((0, 1), (1, 8), (8, 15), (15, 22), "
    "(22, 29), (29, 36), (36, 39), (39, 40))\n"
)


def test_reads_the_last_pipeline_and_the_calibration():
    assert lfl.last_stage_blocks(LOG)[0] == (0, 1)
    m = lfl.measured_stages(LOG)
    assert m["cuts"] == (5, 10, 15, 20, 25, 30, 35) and m["seconds"][0] == 9.0


def test_megatron_cut_is_the_nearest_with_equal_middle_stages():
    # rdsp's cut already expressible (first 1, last 3, six middle stages of 6): kept
    assert lfl.megatron_cut([1, 6, 6, 6, 6, 6, 6, 3], 40) == (1, 3)
    # one middle stage of 7 and one of 5: the equal-middle cut 2 layers away
    assert lfl.megatron_cut([2, 6, 6, 7, 6, 6, 5, 2], 40) == (2, 2)
    # two stages: any split is expressible
    assert lfl.megatron_cut([3, 7], 10) == (3, 7)


def test_vision_width_balances_vision_and_language_gpu_time():
    # middle stages: 2.5 s per 5 blocks; stage 0: 9 s - 2.5 s = 6.5 s of vision on 2 GPUs
    # = 13 GPU-s; language: (9 + 6 x 2.5 + 5) x 2 - 13 = 45 GPU-s.
    # k=2: max(6.5, 3.2) = 6.5; k=4: max(3.25, 3.75) = 3.75; k=8: max(1.6, 5.6) = 5.6
    m = lfl.measured_stages(LOG)
    assert lfl.vision_width(m["cuts"], m["seconds"], 40, 2, 16) == 4
    # little vision: the narrowest stage
    assert lfl.vision_width(m["cuts"], (2.6,) + m["seconds"][1:], 40, 2, 16) == 2
