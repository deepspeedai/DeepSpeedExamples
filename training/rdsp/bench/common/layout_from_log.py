# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Layout choices for later two-node runs, read from rdsp's run 1 log (TP1 x PP8 x DP2, --cuts
measured), so neither is picked by hand:

    python3 layout_from_log.py megatron-cut <rdsp log> <layers>
        "F L": layers on Megatron's first and last pipeline stage
        (model.num_layers_in_first/last_pipeline_stage), the middle stages equal, nearest to
        the cut rdsp measured and chose (fewest layers moved).
    python3 layout_from_log.py vision-width <rdsp log> <GPUs> [language GPUs per stage]
        k: GPUs for a vision-only first stage (--cuts 0,measured), from {2, 4, 8}, the language
        on the other GPUs as pipeline stages of 2 (or the given size, which must divide both
        GPUs - k and k); the k that minimizes the slower of vision time / k and language time /
        (GPUs - k), both in GPU-seconds per step as run 1 measured.

The log lines read: "stages (blocks): ((0, 3), (3, 9), ...)" (the last one is the measured
cut's pipeline) and "measured: MeasuredStages(cuts=(...), seconds=(...), ...)" (the
calibration steps, which ran the balanced cut).
"""

import ast
import re
import sys


def last_stage_blocks(text: str) -> list[tuple[int, int]]:
    found = re.findall(r"^stages \(blocks\): (.*)$", text, re.M)
    if not found:
        raise ValueError("no 'stages (blocks):' line")
    return [tuple(b) for b in ast.literal_eval(found[-1])]


def measured_stages(text: str) -> dict:
    found = re.findall(r"^measured: MeasuredStages\((.*)\)$", text, re.M)
    if not found:
        raise ValueError("no 'measured: MeasuredStages(...)' line")
    fields = {}
    for name in ("cuts", "seconds"):
        m = re.search(name + r"=(\([^)]*\))", found[-1])
        fields[name] = tuple(ast.literal_eval(m.group(1)))
    return fields


def megatron_cut(blocks: list[int], layers: int) -> tuple[int, int]:
    """(first, last): the cut Megatron can express (middle stages equal, every stage at least
    one layer) with the fewest layers away from `blocks`; on a tie, first and last closest."""
    stages = len(blocks)
    best = None
    for first in range(1, layers):
        for last in range(1, layers - first + 1):
            rest = layers - first - last
            if stages > 2 and (rest % (stages - 2) or rest == 0):
                continue
            if stages == 2 and rest:
                continue
            middle = rest // (stages - 2) if stages > 2 else 0
            cut = [first] + [middle] * (stages - 2) + [last]
            moved = sum(abs(a - b) for a, b in zip(cut, blocks))
            key = (moved, abs(first - last))
            if best is None or key < best[0]:
                best = (key, (first, last))
    return best[1]


def vision_width(
    cuts: tuple,
    seconds: tuple,
    layers: int,
    gpus_per_stage: int,
    gpus: int,
    widths=(2, 4, 8),
    stage_gpus: int | None = None,
) -> int:
    """The vision-only stage's GPU count. Run 1's middle stages hold decoder blocks only:
    their seconds per block is the block rate; stage 0's time beyond its blocks at that rate
    is the vision encoder's (and the embedding's, which is small)."""
    bounds = [0, *cuts, layers]
    blocks = [b - a for a, b in zip(bounds, bounds[1:])]
    middle = range(1, len(blocks) - 1)
    rate = sum(seconds[s] for s in middle) / sum(blocks[s] for s in middle)
    vision = max(0.0, seconds[0] - blocks[0] * rate) * gpus_per_stage
    total = sum(seconds) * gpus_per_stage
    language = total - vision

    def step(k):
        return max(vision / k, language / (gpus - k))

    size = stage_gpus or gpus_per_stage  # the new layout's language stages
    return min((k for k in widths if (gpus - k) % size == 0 and k % size == 0), key=step)


def main(argv):
    what, log = argv[1], argv[2]
    text = open(log).read()
    if what == "megatron-cut":
        blocks = [b - a for a, b in last_stage_blocks(text)]
        print(*megatron_cut(blocks, int(argv[3])))
    elif what == "vision-width":
        m = measured_stages(text)
        layers = last_stage_blocks(text)[-1][1]
        stage_gpus = int(argv[4]) if len(argv) > 4 else None
        print(vision_width(m["cuts"], m["seconds"], layers, 2, int(argv[3]), stage_gpus=stage_gpus))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
