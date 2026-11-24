# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""medians.py <bridge|titan|rdsp> <log>...: median seconds per step over steps 6-50 of each log
(one line per log: "<seconds> <log>", or nothing when fewer than 40 steps ran), and with --best
only the log with the lowest median. MEDIAN_FIRST (default 6) and MEDIAN_MIN (default 40): the
first step counted and the fewest steps a median needs (the 10-step receipt control: 4 and 5).

  bridge  Megatron-Bridge: "iteration N/ M | ... elapsed time per iteration (ms): X"
  titan   TorchTitan: the time between consecutive "step: N" lines of one rank
  rdsp    rdsp: "step N loss L  T ms" lines
"""

import os
import re
import statistics
import sys


def bridge(text: str) -> dict[int, float]:
    pattern = r"iteration +(\d+)/ *\d+ \|.*?elapsed time per iteration \(ms\): ([\d.]+)"
    times = {int(m.group(1)): float(m.group(2)) / 1000 for m in re.finditer(pattern, text)}
    if (
        not times
    ):  # two-node logs of node 1 print only "Step Time : X s", one line per step from step 1
        times = {
            i: float(t) for i, t in enumerate(re.findall(r"^Step Time : ([\d.]+)s", text, re.M), 1)
        }
    return times


def titan(text: str) -> dict[int, float]:
    stamps: dict[int, float] = {}
    pattern = (
        r"\[rank0\].*?(\d\d):(\d\d):(\d\d),(\d+) - torchtitan\.observability\.metrics - INFO - "
        r".*?step: +(\d+) "
    )
    for m in re.finditer(pattern, re.sub(r"\x1b\[[0-9;]*m", "", text)):
        h, mi, s, ms, step = m.groups()
        stamps[int(step)] = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000
    return {k: stamps[k] - stamps[k - 1] for k in stamps if k - 1 in stamps}


def rdsp(text: str) -> dict[int, float]:
    return {
        int(m.group(1)): float(m.group(2)) / 1000
        for m in re.finditer(r"^step (\d+) loss \S+ +(\d+) ms", text, re.M)
    }


def median(kind: str, path: str) -> float | None:
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return None
    times = globals()[kind](text)
    first = int(os.environ.get("MEDIAN_FIRST", "6"))
    window = [times[i] for i in range(first, 51) if i in times]
    return (
        statistics.median(window)
        if len(window) >= int(os.environ.get("MEDIAN_MIN", "40"))
        else None
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    best = "--best" in args
    args = [a for a in args if a != "--best"]
    kind, logs = args[0], args[1:]
    found = [(m, log) for log in logs if (m := median(kind, log)) is not None]
    if best and found:
        found = [min(found)]
    for m, log in found:
        print(f"{m:.2f} {log}")
