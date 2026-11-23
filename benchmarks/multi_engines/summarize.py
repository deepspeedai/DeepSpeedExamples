# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Summarize run_checks.sh logs: one line per run, compared with the outcome each case is expected to show.

    python summarize.py <log_dir>
"""
import collections
import json
import os
import re
import sys

# What the proposed design needs from each case. "pass" cases must pass; "hazard" cases are expected to fail
# (hang or wrong values) and are there to show why the design avoids that pattern.
EXPECT = {
    "registry_path": "pass",
    "destroy_same_order": "pass",
    "memory_cycles": "pass",
    "overlap_members_only": "hazard",
    "subset_then_all": "hazard",
    "split_then_all": "hazard (check that group names match across ranks)",
    "destroy_diff_order": "hazard",
    "shrink_then_all": "unknown",
    "timeout_probe": "observe",
}


def records(path):
    out = []
    for line in open(path, errors="replace"):
        for piece in line.split("MER ")[1:]:
            m = re.match(r"(\{.*?\})(?=MER |\s*$)", piece.strip()) or re.match(r"(\{.*\})", piece.strip())
            if m:
                try:
                    out.append(json.loads(m.group(1)))
                except ValueError:
                    pass
    return out


def summarize_run(path):
    text = open(path, errors="replace").read()
    R = records(path)
    rc = (re.findall(r"^rc=(\d+)", text, re.M) or ["?"])[-1]
    init = next((r for r in R if r.get("mark") == "init"), {})
    case = init.get("case") or next((r.get("case") for r in R), "?")
    skipped = [r for r in R if r.get("mark") == "skipped"]
    summaries = {r["rank"]: r["result"] for r in R if r.get("mark") == "summary"}
    wrong = sorted({(r["mark"], tuple(r["got"]), tuple(r["expected"])) for r in R if r.get("ok") is False})
    hung = sorted({r["rank"] for r in R if r.get("mark") == "hang_timeout"})
    errors = sorted(set(re.findall(r"^\[rank\d+\]: ((?:\w+\.)*\w+(?:Error|Exception): [^\n]{0,110})", text, re.M)))
    names = collections.defaultdict(set)
    for r in R:
        if r.get("mark", "").startswith("created ") and r.get("member"):
            names[(r["mark"], tuple(r.get("ranks") or ()))].add(r.get("name"))  # one group = same label and ranks
    diverged = sorted(f"{m[len('created '):]} {list(rk)}" for (m, rk), v in names.items() if len(v) > 1)
    if skipped:
        verdict = "SKIPPED: " + skipped[0].get("reason", "")
    elif case == "timeout_probe":
        h = next((r for r in R if r.get("mark") == "healthy" and r["rank"] == 0), {})
        call = next((r for r in R if r.get("mark", "").startswith("timed-out call")), {})
        alive = [r["since_call_s"] for r in R if r.get("mark") == "alive"]
        verdict = (f"healthy error state {h.get('error_state')}, query {h.get('error_query_us')} us; "
                   f"{call.get('mark')} after {call.get('waited_s')} s with value {call.get('value')} "
                   f"(expected 2.0), error state {call.get('error_state')}; "
                   f"last seen alive {alive[-1] if alive else 0} s after the call")
    elif len(summaries) == int(init.get("world_size", 4) or 4) and all(v == "ALL_OK" for v in summaries.values()) \
            and not hung:
        verdict = "PASS"
    else:
        parts = []
        if wrong:
            parts.append("wrong values " + "; ".join(f"{m}: got {list(g)} expected {list(e)}" for m, g, e in wrong[:3]))
        if hung:
            parts.append(f"hung ranks {hung}")
        if errors:
            parts.append("raised " + errors[0])
        if not parts:
            parts.append(f"incomplete (rc={rc}, summaries from ranks {sorted(summaries)})")
        verdict = "FAIL: " + ", ".join(parts)
    if diverged:
        verdict += f"; group names differ across ranks for {diverged}"
    return case, init, verdict


def main(log_dir):
    rows = []
    for f in sorted(os.listdir(log_dir)):
        if f.endswith(".log"):
            case, init, verdict = summarize_run(os.path.join(log_dir, f))
            rows.append((f[:-4], case, init, verdict))
    for name, case, init, verdict in rows:
        impl = init.get("backend_impl") or init.get("backend")
        print(f"{name}\n    torch {init.get('torch')}, {init.get('backend')} ({impl}), eager={init.get('eager')}, "
              f"device {init.get('device')}\n    expected: {EXPECT.get(case, '?')}\n    result:   {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs")
