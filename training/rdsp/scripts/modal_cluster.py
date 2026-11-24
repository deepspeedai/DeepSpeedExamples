# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Multi-node Modal harness: N clustered GPU containers joined into ONE Ray
cluster; pytest runs on the rank-0 container as the Ray driver.

From the repo root:

    modal run scripts/modal_cluster.py \
        --tests "tests/integration/test_heterogeneous_pipeline.py -k four-stage-mixed"
    RDSP_CLUSTER_SHAPE=l4x4x2 modal run scripts/modal_cluster.py --tests "... -k mini"

Each container runs this function (broadcast). Rank 0 starts the Ray head,
waits until every node has joined, runs pytest with RDSP_RAY_ADDRESS set
(the test fixtures then attach instead of starting a local Ray), and shuts
the cluster down. Other ranks join as Ray workers and exit when the head
goes away. Checkpoint directories are node-local: each stage's shards live
on the node that hosts it (a stage never spans nodes).
"""

import os
import sys
from pathlib import Path

import modal
import modal.experimental

# modal_tests sits next to this file locally, and under the mounted repo in
# the containers.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/root/rdsp/scripts"]
from modal_tests import VOLUMES, hf_cache, image

app = modal.App("rdsp-cluster", image=image)

RAY_PORT = 6379


# Modal validates every function of an app at deploy time against the
# workspace's GPU limit, so only the requested cluster shape is defined:
#   RDSP_CLUSTER_SHAPE=h100x8x4  (default) 4 nodes x 8 H100 = 32 GPUs
#   RDSP_CLUSTER_SHAPE=l4x4x2    2 nodes x 4 L4  =  8 GPUs (fits a 10-GPU limit)
SHAPE = os.environ.get("RDSP_CLUSTER_SHAPE", "h100x8x4")
_SHAPES = {"h100x8x4": ("H100:8", 4, 8, True), "l4x4x2": ("L4:4", 2, 4, False)}
_GPU, _NODES, _PER_NODE, _RDMA = _SHAPES[SHAPE]


@app.function(gpu=_GPU, timeout=5400, volumes=VOLUMES)
@modal.experimental.clustered(size=_NODES, rdma=_RDMA)
def run_cluster(tests: str, nodes: int, per_node: int):
    # Counts arrive as arguments: the env var that picked the shape exists
    # only on the launching machine, not inside the containers.
    return _cluster_main(tests, expected_nodes=nodes, gpus_per_node=per_node)


def _cluster_main(tests: str, expected_nodes: int, gpus_per_node: int):
    import os
    import shlex
    import subprocess
    import sys
    import time

    info = modal.experimental.get_cluster_info()
    rank, ips = info.rank, info.container_ips
    head = ips[0]
    subprocess.run(["pip", "install", "-e", "/root/rdsp"], check=True, capture_output=True)
    env = dict(os.environ, RAY_DEDUP_LOGS="0")

    if rank != 0:
        subprocess.run(
            [
                "ray",
                "start",
                f"--address={head}:{RAY_PORT}",
                f"--node-ip-address={ips[rank]}",
                f"--num-gpus={gpus_per_node}",
            ],
            check=True,
            env=env,
        )
        while (
            subprocess.run(
                ["ray", "status", f"--address={head}:{RAY_PORT}"], capture_output=True
            ).returncode
            == 0
        ):
            time.sleep(15)
        return 0, f"worker {rank} done"

    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            f"--port={RAY_PORT}",
            f"--node-ip-address={head}",
            f"--num-gpus={gpus_per_node}",
            "--include-dashboard=false",
        ],
        check=True,
        env=env,
    )
    import ray

    ray.init(address=f"{head}:{RAY_PORT}")
    deadline = time.time() + 600
    while len([n for n in ray.nodes() if n["Alive"]]) < expected_nodes:
        if time.time() > deadline:
            raise RuntimeError(f"only {len(ray.nodes())} of {expected_nodes} nodes joined")
        time.sleep(5)
    print(f"cluster up: {ray.cluster_resources()}", flush=True)
    ray.shutdown()

    env["RDSP_RAY_ADDRESS"] = f"{head}:{RAY_PORT}"
    env["RDSP_CLUSTER_GPUS"] = str(expected_nodes * gpus_per_node)
    proc = subprocess.Popen(
        [
            "python",
            "-u",
            "-m",
            "pytest",
            "-v",
            "-s",
            "--tb=short",
            "-rA",
            "-o",
            "faulthandler_timeout=1200",
            *shlex.split(tests),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/root/rdsp",
        env=env,
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    subprocess.run(["ray", "stop", "--force"])
    hf_cache.commit()
    return proc.returncode, "".join(lines)[-40000:]


@app.local_entrypoint()
def main(tests: str):
    print(f"cluster shape {SHAPE}: {_NODES} nodes x {_GPU}")
    code, output = run_cluster.remote(tests, _NODES, _PER_NODE)
    print(output)
    print(f"\nexit code: {code}")
