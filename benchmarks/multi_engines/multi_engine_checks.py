# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Process-group checks behind the multi-engine design discussed in DeepSpeed RFC #9081.

Run one case per process with torchrun (see README.md and run_checks.sh):

    CASE=registry_path torchrun --nproc_per_node=4 multi_engine_checks.py

Cases needing 4 ranks:
  registry_path            The proposed scheme: every rank creates every group of every engine in the same order,
                           each engine has its own root group, and release is deferred to the next all-ranks call.
                           Engines A (TP2 x DP2) and B (TP4), then DP4 -> DP2 -> DP4 with the DP2 engine on ranks 0-1.
  overlap_members_only     Overlapping groups X=[0,1,2], Y=[1,2,3] created only by their members
                           (use_local_synchronization=True), in one order.
  subset_then_all          A members-only group [0,1,2], then a group over all ranks created by all ranks.
  split_then_all           An all-ranks root [0,1,2], a split_group [[0,1],[2]] called only by its members,
                           then an all-ranks group. Compare group names across ranks.
  destroy_same_order       Two groups over [0,1] destroyed in the same order on both member ranks.
  destroy_diff_order       The same two groups destroyed in different orders on the two member ranks.
  shrink_then_all          shrink_group([3]) on a full group; rank 3 waits until the others finish. FOLLOW=1 then
                           creates an all-ranks group.
  memory_cycles            5 create/use/destroy cycles of a full-world group, reporting free device memory.
Cases needing 2 ranks:
  timeout_probe            Rank 1 never joins a collective. Records what the call returns, the backend's error state,
                           the cost of querying it, and how long the process lives afterwards.
"""
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.distributed as dist
from common import Checker, Env, emit, watchdog

CASE = os.environ.get("CASE", "registry_path")
NEEDS = {"timeout_probe": 2}


def main():
    pg_timeout = int(os.environ.get("PG_TIMEOUT", "15" if CASE == "timeout_probe" else "30"))
    timer = watchdog(int(os.environ.get("CASE_TIMEOUT", "150")), CASE)
    if CASE == "timeout_probe":
        timer.cancel()  # this case outlives CASE_TIMEOUT on purpose; run_checks.sh bounds it from outside
    env = Env(pg_timeout_s=pg_timeout)
    need = NEEDS.get(CASE, 4)
    if env.world_size != need:
        emit(CASE, mark="skipped", reason=f"needs {need} ranks, got {env.world_size}")
        return 0
    emit(CASE, mark="init", **env.info())
    fn = globals().get(f"case_{CASE}")
    if fn is None:
        emit(CASE, mark="skipped", reason="unknown case")
        return 2
    rc = fn(env, Checker(env, CASE))
    if CASE != "timeout_probe":
        dist.barrier()
        dist.destroy_process_group()
    return rc


class Registry:
    """Test stand-in for the proposed registry: all ranks create every group in the same order."""

    def __init__(self, env):
        self.env, self.groups, self.pending = env, [], []

    def declare(self, engine, layout):
        self.collect()
        mine = {}
        for name, rank_lists in layout:
            for ranks in rank_lists:
                h = dist.new_group(ranks)  # every rank, every group, same order
                self.groups.append((engine, tuple(ranks), h))
                if self.env.rank in ranks:
                    mine[name] = (h, list(ranks))
        return mine

    def destroy(self, engine):
        self.pending += [g for g in self.groups if g[0] == engine]
        self.groups = [g for g in self.groups if g[0] != engine]

    def collect(self):
        for _, ranks, h in self.pending:  # same creation order on every member rank
            if self.env.rank in ranks:
                dist.destroy_process_group(h)
        self.pending = []


def case_registry_path(env, ck):
    reg, world = Registry(env), list(range(env.world_size))

    def use(engine, groups):
        for name, (h, ranks) in groups.items():
            ck.group(f"{engine}.{name}", h, ranks)

    A = reg.declare("A", [("root", [world]), ("tp", [[0, 1], [2, 3]]), ("dp", [[0, 2], [1, 3]])])
    B = reg.declare("B", [("root", [world]), ("tp", [world])])
    for _ in range(3):  # engines used one after the other
        use("A", A)
        use("B", B)
    C = reg.declare("C", [("root", [world]), ("dp", [world])])
    use("C", C)
    reg.destroy("C")
    dist.barrier()
    before = env.free_mib()
    D = reg.declare("D", [("root", [[0, 1]]), ("dp", [[0, 1]])])  # releases C; D lives on ranks 0-1 only
    use("D", D)
    reg.destroy("D")
    E = reg.declare("E", [("root", [world]), ("dp", [world])])  # releases D
    use("E", E)
    use("A", A)
    use("B", B)
    dist.barrier()
    emit(CASE, mark="memory", free_mib_with_C=before, free_mib_with_E=env.free_mib())
    return ck.summary()


def case_overlap_members_only(env, ck):
    gs = {}
    for name, ranks in (("X", [0, 1, 2]), ("Y", [1, 2, 3])):
        if env.rank in ranks:
            gs[name] = (dist.new_group(ranks, use_local_synchronization=True), ranks)
            ck.created(name, gs[name][0], ranks)
    for name, (g, ranks) in gs.items():
        ck.group(name, g, ranks)
    return ck.summary()


def case_subset_then_all(env, ck):
    if env.rank in (0, 1, 2):
        g = dist.new_group([0, 1, 2], use_local_synchronization=True)
        ck.created("members_only", g, [0, 1, 2])
        ck.group("members_only", g, [0, 1, 2])
    world = list(range(env.world_size))
    g2 = dist.new_group(world)
    ck.created("all_ranks", g2, world)
    ck.group("all_ranks", g2, world)
    return ck.summary()


def case_split_then_all(env, ck):
    if not hasattr(dist, "split_group"):
        emit(CASE, mark="skipped", reason="torch has no split_group")
        return 0
    root = dist.new_group([0, 1, 2])
    ck.created("root", root, [0, 1, 2])
    if env.rank in (0, 1, 2):
        sub = dist.split_group(parent_pg=root, split_ranks=[[0, 1], [2]])
        mine = [0, 1] if env.rank < 2 else [2]
        ck.created("split", sub, mine)
        ck.group("split", sub, mine)
    world = list(range(env.world_size))
    g2 = dist.new_group(world)
    ck.created("all_ranks", g2, world)
    ck.group("all_ranks", g2, world)
    return ck.summary()


def _destroy(env, ck, different_order):
    X, Y = dist.new_group([0, 1]), dist.new_group([0, 1])
    if env.rank in (0, 1):
        ck.group("X", X, [0, 1])
        ck.group("Y", Y, [0, 1])
        order = [("X", X), ("Y", Y)]
        if different_order and env.rank == 1:
            order.reverse()
        for name, g in order:
            emit(CASE, mark=f"destroy {name} start")
            dist.destroy_process_group(g)
            emit(CASE, mark=f"destroy {name} done")
    ck.group("default_after", None, list(range(env.world_size)))
    return ck.summary()


def case_destroy_same_order(env, ck):
    return _destroy(env, ck, False)


def case_destroy_diff_order(env, ck):
    return _destroy(env, ck, True)


def case_shrink_then_all(env, ck):
    if not hasattr(dist, "shrink_group"):
        emit(CASE, mark="skipped", reason="torch has no shrink_group")
        return 0
    store = dist.distributed_c10d._get_default_store()
    world = list(range(env.world_size))
    g = dist.new_group(world)
    ck.group("full", g, world)
    if env.rank != 3:
        try:
            ng = dist.shrink_group([3], group=g)
        except Exception as e:  # noqa: BLE001
            emit(CASE, mark="shrink raised", detail=f"{type(e).__name__}: {str(e)[:160]}")
            store.set(f"shrink_done_{env.rank}", "raised")
            return 1
        ck.created("shrunk", ng, [0, 1, 2])
        ck.group("shrunk", ng, [0, 1, 2])
        store.set(f"shrink_done_{env.rank}", "1")
    else:
        store.wait([f"shrink_done_{k}" for k in (0, 1, 2)], datetime.timedelta(seconds=90))
        emit(CASE, mark="rank 3 saw the shrink finish")
    if os.environ.get("FOLLOW", "0") == "1":
        g2 = dist.new_group(world)
        ck.created("all_ranks_after", g2, world)
        ck.group("all_ranks_after", g2, world)
    return ck.summary()


def case_memory_cycles(env, ck):
    world = list(range(env.world_size))

    def snap(phase, k):
        env.sync()
        dist.barrier()
        time.sleep(2)
        emit(CASE, mark="memory", phase=phase, cycle=k, free_mib=env.free_mib(), reserved_mib=env.reserved_mib())

    snap("start", 0)
    for k in range(1, 6):
        g = dist.new_group(world)
        ck.group(f"cycle{k}", g, world)
        snap("alive", k)
        dist.destroy_process_group(g)
        time.sleep(3)
        snap("destroyed", k)
    return ck.summary()


def case_timeout_probe(env, ck):
    x = torch.ones(8, device=env.device)
    dist.all_reduce(x)
    env.sync()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        env.error_state()
    cost_us = (time.perf_counter() - t0) / n * 1e6
    emit(CASE, mark="healthy", error_state=env.error_state(), error_query_us=round(cost_us, 3))
    if env.rank == 1:
        time.sleep(200)
        os._exit(0)
    t0 = time.time()
    y = torch.ones(8, device=env.device)
    try:
        dist.all_reduce(y)
        env.sync()
        emit(CASE, mark="timed-out call returned", value=y[0].item(), expected=2.0,
             waited_s=round(time.time() - t0, 1), error_state=env.error_state())
    except Exception as e:  # noqa: BLE001
        emit(CASE, mark="timed-out call raised", detail=f"{type(e).__name__}: {str(e)[:160]}",
             waited_s=round(time.time() - t0, 1), error_state=env.error_state())
    for _ in range(24):  # up to 120 s after the call; torch normally aborts the process before that
        time.sleep(5)
        emit(CASE, mark="alive", since_call_s=round(time.time() - t0, 1), error_state=env.error_state())
    emit(CASE, mark="end", note="process still alive at the end", since_call_s=round(time.time() - t0, 1))
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
