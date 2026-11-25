# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Driver-side handles to the stage actor groups, and their start-up.

create_stage_clients() places each stage on one node, starts one
StageWorkerActor per GPU, forms every stage's DeepSpeed world and opens the
cross-stage p2p groups. The coordinator talks to each stage through a
StageGroupClient; Ray ObjectRefs never leave this module unresolved.
"""

import os

import torch

from ray_deepspeed_pipeline.boundary import (
    Grid,
    cell_slice,
    rank_cell,
    rank_coords,
    representatives,
    slice_inputs,
)
from ray_deepspeed_pipeline.errors import CheckpointError, ValidationError
from ray_deepspeed_pipeline.losses import TokenMeanLoss
from ray_deepspeed_pipeline.partition import compile_blocks, recompute_blocks, select_stage_builder
from ray_deepspeed_pipeline.protocols import Command
from ray_deepspeed_pipeline.stage_worker import StageWorkerActor
from ray_deepspeed_pipeline.vision import VisionLayout, build_vision_tower, route_images

_PLACEMENT_TIMEOUT_S = float(os.environ.get("RDSP_PLACEMENT_TIMEOUT_S", "600"))


def _p2p_timeout_s() -> float:
    """How long a stage waits on a neighbour before the step fails."""
    return float(os.environ.get("RDSP_P2P_TIMEOUT_S", "300"))


def _alive_timeout_s() -> float:
    return float(os.environ.get("RDSP_ALIVE_TIMEOUT_S", "60"))


class _GroupHandle:
    """Future-like handle for protocols.resolve(): resolves every rank of a
    stage and reduces to one stage-level value. The raw ObjectRefs never leave
    the runtime layer."""

    def __init__(self, refs, reduce):
        self.refs = refs
        self.reduce = reduce

    def result(self):
        import ray

        return self.reduce(ray.get(self.refs))


def _all_ok(results):
    """True if every rank succeeded; a rank's rejection text wins."""
    for result in results:
        if isinstance(result, str):
            return result
    return all(result is not False for result in results)


def _mean(values):
    return sum(float(v) for v in values) / len(values)


class StageGroupClient:
    """Driver-side client for one stage's actors. Every command goes to every
    rank in the same order, so the ranks' collectives line up."""

    def __init__(
        self,
        actors,
        stage: int = 0,
        grid: Grid | None = None,
        is_last: bool = False,
        placement_group=None,
        vision=None,
        count_tokens=None,
    ):
        """vision: (VisionLayout, this stage's global rank offset) under
        colocated vision. count_tokens: a TokenMeanLoss's count_fn (last
        stage), which counts the step's tokens from all its labels."""
        self.actors = actors
        self.count_tokens = count_tokens
        self.vision = vision
        self.stage = stage
        self.grid = grid or Grid(dp=len(actors))
        self.is_last = is_last
        self.placement_group = placement_group
        self.node_id = None  # set at bootstrap; rebuilds prefer this node

    def submit(self, command: Command, *, inputs=None, labels=None, control=None, prepared=None):
        """Send a command to every rank; returns a handle resolving to one
        stage-level result. prepared: prepare()'s result, in place of
        inputs and labels."""
        if command.kind in ("step", "eval_step"):
            payloads = prepared if prepared is not None else self._rank_payloads(inputs, labels)
            return self._submit_step(command, payloads, control)
        refs = [
            actor.execute.remote(
                command.command_id, command.kind, generation=command.generation, control=control
            )
            for actor in self.actors
        ]
        if command.kind == "save":
            return _GroupHandle(refs, self._stage_record)
        return _GroupHandle(refs, _all_ok)

    def prepare(self, inputs=None, labels=None):
        """A step's per-rank payloads, already in Ray's object store, so a
        later submit() sends only references. Safe to call from another
        thread while a step runs."""
        import ray

        return [
            {k: v if v is None or k == "token_total" else ray.put(v) for k, v in payload.items()}
            for payload in self._rank_payloads(inputs, labels)
        ]

    def _rank_payloads(self, inputs, labels) -> list[dict]:
        """Each rank's share of a step. Under colocated vision every stage
        gets the inputs: each rank takes its images, and the first stage the
        rest without the pixels."""
        token_total = None
        if self.count_tokens is not None and labels is not None:
            token_total = sum(int(self.count_tokens(y)) for y in labels)
        payloads = []
        for rank in range(len(self.actors)):
            cell = rank_cell(self.grid, rank)
            routes, cell_inputs = None, None
            if self.vision is not None and inputs is not None:
                layout, offset = self.vision
                routes = route_images(inputs, layout, offset + rank)
                if self.stage == 0:
                    cell_inputs = [
                        slice_inputs({k: v for k, v in x.items() if k != "pixel_values"}, cell)
                        for x in inputs
                    ]
            elif inputs is not None:
                cell_inputs = [slice_inputs(x, cell) for x in inputs]
            payloads.append(
                {
                    "inputs": cell_inputs,
                    "labels": [cell_slice(y, cell) for y in labels] if labels is not None else None,
                    "vision": routes,
                    "token_total": token_total,
                }
            )
        return payloads

    def _submit_step(self, command, payloads, control):
        """One call per rank carrying the rank's whole op list for the step."""
        refs = [
            actor.run_step.remote(
                command.generation,
                control["ops"],
                command.kind == "step",
                varying=control.get("varying", False),
                **payload,
            )
            for actor, payload in zip(self.actors, payloads)
        ]
        reps = representatives(self.grid)

        def reduce(results):
            out = {"ready": all(res["ready"] for res in results)}
            if self.is_last:  # per microbatch: mean over the data-parallel cells
                per_rep = [results[r]["losses"] for r in reps]
                out["losses"] = [_mean(v) for v in zip(*per_rep)]
            return out

        return _GroupHandle(refs, reduce)

    def init_p2p(self, host: str, port: int, epoch: int, grids: list):
        return [
            actor.init_p2p.remote(host, port, epoch, grids, _p2p_timeout_s())
            for actor in self.actors
        ]

    @staticmethod
    def reconnect_p2p(clients, epoch: int) -> None:
        connect_p2p(clients, epoch)

    def _stage_record(self, rank_records):
        apart = [r["rank"] for r in rank_records if not r.get("shares_dir", True)]
        if apart:
            raise CheckpointError(
                f"stage {self.stage}: rank {apart[0]} does not see rank 0's files in the "
                f"checkpoint directory; a stage that spans nodes needs a path on shared storage"
            )
        files = next(r["files"] for r in rank_records if r["rank"] == 0)
        return {
            "index": self.stage,
            "world_size": len(self.actors),
            "ranks": [{"rank": r["rank"]} for r in rank_records],
            "files": files,
        }

    def alive(self, timeout: float | None = None) -> bool:
        import ray

        timeout = _alive_timeout_s() if timeout is None else timeout
        try:
            ray.get([actor.ping.remote() for actor in self.actors], timeout=timeout)
            return True
        except Exception:
            return False

    def set_profile(self, on: bool) -> None:
        import ray

        ray.get([actor.set_profile.remote(on) for actor in self.actors])

    def profile_records(self) -> list[dict]:
        """Each rank's StageWorkerActor.profile_records(), in rank order."""
        import ray

        return ray.get([actor.profile_records.remote() for actor in self.actors])

    def shutdown(self) -> None:
        """Best effort: kill the actors and release the placement group."""
        import ray

        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        if self.placement_group is not None:
            try:
                ray.util.remove_placement_group(self.placement_group)
            except Exception:
                pass


def check_mesh(stage: int, grid: Grid, observed: list[dict | None]) -> None:
    """Check that DeepSpeed placed every rank where the plan's grid says, with
    each cell covered exactly once. Ranks reporting None (non-DeepSpeed stub
    engines) are checked only for count."""
    if len(observed) != grid.world:
        raise ValidationError(
            f"stage {stage}: {len(observed)} ranks bootstrapped, grid needs {grid.world}"
        )
    seen = set()
    for rank, mesh in enumerate(observed):
        if mesh is None:
            continue
        d, q, t = rank_coords(grid, rank)
        expected = {
            "dp_rank": d,
            "dp_world": grid.dp,
            "tp_rank": t,
            "tp_world": grid.tp,
            "sp_rank": q,
            "sp_world": grid.sp,
        }
        for key, want in expected.items():
            got = mesh.get(key)
            if got is not None and got != want:
                raise ValidationError(
                    f"stage {stage} rank {rank}: DeepSpeed places it at "
                    f"{key}={got}, the plan's grid expects {want} (mesh mismatch)"
                )
        coords = (mesh.get("dp_rank", d), mesh.get("sp_rank", q), mesh.get("tp_rank", t))
        if coords in seen:
            raise ValidationError(f"stage {stage}: duplicate rank at grid position {coords}")
        seen.add(coords)


def connect_p2p(clients, epoch: int) -> None:
    """(Re)build the cross-stage groups on every rank of every stage. The
    TCPStore lives in stage 0 rank 0; each epoch uses fresh groups under a new
    key prefix, because a failed step leaves the old ones unusable."""
    import ray

    if epoch == 0:
        host, port, _ = ray.get(clients[0].actors[0].address.remote())
        for client in clients:
            client.p2p_address = (host, port)
    host, port = clients[0].p2p_address
    grids = [(client.grid.dp, client.grid.sp, client.grid.tp) for client in clients]
    ray.get([ref for client in clients for ref in client.init_p2p(host, port, epoch, grids)])


def _builder_kwargs(builder, spec) -> dict:
    """The HF stage builder also takes the stage's index: after a
    vision-only stage, the next one starts at block 0 too."""
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage

    return {"stage_index": spec.index} if builder is build_hf_stage else {}


def fill_nodes_in_order(stage_gpus: list[int], nodes: list[tuple[str, int]]) -> list | None:
    """A node for each stage: consecutive stages fill the first node, then
    the next, so the pipeline crosses between nodes as few times as it can
    (each crossing sends every microbatch's boundary over the network). nodes:
    (node id, GPUs) in the order to fill. None if a stage fits no node."""
    out, i, free = [], 0, nodes[0][1] if nodes else 0
    for need in stage_gpus:
        while i < len(nodes) and need > free:
            i += 1
            free = nodes[i][1] if i < len(nodes) else 0
        if i == len(nodes):
            return None
        out.append(nodes[i][0])
        free -= need
    return out


def bundles_in_rank_order(bundles_to_node: dict, node_order: list, group_size: int) -> list[int]:
    """Placement-group bundle for each rank of a stage that spans nodes: the
    bundles grouped by node (in node_order), so consecutive ranks share a
    node. A stage's ranks are tensor-parallel fastest, so a tensor-parallel
    (x sequence-parallel) group of group_size consecutive ranks then stays on
    one node, provided each node holds a multiple of group_size of the stage's
    GPUs."""
    where = {node: i for i, node in enumerate(node_order)}
    order = sorted(bundles_to_node, key=lambda b: (where.get(bundles_to_node[b], len(where)), b))
    per_node = {}
    for b in order:
        per_node[bundles_to_node[b]] = per_node.get(bundles_to_node[b], 0) + 1
    if any(n % group_size for n in per_node.values()):
        raise ValidationError(
            f"a stage spanning nodes got {sorted(per_node.values())} GPUs per node; each node "
            f"must hold a multiple of tp x sp = {group_size} so no tensor-parallel group "
            f"crosses nodes; use a stage width that splits evenly over the nodes"
        )
    return order


def _nodes_in_fill_order() -> list[tuple[str, int]]:
    """Alive nodes with GPUs: the driver's first, the rest by address."""
    import ray

    here = ray.get_runtime_context().get_node_id()
    nodes = [
        (n["NodeID"], int(n["Resources"].get("GPU", 0)), n["NodeManagerAddress"])
        for n in ray.nodes()
        if n["Alive"] and n["Resources"].get("GPU", 0) >= 1
    ]
    nodes.sort(key=lambda n: (n[0] != here, n[2]))
    return [(node, gpus) for node, gpus, _ in nodes]


def create_stage_clients(
    model,
    plan,
    loss_fn,
    *,
    engine_factory=None,
    use_gpu: bool | None = None,
    stage_builder=None,
    prefer_nodes: list | None = None,
    weights: str | None = None,
):
    """Start one actor group per plan stage and return their clients.

    Needs an attached Ray context. Each stage gets its own STRICT_PACK
    placement group (all its GPUs on one node, so TP/EP/SP collectives stay on
    the node's interconnect and its checkpoint shards land on one disk), its
    own rendezvous and its own stage-local world. A stage that needs more GPUs
    than a node has spans nodes instead (PACK), its ranks ordered so its
    tensor-parallel groups stay inside a node. prefer_nodes: per-stage node
    ids to reuse on a rebuild, so node-local shards stay reachable after an
    actor (not node) failure. Without them, on several nodes, stages fill the
    driver's node first, then the next (fill_nodes_in_order). weights: HF
    checkpoint dir each stage loads its meta-device parameters from. On any
    failure every started stage is shut down.
    """
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if not ray.is_initialized():
        raise ValidationError(
            "no Ray context: attach with ray.init(...) before rdsp.initialize() "
            "(the package uses the current context and never creates one)"
        )
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    needed = sum(spec.num_gpus for spec in plan.stages)
    available = int(ray.cluster_resources().get("GPU", 0))
    if use_gpu and needed > available:  # placement would wait out its timeout
        raise ValidationError(
            f"the pipeline needs {needed} GPUs; this Ray cluster has {available}. "
            f"Reduce gpus= in the stage layouts, or add GPUs"
        )
    backend = "nccl" if use_gpu else "gloo"
    actor_cls = ray.remote(StageWorkerActor)
    nodes = _nodes_in_fill_order() if use_gpu else []
    max_node_gpus = max((gpus for _, gpus in nodes), default=0)
    if use_gpu and prefer_nodes is None and len(nodes) > 1:
        prefer_nodes = fill_nodes_in_order([s.num_gpus for s in plan.stages], nodes)

    clients = []
    n_stages = len(plan.stages)
    # chosen from the model itself, so rdsp.initialize() (which passes no
    # builder) and recovery rebuilds get the right one too
    builder = stage_builder or select_stage_builder(model)
    vision, layout, offset = plan.colocated_vision, None, 0
    if vision is not None:
        first = plan.stages[0]
        layout = VisionLayout(
            first=Grid(dp=first.dp, sp=first.sp, tp=first.tp),
            world=sum(spec.num_gpus for spec in plan.stages),
        )
        tower_ref = ray.put((build_vision_tower(model, vision.module), vision))
    try:
        for spec in plan.stages:
            grid = Grid(dp=spec.dp, sp=spec.sp, tp=spec.tp)
            module = builder(
                model,
                spec.block_start,
                spec.block_stop,
                spec.parameter_names,
                **_builder_kwargs(builder, spec),
            )
            if spec.recompute or spec.recompute_blocks is not None:
                recompute_blocks(module, None if spec.recompute else spec.recompute_blocks)
            if spec.compile or spec.compile_vision:
                compile_blocks(module, encoder_only=not spec.compile)
            is_last = spec.index == n_stages - 1
            pg = None
            options = {"num_gpus": 1 if use_gpu else 0}
            bundle_of_rank = list(range(spec.num_gpus))
            if use_gpu:
                spans = spec.num_gpus > max_node_gpus
                strategy = "PACK" if spans else "STRICT_PACK"
                target = prefer_nodes[spec.index] if prefer_nodes and not spans else None
                pg = placement_group(
                    [{"GPU": 1, "CPU": 1}] * spec.num_gpus,
                    strategy=strategy,
                    _soft_target_node_id=target,
                )
                try:
                    ray.get(pg.ready(), timeout=_PLACEMENT_TIMEOUT_S)
                except Exception as e:
                    ray.util.remove_placement_group(pg)
                    raise ValidationError(
                        f"stage {spec.index}: cannot place {spec.num_gpus} GPUs "
                        f"({strategy}) in this Ray cluster: {e}"
                    ) from e
                if spans:
                    table = ray.util.placement_group_table(pg)["bundles_to_node_id"]
                    try:
                        bundle_of_rank = bundles_in_rank_order(
                            table, [node for node, _ in nodes], spec.tp * spec.sp
                        )
                    except ValidationError:
                        ray.util.remove_placement_group(pg)
                        raise
            module_ref = ray.put(module)
            actors = []
            for rank in range(spec.num_gpus):
                rank_options = dict(options)
                if pg is not None:
                    rank_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=bundle_of_rank[rank]
                    )
                actors.append(
                    actor_cls.options(**rank_options).remote(
                        module_ref,
                        spec.ds_config_json,
                        stage=spec.index,
                        rank=rank,
                        grid=grid,
                        backend=backend,
                        n_microbatches=plan.global_microbatches,
                        is_first=spec.index == 0,
                        is_last=is_last,
                        loss_fn=loss_fn if is_last else None,
                        engine_factory=engine_factory,
                        weights=weights,
                        vision=tower_ref if vision is not None else None,
                    )
                )
            clients.append(
                StageGroupClient(
                    actors,
                    stage=spec.index,
                    grid=grid,
                    is_last=is_last,
                    placement_group=pg,
                    vision=(layout, offset) if vision is not None else None,
                    count_tokens=(
                        loss_fn.count_fn if is_last and isinstance(loss_fn, TokenMeanLoss) else None
                    ),
                )
            )
            offset += spec.num_gpus

        # bootstrap every stage's world concurrently
        boots = []
        for client in clients:
            addr, port, client.node_id = ray.get(client.actors[0].address.remote())
            boots.append([actor.bootstrap.remote(addr, port) for actor in client.actors])
        for client, refs in zip(clients, boots):
            check_mesh(client.stage, client.grid, ray.get(refs))
        connect_p2p(clients, epoch=0)
    except Exception:
        for client in clients:
            client.shutdown()
        raise
    return clients
