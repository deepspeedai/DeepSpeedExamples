# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The one-step coordinator state machine, on fake workers."""

import pytest
from test_schedule import make_plan

from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.errors import PipelinePoisoned, StepFailed, ValidationError


class FakeStageWorker:
    """Synchronous stand-in for a stage actor group speaking the step protocol:
    a step carries the stage's op list and returns {"losses", "ready"}.
    Records every command and every op it was asked to run."""

    reconnects = 0

    def __init__(self, stage, is_terminal, fail_on=None, ready=True):
        self.stage = stage
        self.is_terminal = is_terminal
        self.fail_on = fail_on or set()  # command or op kinds that raise
        self.ready = ready
        self.commands = []
        self.ops = []
        self.controls = []
        self.fail_prepare = False
        self.saw_labels = False
        self.saw_inputs = False
        self.loss_calls = 0
        self.applies = 0

    @staticmethod
    def reconnect_p2p(clients, epoch):
        FakeStageWorker.reconnects += 1

    def prepare(self, inputs=None, labels=None):
        """A step's payload, made ready ahead of its dispatch."""
        if self.fail_prepare:
            raise ValueError("cannot split the rows")
        return (inputs, labels)

    def submit(self, command, *, inputs=None, labels=None, control=None, prepared=None):
        if prepared is not None:
            inputs, labels = prepared
        if command.kind != "apply":
            self.last_inputs = inputs
        self.commands.append(command)
        if command.kind in self.fail_on:
            raise RuntimeError(f"injected failure on {command.kind}")
        if command.kind == "apply":
            self.applies += 1
            return True
        assert command.kind in ("step", "eval_step"), command.kind
        ops = control["ops"]
        self.controls.append(control)
        self.ops.extend(ops)
        if any(kind in self.fail_on for kind, _, _ in ops):
            raise RuntimeError("injected failure inside the step")
        self.saw_inputs |= inputs is not None
        self.saw_labels |= labels is not None
        assert (inputs is not None) == (self.stage == 0), "only stage 0 gets inputs"
        assert (labels is not None) == self.is_terminal, "only the last stage gets labels"
        forwards = [mb for kind, mb, _ in ops if kind in ("forward", "eval")]
        losses = None
        if self.is_terminal:
            self.loss_calls += len(forwards)
            losses = [0.5 + mb for mb in forwards]
        return {"losses": losses, "ready": self.ready}


def build(stages=2, microbatches=4, fail_on=None):
    plan = make_plan(stages, microbatches)
    workers = [FakeStageWorker(s, s == stages - 1, (fail_on or {}).get(s)) for s in range(stages)]
    return PipelineCoordinator(plan, workers), workers


def test_one_call_one_global_step_and_mean_loss():
    coord, workers = build()
    data = [(f"x{k}", f"y{k}") for k in range(4)]
    loss = coord.train_batch(iter(data))
    assert loss == pytest.approx((0.5 + 1.5 + 2.5 + 3.5) / 4)
    assert coord.global_steps == 1
    assert all(w.applies == 1 for w in workers)


def test_only_terminal_stage_sees_labels_and_loss():
    coord, workers = build(stages=3, microbatches=6)
    coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(6)]))
    assert workers[0].saw_inputs and not workers[0].saw_labels
    assert not workers[1].saw_inputs and not workers[1].saw_labels
    assert workers[2].saw_labels and workers[2].loss_calls == 6


def test_eval_never_increments_global_step_or_applies():
    coord, workers = build()
    data = [(f"x{k}", f"y{k}") for k in range(4)]
    loss = coord.eval_batch(iter(data))
    assert loss == pytest.approx(2.0)
    assert coord.global_steps == 0
    assert all(w.applies == 0 for w in workers)
    assert all(c.kind == "eval_step" for w in workers for c in w.commands)
    assert all(kind == "eval" for w in workers for kind, _, _ in w.ops)


def test_exhausted_iterator_fails_before_any_dispatch():
    coord, workers = build()
    with pytest.raises(StepFailed):
        coord.train_batch(iter([("x0", "y0")]))
    assert coord.global_steps == 0
    assert all(w.commands == [] for w in workers), "no command may be issued when consumption fails"


def test_forward_failure_aborts_without_poison_and_is_retryable():
    coord, workers = build(fail_on={1: {"forward"}})
    with pytest.raises(StepFailed, match="stage 1"):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert coord.global_steps == 0
    assert all(w.applies == 0 for w in workers)
    # not poisoned: after the links are rebuilt, a fresh call succeeds
    workers[1].fail_on = set()
    reconnects = FakeStageWorker.reconnects
    assert coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)])) > 0
    assert FakeStageWorker.reconnects > reconnects
    assert coord.global_steps == 1


def test_stage_not_ready_aborts_before_apply():
    coord, workers = build()
    workers[0].ready = False
    with pytest.raises(StepFailed, match="not ready"):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert coord.global_steps == 0
    assert all(w.applies == 0 for w in workers)


def test_each_stage_receives_its_1f1b_op_list():
    coord, workers = build(stages=3, microbatches=4)
    coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    first = [(kind, mb) for kind, mb, _ in workers[0].ops]
    assert first[:2] == [("forward", 0), ("forward", 1)]  # two warm-up forwards
    assert first[-1] == ("ready", None)
    assert all(len(w.commands) == 2 for w in workers)  # one step + one apply


def test_apply_failure_poisons_generation():
    coord, workers = build(fail_on={1: {"apply"}})
    with pytest.raises(PipelinePoisoned):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert coord.global_steps == 0, "a poisoned generation never counts"
    # every subsequent call is rejected, even after workers heal
    workers[1].fail_on = set()
    with pytest.raises(PipelinePoisoned):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    with pytest.raises(PipelinePoisoned):
        coord.eval_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))


def test_no_tensor_handles_resolved_by_driver():
    """The coordinator returns a plain float built only from terminal losses."""
    coord, _ = build()
    loss = coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert isinstance(loss, float)
    assert loss == pytest.approx(2.0)


def test_facade_integration_with_real_coordinator():
    """rdsp.initialize wired to a real PipelineCoordinator over fake workers
    honours the public engine contract."""
    import torch
    from test_partition import ToyLM

    import ray_deepspeed_pipeline as rdsp
    from ray_deepspeed_pipeline import api
    from ray_deepspeed_pipeline.compiler import lower

    model = ToyLM()

    def factory(*, model, pipeline_config, ds_config, loss_fn, weights=None):
        plan = lower(model, pipeline_config, ds_config)
        workers = [FakeStageWorker(s, s == len(plan.stages) - 1) for s in range(len(plan.stages))]
        return PipelineCoordinator(plan, workers)

    original = api._coordinator_factory
    api._coordinator_factory = factory
    try:
        engine, _, _, _ = rdsp.initialize(
            model=model,
            config={"train_batch_size": 8},
            pipeline_config=rdsp.PipelineConfig(
                stages=2, partition=rdsp.UniformTransformerBlocks(), microbatches=4
            ),
            loss_fn=lambda outputs, labels: 0.0,
        )
        loss = engine.train_batch(data_iter=iter([(f"x{k}", f"y{k}") for k in range(4)]))
        assert isinstance(loss, torch.Tensor) and loss.shape == ()
        assert engine.global_steps == 1
    finally:
        api._coordinator_factory = original


def build_prefetching(microbatches=2):
    import dataclasses

    plan = dataclasses.replace(make_plan(2, microbatches), prefetch=True)
    workers = [FakeStageWorker(s, s == 1) for s in range(2)]
    return PipelineCoordinator(plan, workers), workers


def test_prefetch_reads_the_next_step_during_this_one_and_keeps_order():
    coord, workers = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(6)])
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x0", "x1"]
    assert next(data) == ("x4", "y4")  # x2, x3 were read ahead for step 2
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x2", "x3"]


def test_prefetch_refuses_a_different_iterator_and_eval_leaves_it_alone():
    coord, workers = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(6)])
    coord.train_batch(data)
    coord.eval_batch(iter([("e0", "f0"), ("e1", "f1")]))
    assert workers[0].last_inputs == ["e0", "e1"]
    with pytest.raises(StepFailed, match="same iterator"):
        coord.train_batch(iter([("z0", "w0"), ("z1", "w1")]))
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x2", "x3"]


def test_prefetch_reports_an_exhausted_iterator_on_the_step_that_needs_it():
    coord, _ = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(3)])
    coord.train_batch(data)  # reading ahead finds only one entry: not this step's problem
    with pytest.raises(StepFailed, match="exhausted"):
        coord.train_batch(data)


def test_a_missing_stage_client_is_a_validation_error():
    plan = make_plan(2, 2)
    with pytest.raises(ValidationError, match="stage clients"):
        PipelineCoordinator(plan, [FakeStageWorker(0, False)])


def test_engine_reports_each_stages_blocks():
    """Which transformer blocks each stage holds, e.g. to log the cut."""
    from ray_deepspeed_pipeline.engine import RayPipelineEngine

    coord, _ = build(stages=2)
    blocks = RayPipelineEngine(coord).stage_blocks
    assert len(blocks) == 2 and blocks[0][0] == 0 and blocks[0][1] == blocks[1][0]


def test_stages_fill_nodes_in_order():
    from ray_deepspeed_pipeline.stage_group import fill_nodes_in_order

    nodes = [("a", 8), ("b", 8)]
    assert fill_nodes_in_order([2] * 8, nodes) == ["a"] * 4 + ["b"] * 4
    assert fill_nodes_in_order([4, 4, 4, 4], nodes) == ["a", "a", "b", "b"]
    # a stage never straddles: 3 + 3 fill a, the next 3 goes to b
    assert fill_nodes_in_order([3, 3, 3, 3], nodes) == ["a", "a", "b", "b"]
    assert fill_nodes_in_order([8, 8], nodes) == ["a", "b"]
    assert fill_nodes_in_order([8, 8, 1], nodes) is None
    assert fill_nodes_in_order([1], []) is None


def test_a_stage_spanning_nodes_keeps_tensor_parallel_groups_on_one_node():
    from ray_deepspeed_pipeline.errors import ValidationError
    from ray_deepspeed_pipeline.stage_group import bundles_in_rank_order

    # 9 | 7 GPUs per node: a TP4 group would cross nodes
    table = {b: ("a" if b < 9 else "b") for b in range(16)}
    with pytest.raises(ValidationError, match="multiple of tp x sp = 4"):
        bundles_in_rank_order(table, ["a", "b"], 4)
    even = {b: ("a" if b in (0, 2, 4, 6, 8, 10, 12, 14) else "b") for b in range(16)}
    order = bundles_in_rank_order(even, ["a", "b"], 4)
    assert [even[b] for b in order] == ["a"] * 8 + ["b"] * 8
    assert sorted(order) == list(range(16))
    # TP4 groups are ranks 0-3, 4-7, ...: each on one node
    assert all(len({even[order[r]] for r in range(g, g + 4)}) == 1 for g in range(0, 16, 4))
    assert bundles_in_rank_order(even, ["b", "a"], 1)[:8] == [1, 3, 5, 7, 9, 11, 13, 15]


def test_a_stage_whose_ranks_do_not_share_the_checkpoint_directory_refuses_to_save():
    # a stage spanning nodes on node-local storage: rank 0 would fingerprint only its own
    # node's files, and the other node's ranks could not load what rank 0's node wrote
    from ray_deepspeed_pipeline.errors import CheckpointError
    from ray_deepspeed_pipeline.stage_group import StageGroupClient

    client = StageGroupClient([object(), object()], stage=3)
    shared = [{"rank": 0, "files": {"a": "1"}, "shares_dir": True}, {"rank": 1, "shares_dir": True}]
    assert client._stage_record(shared)["files"] == {"a": "1"}
    apart = [shared[0], {"rank": 1, "shares_dir": False}]
    with pytest.raises(CheckpointError, match="stage 3.*rank 1.*shared"):
        client._stage_record(apart)


def test_microbatches_of_different_row_counts_travel_with_their_shapes():
    import torch

    coord, workers = build(microbatches=2)
    same = [(torch.zeros(2, 8), "y0"), (torch.zeros(2, 8), "y1")]
    coord.train_batch(iter(same))
    assert workers[0].controls[-1]["varying"] is False
    short_last = [(torch.zeros(2, 8), "y0"), (torch.zeros(1, 8), "y1")]
    coord.train_batch(iter(short_last))
    assert workers[0].controls[-1]["varying"] is True


def test_a_payload_that_cannot_be_built_dispatches_nothing():
    coord, workers = build(stages=3)
    workers[2].fail_prepare = True
    with pytest.raises(StepFailed, match="stage 2"):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert all(w.commands == [] for w in workers), "no stage may start a step it cannot finish"


def test_a_retry_after_step_failed_reruns_the_failed_steps_entries():
    coord, workers = build(fail_on={1: {"forward"}})
    data = iter([(f"x{k}", f"y{k}") for k in range(8)])
    with pytest.raises(StepFailed):
        coord.train_batch(data)
    workers[1].fail_on = set()
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x0", "x1", "x2", "x3"]
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x4", "x5", "x6", "x7"]


def test_a_new_iterator_after_step_failed_drops_the_failed_steps_entries():
    coord, workers = build(fail_on={1: {"forward"}})
    with pytest.raises(StepFailed):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    workers[1].fail_on = set()
    coord.train_batch(iter([(f"z{k}", f"w{k}") for k in range(4)]))
    assert workers[0].last_inputs == ["z0", "z1", "z2", "z3"]


def test_a_retry_under_prefetch_keeps_the_step_read_ahead():
    coord, workers = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(8)])
    workers[0].ready = False  # fails after dispatch, once the next step is read ahead
    with pytest.raises(StepFailed, match="not ready"):
        coord.train_batch(data)
    workers[0].ready = True
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x0", "x1"]
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x2", "x3"]
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x4", "x5"]


def test_eval_batch_takes_its_own_iterator_beside_the_engines_loader():
    from torch.utils.data import DataLoader

    from ray_deepspeed_pipeline.engine import RayPipelineEngine

    coord, workers = build()
    rows = [(f"x{k}", f"y{k}") for k in range(8)]
    engine = RayPipelineEngine(coord, DataLoader(rows, batch_size=1, collate_fn=lambda b: b[0]))
    engine.eval_batch(iter([(f"e{k}", f"f{k}") for k in range(4)]))
    assert workers[0].last_inputs == ["e0", "e1", "e2", "e3"]
    engine.train_batch()  # the loader's entries were not consumed by the eval
    assert workers[0].last_inputs == ["x0", "x1", "x2", "x3"]


def test_a_stage_worker_keeps_a_bounded_record_of_executed_commands():
    import torch

    from ray_deepspeed_pipeline.boundary import Grid
    from ray_deepspeed_pipeline.stage_worker import StageWorkerActor, _PhaseTimer

    class Adapter:
        def apply(self):
            return True

    worker = StageWorkerActor(
        None,
        "{}",
        stage=0,
        rank=0,
        grid=Grid(1, 1),
        backend="gloo",
        n_microbatches=4,
        is_first=True,
        is_last=False,
    )
    worker.adapter, worker._timer = Adapter(), _PhaseTimer(torch.device("cpu"))
    for step in range(1000):
        worker.execute(f"g1.t{step}.apply", "apply")
    executed = worker.executed_commands()
    assert executed[-1] == "g1.t999.apply"
    assert len(executed) < 100, "one step's commands are kept, not every step's"
