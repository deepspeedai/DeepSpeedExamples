# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The stage-local adapter, driven by an injected plain-torch engine stub.

The adapter's routing, cut, VJP, scaling and lifecycle logic do not depend on
the engine; real DeepSpeed engines are covered by the GPU integration tests.
StubEngine and the helpers here are shared with the CPU integration tests."""

import ast
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F
from test_partition import ToyLM

from ray_deepspeed_pipeline.config import UniformTransformerBlocks
from ray_deepspeed_pipeline.deepspeed_adapter import DeepSpeedStageAdapter
from ray_deepspeed_pipeline.errors import CheckpointError, ValidationError
from ray_deepspeed_pipeline.partition import build_stage_module, partition_parameters

N_MB, ROWS, SEQ, VOCAB = 4, 2, 6, 20


class StubEngine:
    """Plain-torch stand-in for a DeepSpeed engine on CPU, with the surface the
    adapter uses: __call__, backward, step, save/load_checkpoint.

    Data parallel like ZeRO-0: gradients are averaged over the stage-local
    world before the optimizer step. Optional StepLR via
    conf["scheduler"]["params"]["gamma"]. Fault injection: while
    $RDSP_TEST_FAIL_APPLY names an existing file, the terminal stage's step()
    raises while the other stages still apply (a genuine partial apply)."""

    def __init__(self, module, conf):
        self.module = module
        opt = conf.get("optimizer", {})
        params = opt.get("params", {})
        lr = params.get("lr", 0.05)
        if opt.get("type") == "SGD":
            self.optimizer = torch.optim.SGD(
                module.parameters(), lr=lr, momentum=params.get("momentum", 0.0)
            )
        else:
            self.optimizer = torch.optim.AdamW(module.parameters(), lr=lr)
        self.lr_scheduler = None
        if "scheduler" in conf:
            gamma = conf["scheduler"]["params"]["gamma"]
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=gamma
            )
        self.global_steps = 0

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, loss):
        # fault injection: while $RDSP_TEST_FAIL_BACKWARD names an existing
        # file, the FIRST stage's third backward of a step raises (a failure
        # in the middle of a step, after partial accumulation)
        flag = os.environ.get("RDSP_TEST_FAIL_BACKWARD")
        self._bwd = getattr(self, "_bwd", 0) + 1
        if (
            flag
            and os.path.exists(flag)
            and not self._is_terminal()
            and len(getattr(self.module, "pre", ())) > 0
            and self._bwd == 3
        ):
            raise RuntimeError("injected backward failure")
        loss.backward()

    def zero_grad(self):
        self._bwd = 0
        for p in self.module.parameters():
            p.grad = None

    def _is_terminal(self):
        return len(getattr(self.module, "post", ())) > 0

    def step(self):
        self._bwd = 0
        flag = os.environ.get("RDSP_TEST_FAIL_APPLY")
        if flag and os.path.exists(flag) and self._is_terminal():
            raise RuntimeError("injected optimizer-apply failure")
        dist = torch.distributed
        if dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.module.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= dist.get_world_size()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.global_steps += 1

    def _rank(self):
        dist = torch.distributed
        return dist.get_rank() if dist.is_initialized() else 0

    def save_checkpoint(self, save_dir, tag=None, client_state=None, save_latest=True):
        os.makedirs(os.path.join(save_dir, tag), exist_ok=True)
        torch.save(
            {
                "module": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "global_steps": self.global_steps,
            },
            os.path.join(save_dir, tag, f"stub_rank{self._rank()}.pt"),
        )

    def load_checkpoint(
        self, load_dir, tag=None, load_optimizer_states=True, load_lr_scheduler_states=True
    ):
        path = os.path.join(load_dir, tag, f"stub_rank{self._rank()}.pt")
        if not os.path.exists(path):
            return None, None
        state = torch.load(path, weights_only=False)
        self.module.load_state_dict(state["module"])
        if load_optimizer_states:
            self.optimizer.load_state_dict(state["optimizer"])
        if load_lr_scheduler_states and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.global_steps = state["global_steps"]
        return path, {}


def stub_engine_factory(module, conf):
    return StubEngine(module, conf)


def loss_fn(outputs, labels):
    return F.cross_entropy(outputs.reshape(-1, VOCAB), labels.reshape(-1))


def make_adapters(model):
    parts = partition_parameters(model, UniformTransformerBlocks(), 2)
    modules = [
        build_stage_module(model, p.block_start, p.block_stop, p.parameter_names) for p in parts
    ]
    a0 = DeepSpeedStageAdapter(
        modules[0], {}, N_MB, is_first=True, is_last=False, engine_factory=stub_engine_factory
    )
    a1 = DeepSpeedStageAdapter(
        modules[1],
        {},
        N_MB,
        is_first=False,
        is_last=True,
        loss_fn=loss_fn,
        engine_factory=stub_engine_factory,
    )
    return a0, a1, modules


def make_data():
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (N_MB * ROWS, SEQ))
    labels = torch.randint(0, VOCAB, (N_MB * ROWS, SEQ))
    return ids, labels


def test_loss_fn_only_on_terminal():
    model = ToyLM()
    with pytest.raises(ValidationError, match="terminal stage"):
        parts = partition_parameters(model, UniformTransformerBlocks(), 2)
        m = build_stage_module(
            model, parts[0].block_start, parts[0].block_stop, parts[0].parameter_names
        )
        DeepSpeedStageAdapter(
            m,
            {},
            N_MB,
            is_first=True,
            is_last=False,
            loss_fn=loss_fn,
            engine_factory=stub_engine_factory,
        )


def test_two_stage_parity_with_monolithic_baseline():
    torch.manual_seed(0)
    model = ToyLM()
    a0, a1, _ = make_adapters(model)
    ids, labels = make_data()

    # GPipe order (all forwards, then backwards in reverse): the adapter is
    # schedule-agnostic
    losses = {}
    for k in range(N_MB):
        h = a0.forward(k, ids[k * ROWS : (k + 1) * ROWS])
        losses[k] = a1.forward(k, h, labels=labels[k * ROWS : (k + 1) * ROWS])
    for k in reversed(range(N_MB)):
        g = a1.backward(k)
        assert g is not None and g.device.type == "cpu"
        assert a0.backward(k, grad=g) is None  # first stage returns nothing

    # baseline: same weights (fresh copies from the same model), one graph
    parts = partition_parameters(model, UniformTransformerBlocks(), 2)
    m0 = build_stage_module(
        model, parts[0].block_start, parts[0].block_stop, parts[0].parameter_names
    )
    m1 = build_stage_module(
        model, parts[1].block_start, parts[1].block_stop, parts[1].parameter_names
    )
    ref_loss = loss_fn(m1(m0(ids)), labels)
    ref_loss.backward()

    mean_staged = sum(losses.values()) / N_MB
    assert mean_staged == pytest.approx(ref_loss.item(), rel=1e-6)
    for (n, p), (rn, rp) in zip(a0.engine.module.named_parameters(), m0.named_parameters()):
        assert n == rn
        assert torch.allclose(p.grad, rp.grad, atol=1e-6), f"stage0 grad {n}"
    for (n, p), (rn, rp) in zip(a1.engine.module.named_parameters(), m1.named_parameters()):
        assert n == rn
        assert torch.allclose(p.grad, rp.grad, atol=1e-6), f"stage1 grad {n}"


def test_ready_apply_lifecycle():
    model = ToyLM()
    a0, a1, _ = make_adapters(model)
    ids, labels = make_data()
    assert not a0.ready()
    for k in range(N_MB):
        h = a0.forward(k, ids[k * ROWS : (k + 1) * ROWS])
        a1.forward(k, h, labels=labels[k * ROWS : (k + 1) * ROWS])
    assert not a0.ready(), "activations still cached"
    for k in range(N_MB):
        a0.backward(k, grad=a1.backward(k))
    assert a0.ready() and a1.ready()
    before = a0.engine.module.pre[0].weight.detach().clone()
    assert a0.apply() and a1.apply()
    assert not torch.equal(before, a0.engine.module.pre[0].weight), (
        "apply must perform the optimizer update"
    )
    assert not a0.ready(), "lifecycle resets after apply"


def test_eval_forward_builds_no_graph_and_caches_nothing():
    model = ToyLM()
    a0, a1, _ = make_adapters(model)
    ids, labels = make_data()
    h = a0.eval_forward(0, ids[:ROWS])
    loss = a1.eval_forward(0, h, labels=labels[:ROWS])
    assert isinstance(loss, float)
    assert not a0._acts and not a1._acts


def test_adapter_module_imports_no_ray():
    spec = importlib.util.find_spec("ray_deepspeed_pipeline.deepspeed_adapter")
    tree = ast.parse(open(spec.origin).read())
    for node in ast.walk(tree):
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        assert all(n.split(".")[0] != "ray" for n in names), (
            "the DeepSpeed adapter must not know Ray exists"
        )


def test_abandoned_generation_leaves_no_gradients_behind():
    """A step that fails after some backwards (before apply) must not leak its
    partial gradients into the retry: begin_generation() discards them."""
    torch.manual_seed(0)
    model = ToyLM()
    a0, a1, _ = make_adapters(model)
    ids, labels = make_data()
    h = a0.forward(0, ids[:ROWS])
    a1.forward(0, h, labels=labels[:ROWS])
    a0.backward(0, grad=a1.backward(0))  # 1 of N_MB backwards, then "failure"
    assert not a0.drained()
    assert any(p.grad is not None for p in a0.engine.module.parameters())
    a0.begin_generation()
    a1.begin_generation()
    assert a0.drained() and a1.drained()
    assert all(p.grad is None for p in a0.engine.module.parameters())
    assert all(p.grad is None for p in a1.engine.module.parameters())


def test_abandoned_generation_clears_offloaded_gradient_sums():
    """Under ZeRO optimizer offload the partial sum of an abandoned step sits
    in host buffers and a micro-step counter says whether the next backward
    adds to it; both must be reset, or a retry adds onto the failed step."""
    torch.manual_seed(0)
    a0, a1, _ = make_adapters(ToyLM())
    ids, labels = make_data()
    h = a0.forward(0, ids[:ROWS])
    a1.forward(0, h, labels=labels[:ROWS])
    a0.backward(0, grad=a1.backward(0))
    optimizer = a0.engine.optimizer
    optimizer.cpu_offload = True  # the state ZeRO-Offload keeps mid-step
    optimizer.accumulated_grads_in_cpu = {0: torch.ones(3)}
    optimizer.micro_step_id = 0
    a0.begin_generation()
    assert optimizer.accumulated_grads_in_cpu == {} and optimizer.micro_step_id == -1


def test_loading_a_missing_shard_is_a_checkpoint_error(tmp_path):
    first, _, _ = make_adapters(ToyLM())
    with pytest.raises(CheckpointError, match="no checkpoint"):
        first.load_shard(str(tmp_path), "missing")
