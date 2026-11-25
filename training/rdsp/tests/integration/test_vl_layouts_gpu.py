# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The three vision placements on GPU with real DeepSpeed engines, each
against the unsplit model (tiny Qwen3.5-VL, fp32, SGD so any gradient
scaling error shows):

  shared-layout   the vision encoder on the first stage, both stages TP2
                  (the encoder's blocks split by AutoTP like the decoder's)
  vision-stage    a vision-only first stage (first cut at 0), DP2, then a
                  TP2 language stage
  colocated       the encoder on all 4 GPUs (colocated vision), language
                  stages TP2 and DP2

    pytest tests/integration/test_vl_layouts_gpu.py   # needs 4 GPUs
"""

import os

import pytest
import torch

ray = pytest.importorskip("ray")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
    pytest.skip("transformers without Qwen3.5", allow_module_level=True)
if (torch.cuda.device_count() if torch.cuda.is_available() else 0) < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)
pytest.importorskip("deepspeed")

from test_vl_pipeline import lm_loss, make_batches, tiny_qwen3_5_vl  # noqa: E402

import ray_deepspeed_pipeline as rdsp  # noqa: E402
from ray_deepspeed_pipeline.config import StageOverride  # noqa: E402

# first steps compile the linear-attention Triton kernels, which takes far
# longer than the CPU suite's stage-to-stage wait (tests/conftest.py)
os.environ["RDSP_P2P_TIMEOUT_S"] = "600"

HERE = os.path.dirname(os.path.abspath(__file__))
N_MB, ROWS, LR = 2, 2, 0.05
DS = {
    "train_batch_size": N_MB * ROWS,
    "gradient_accumulation_steps": N_MB,
    "optimizer": {"type": "SGD", "params": {"lr": LR, "momentum": 0.9}},
    "zero_optimization": {"stage": 1},
    "zero_allow_untested_optimizer": True,
    "steps_per_print": 10**6,
}

LAYOUTS = {
    "shared-layout": dict(
        cuts=(3,),
        overrides=(
            StageOverride(stage=0, num_gpus=2, tp=2),
            StageOverride(stage=1, num_gpus=2, tp=2),
        ),
    ),
    "vision-stage": dict(
        cuts=(0,),
        overrides=(StageOverride(stage=0, num_gpus=2), StageOverride(stage=1, num_gpus=2, tp=2)),
    ),
    "colocated": dict(
        cuts=(3,),
        overrides=(StageOverride(stage=0, num_gpus=2, tp=2), StageOverride(stage=1, num_gpus=2)),
        colocated_vision=rdsp.ColocatedVision(),
    ),
}


@pytest.fixture()
def ray_ctx():
    """A fresh Ray per case: a failed case's actors cannot hold the GPUs."""
    unit = os.path.abspath(os.path.join(HERE, "..", "unit"))
    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": f"{HERE}:{unit}"}},
    )
    yield
    ray.shutdown()


def model(seed=0):
    torch.manual_seed(seed)
    return transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl()).float()


def reference_losses(steps):
    net = model().cuda()
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        total = 0.0
        for inputs, labels in make_batches():
            full = {
                k: (torch.cat(v) if isinstance(v, list) else v).cuda() for k, v in inputs.items()
            }
            loss = lm_loss(net(**full, use_cache=False).logits, labels.cuda()) / N_MB
            loss.backward()
            total += float(loss)
        opt.step()
        losses.append(total)
    return losses


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layout_matches_unsplit_model(ray_ctx, name):
    layout = LAYOUTS[name]
    engine, _, _, _ = rdsp.initialize(
        model=model(),
        config=DS,
        loss_fn=lm_loss,
        pipeline_config=rdsp.PipelineConfig(
            stages=len(layout["cuts"]) + 1,
            partition=rdsp.ExplicitCuts(layout["cuts"]),
            stage_overrides=layout["overrides"],
            colocated_vision=layout.get("colocated_vision"),
        ),
    )
    try:
        got = [float(engine.train_batch(data_iter=iter(make_batches()))) for _ in range(3)]
    finally:
        for worker in engine._coordinator._workers:
            worker.shutdown()
    assert got == pytest.approx(reference_losses(3), rel=2e-3)
