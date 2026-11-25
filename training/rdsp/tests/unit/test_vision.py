# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Colocated vision: who encodes which image, who receives its features, and
the per-rank vision engine, in one process."""

import pytest
import torch

from ray_deepspeed_pipeline.boundary import Grid
from ray_deepspeed_pipeline.vision import ColocatedVisionEngine, VisionLayout, route_images


def step_inputs(has_image):
    """Per microbatch, per row: one 4-patch image or none."""
    out = []
    for row_flags in has_image:
        out.append(
            {
                "input_ids": torch.zeros(len(row_flags), 3, dtype=torch.long),
                "pixel_values": [torch.ones(4 if f else 0, 2) for f in row_flags],
                "image_grid_thw": [
                    torch.tensor([[1, 2, 2]]) if f else torch.zeros(0, 3, dtype=torch.long)
                    for f in row_flags
                ],
            }
        )
    return out


def test_first_stage_encodes_its_first_images_and_the_rest_go_to_later_stages():
    """Each first-stage cell encodes half its fair share itself (its first
    microbatches' images, so the pipeline starts at once); the other stages'
    ranks encode the rest, in microbatch order, while the pipeline fills."""
    inputs = step_inputs([[1, 1]] * 4)  # images 0..7 = (mb, row) in order
    layout = VisionLayout(first=Grid(dp=2), world=4)
    routes = [route_images(inputs, layout, rank) for rank in range(4)]
    assert [[image for image, *_ in r["own"]] for r in routes] == [[0], [1], [5, 6, 7], [2, 3, 4]]
    # image 1 is row 1 of microbatch 0: data-parallel cell 1 of stage 0
    assert [(i, dests, src) for i, _, _, dests, src in routes[1]["own"]] == [(1, [1], 1)]
    assert routes[0]["need"] == [(0, 0, 0), (1, 2, 3), (2, 4, 3), (3, 6, 2)]  # (mb, image, owner)
    assert routes[1]["need"] == [(0, 1, 1), (1, 3, 3), (2, 5, 2), (3, 7, 2)]
    assert routes[2]["need"] == [] and routes[3]["need"] == []


def test_rows_without_images_are_skipped_and_tp_peers_all_receive():
    inputs = step_inputs([[1, 0], [0, 1]])
    layout = VisionLayout(first=Grid(dp=1, tp=2), world=2)  # one stage: it encodes all
    routes = [route_images(inputs, layout, rank) for rank in range(2)]
    assert [[i for i, *_ in r["own"]] for r in routes] == [[0, 3], []]
    assert routes[0]["own"][0][3:] == ([0, 1], 0)  # both TP ranks; TP rank 0 returns grads
    assert routes[1]["need"] == [(0, 0, 0), (1, 3, 0)]


def test_per_row_pixels_required():
    inputs = step_inputs([[1, 1]])
    inputs[0]["pixel_values"] = torch.cat(inputs[0]["pixel_values"])
    with pytest.raises(Exception, match="per row"):
        route_images(inputs, VisionLayout(first=Grid(), world=1), 0)


def tiny_vision():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5VisionModel"):
        pytest.skip("transformers without Qwen3.5")
    cfg = transformers.Qwen3_5VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=16,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return transformers.Qwen3_5VisionModel(cfg).float()


def test_engine_features_and_gradients_match_the_encoder():
    """The engine encodes several images in one call and splits the output
    per image; its gradients are the encoder's, divided as asked."""
    import copy

    from ray_deepspeed_pipeline.vision import VisionTower

    encoder = tiny_vision()
    reference = copy.deepcopy(encoder)
    engine = ColocatedVisionEngine(
        VisionTower(encoder, "visual"),
        {"optimizer": {"type": "SGD", "params": {"lr": 1.0}}},
        torch.device("cpu"),
    )
    g = torch.Generator().manual_seed(1)
    grid = torch.tensor([[1, 4, 4]])
    images = [(i, torch.randn(16, 48, generator=g), grid) for i in (3, 7)]
    features = engine.forward(images, train=True)
    assert list(features) == [3, 7] and features[3].shape == (4, 16)

    full = reference(torch.cat([p for _, p, _ in images]), grid_thw=grid.repeat(2, 1)).pooler_output
    assert torch.allclose(torch.cat([features[3], features[7]]), full, atol=1e-6)

    grads = {3: torch.randn(4, 16, generator=g), 7: torch.randn(4, 16, generator=g)}
    before = [q.detach().clone() for q in reference.parameters()]
    engine.backward(grads)
    engine.reduce_gradients(group=None, divide_by=2)
    engine.apply()  # SGD, lr 1, no momentum: each weight moves by its gradient
    full.backward(torch.cat([grads[3], grads[7]]) / 2)
    for (name, p), q, b in zip(
        engine.tower.module.named_parameters(), reference.parameters(), before
    ):
        assert torch.allclose(b - p.detach(), q.grad, atol=1e-6), name


def test_bf16_encoder_keeps_fp32_master_weights():
    """With bf16 weights the optimizer steps an fp32 master copy: updates
    far below bf16's resolution still add up instead of rounding away."""
    from ray_deepspeed_pipeline.vision import VisionTower

    encoder = tiny_vision().to(torch.bfloat16)
    engine = ColocatedVisionEngine(
        VisionTower(encoder, "visual"),
        {"optimizer": {"type": "SGD", "params": {"lr": 1e-5}}},
        torch.device("cpu"),
    )
    first = next(engine.tower.module.parameters())
    start = first.detach().float().clone()
    for _ in range(400):
        for p in engine.tower.module.parameters():
            p.grad = torch.ones_like(p)
        engine.reduce_gradients(group=None, divide_by=1)
        engine.apply()
    # 400 steps of 1e-5: a change of 4e-3, each step far below bf16 resolution near 1
    moved = (start - first.detach().float()).abs().mean()
    assert 2e-3 < float(moved) < 6e-3


def _sharded_rank(rank, init_file, out_dir):
    import torch.distributed as dist

    from ray_deepspeed_pipeline.vision import VisionTower

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    # SGD with momentum: per-weight state to split, and steps linear in the
    # gradient (Adam would turn last-bit differences in near-zero gradients
    # between the two image groupings into full steps)
    config = {"optimizer": {"type": "SGD", "params": {"lr": 0.1, "momentum": 0.9}}}
    engine = ColocatedVisionEngine(
        VisionTower(tiny_vision(), "visual"), config, torch.device("cpu")
    )
    g = torch.Generator().manual_seed(1)
    grid = torch.tensor([[1, 4, 4]])
    images = [(i, torch.randn(16, 48, generator=g), grid) for i in range(4)]
    grads = {i: torch.randn(4, 16, generator=g) for i in range(4)}
    mine = images[rank * 2 : rank * 2 + 2]  # each rank encodes two of the four images
    for _ in range(2):
        engine.forward(mine, train=True)
        engine.backward({i: grads[i] for i, _, _ in mine})
        engine.reduce_gradients(group=dist.group.WORLD, divide_by=2)
        engine.apply()
    state = engine.full_state()
    torch.save(
        {
            "params": [p.detach() for p in engine.tower.module.parameters()],
            "shard": engine.master.numel(),
            "state": state if rank == 0 else None,
        },
        f"{out_dir}/r{rank}.pt",
    )
    dist.destroy_process_group()


def test_sharded_optimizer_matches_one_unsharded_encoder(tmp_path):
    """Two ranks, each encoding half of the images: each keeps half of the
    fp32 master weights and optimizer state, and both end with the weights
    one unsharded encoder gets from all the images. The checkpoint holds the
    whole state and resumes it."""
    import torch.multiprocessing as mp

    from ray_deepspeed_pipeline.vision import VisionTower

    mp.spawn(_sharded_rank, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2)
    ranks = [torch.load(tmp_path / f"r{r}.pt", weights_only=False) for r in range(2)]

    # SGD with momentum: per-weight state to split, and steps linear in the
    # gradient (Adam would turn last-bit differences in near-zero gradients
    # between the two image groupings into full steps)
    config = {"optimizer": {"type": "SGD", "params": {"lr": 0.1, "momentum": 0.9}}}
    one = ColocatedVisionEngine(VisionTower(tiny_vision(), "visual"), config, torch.device("cpu"))
    g = torch.Generator().manual_seed(1)
    grid = torch.tensor([[1, 4, 4]])
    images = [(i, torch.randn(16, 48, generator=g), grid) for i in range(4)]
    grads = {i: torch.randn(4, 16, generator=g) for i in range(4)}
    for _ in range(2):
        one.forward(images, train=True)
        one.backward(grads)
        one.reduce_gradients(group=None, divide_by=2)
        one.apply()
    total = sum(p.numel() for p in one.tower.module.parameters())
    assert ranks[0]["shard"] == ranks[1]["shard"] == -(-total // 2)
    for r in ranks:
        for p, q in zip(r["params"], one.tower.module.parameters()):
            assert torch.allclose(p, q.detach(), atol=1e-6)

    # resume from the full state: one more step matches one more unsharded step
    path = tmp_path / "vision.pt"
    torch.save(ranks[0]["state"], path)
    resumed = ColocatedVisionEngine(
        VisionTower(tiny_vision(), "visual"), config, torch.device("cpu")
    )
    resumed.load(str(path))
    for e in (one, resumed):
        e.forward(images, train=True)
        e.backward(grads)
        e.reduce_gradients(group=None, divide_by=2)
        e.apply()
    for p, q in zip(resumed.tower.module.parameters(), one.tower.module.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_vision_schedule_encodes_ahead_and_backpropagates_once_gradients_exist():
    """1F1B ops of the last of 2 stages over 6 microbatches. Microbatch m is
    encoded before forward m-2 (the first stage may run 2 ahead), and
    backpropagated before forward m+2 (the first stage's backward of m is
    done by then), or at the end of the step."""
    from ray_deepspeed_pipeline.vision import vision_schedule

    ops = [(k, m, f"{k}{m}") for m in range(6) for k in ("forward", "backward")]
    before, end = vision_schedule(ops, microbatches={0, 1, 3, 5}, n_stages=2)
    at = {ops[i][2]: v for i, v in before.items()}
    assert at == {
        "forward0": ([], [0, 1]),
        "forward1": ([], [3]),
        "forward2": ([0], []),
        "forward3": ([1], [5]),
        "forward5": ([3], []),
    }
    assert end == [5]


def test_engine_encodes_and_backpropagates_microbatch_by_microbatch():
    """Two microbatches encoded in turn, then backpropagated in turn: the
    same gradients as one batch, and only the images still waiting for their
    gradient keep a graph."""
    import copy

    from ray_deepspeed_pipeline.vision import VisionTower

    encoder = tiny_vision()
    reference = copy.deepcopy(encoder)
    engine = ColocatedVisionEngine(
        VisionTower(encoder, "visual"),
        {"optimizer": {"type": "SGD", "params": {"lr": 1.0}}},
        torch.device("cpu"),
    )
    g = torch.Generator().manual_seed(2)
    grid = torch.tensor([[1, 4, 4]])
    images = [(i, torch.randn(16, 48, generator=g), grid) for i in range(4)]
    grads = {i: torch.randn(4, 16, generator=g) for i in range(4)}
    engine.forward(images[:2], train=True)
    engine.forward(images[2:], train=True)
    engine.backward({i: grads[i] for i in (0, 1)})
    assert engine.waiting() == {2, 3}
    engine.backward({i: grads[i] for i in (2, 3)})
    assert engine.waiting() == set()
    before = [q.detach().clone() for q in reference.parameters()]
    engine.reduce_gradients(group=None, divide_by=1)
    engine.apply()
    full = reference(torch.cat([p for _, p, _ in images]), grid_thw=grid.repeat(4, 1)).pooler_output
    full.backward(torch.cat([grads[i] for i in range(4)]))
    for p, q, b in zip(engine.tower.module.parameters(), reference.parameters(), before):
        assert torch.allclose(b - p.detach(), q.grad, atol=1e-5)
