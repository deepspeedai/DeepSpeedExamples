# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""TorchTitan on the rows Megatron-Bridge exported: Qwen3.5 on CORD-v2.

Bridge's SFT rows, exported by bench/mimo/export_bridge_batches.py (the files rdsp
replays), are fed to TorchTitan unchanged: same rows in the same order, same token
ids, same answer-only loss mask, same image patches, every row padded to 2048. Only
the conventions change:

- labels: the export keeps Hugging Face's (label t is token t; the loss shifts);
  TorchTitan's label t is the target of input t, so label t = exported label t + 1,
  and the last input has none.
- images: the export holds the processor's patches; TorchTitan's collator patchifies
  image tensors itself, so cord_data.ExportedRowCollator takes the patches as they are.
  TorchTitan's Hugging Face parity test reads both with one layout
  (scripts/checkpoint_conversion/numerical_tests_qwen3_5.py, the rearrange pattern
  "(t bh bw m n) (c pt ph pw)").

Each row is one document of 2048 positions (padding included, causal), as rdsp runs
it with --drop-padding-mask. The data path (cord_data.py) is checked on a CPU against
the export and Hugging Face's M-RoPE positions: checks/, results/data_check.txt.

Run from a TorchTitan checkout (read at 3c36a77; no change to it), on 8 GPUs:

    PYTHONPATH=<this directory> EXPORTED=<dir of step_*.pt> HF_ASSETS=<HF snapshot of
    Qwen/Qwen3.5-27B> MODULE=cord_qwen35 CONFIG=qwen35_27b_cord_tp4pp2 ./run_train.sh

STEPS (default 50). CONFIG=qwen35_27b_cord_tp4pp2_rc: the same with full recompute.
CONFIG=qwen35_27b_cord_tp4dp2 (no pipeline: TP4 x FSDP over 2) and qwen35_27b_cord_tp4dp2_rc;
qwen35_27b_cord_tp2pp2dp2 (TP2 x PP2 x FSDP 2) and qwen35_27b_cord_tp2pp4 (TP2 x PP4), each also
_rc.
"""

import os

from cord_data import ROWS_PER_STEP, exported_dataloader
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import QWEN3_5_SPECIAL_TOKENS, build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

SEQ = int(os.environ.get("SEQ", "2048"))  # 4096 for NVIDIA's published SFT setup


def qwen35_27b_cord_tp4pp2() -> Trainer.Config:
    """Qwen3.5-27B, TP4 x PP2 x DP1 on 8 GPUs, the equal-work settings of the
    Bridge and rdsp runs: 64 rows of 2048 tokens per step, 1 row per microbatch,
    1F1B, layers split 32 | 32 with the vision encoder on stage 0 (TorchTitan's
    default split), token-mean loss over the step, AdamW lr 1e-5 constant, betas
    0.9/0.95, eps 1e-8, no weight decay, no clipping, no warm-up, fp32 master
    weights and Adam moments, no recompute."""
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=4,
        enable_sequence_parallel=True,  # TorchTitan's default
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        num_pp_microbatches=ROWS_PER_STEP,
    )
    model_config = build_model_config("27B", seq_len=SEQ, parallelism=parallelism)
    return Trainer.Config(
        model=model_config,
        parallelism=parallelism,
        hf_assets_path=os.environ["HF_ASSETS"],
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        dataloader=exported_dataloader(os.environ["EXPORTED"]),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_config)),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=1e-5,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                        weight_decay=0.0,
                        fused=True,
                        moment_dtype="parameter",
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0, decay_ratio=0.0),
            # 0 would scale every gradient by 0; inf leaves them unchanged
            max_norm=float("inf"),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=SEQ,
            num_tokens_per_train_step=-1,  # one row per PP microbatch: 64 rows a step
            max_context_length=SEQ,
            steps=int(os.environ.get("STEPS", "50")),
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,  # image sizes vary from row to row
        ),
        checkpointer=CheckpointManager.Config(initial_load_in_hf=True, load_only=True),
        activation_checkpoint=None,
        metrics=MetricsProcessor.Config(log_freq=1),
    )


def qwen35_27b_cord_tp4pp2_rc() -> Trainer.Config:
    """qwen35_27b_cord_tp4pp2 with every block recomputed (TorchTitan's FullAC), for
    when the plain layout runs out of memory."""
    config = qwen35_27b_cord_tp4pp2()
    config.activation_checkpoint = FullAC.Config()
    return config


def qwen35_27b_cord_tp4dp2() -> Trainer.Config:
    """No pipeline: TP4 x data parallel 2 on 8 GPUs, the layout Bridge and rdsp run
    as TP4 x DP2. TorchTitan's data parallelism is FSDP (dp_shard 2): weights,
    gradients and Adam are split over the two copies, where Bridge's distributed
    optimizer and DeepSpeed's ZeRO-1 split only the optimizer state. 32 microbatches
    of 2 rows (one per copy) per step, the same rows as Bridge's; everything else as
    qwen35_27b_cord_tp4pp2."""
    config = qwen35_27b_cord_tp4pp2()
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=4,
        enable_sequence_parallel=True,
    )
    config.parallelism = parallelism
    config.model = build_model_config("27B", seq_len=SEQ, parallelism=parallelism)
    config.dataloader = exported_dataloader(os.environ["EXPORTED"], data_parallel=2)
    # 64 rows of 2048 tokens a step: 32 accumulation steps of one row per copy
    config.training.num_tokens_per_train_step = ROWS_PER_STEP * SEQ
    return config


def qwen35_27b_cord_tp4dp2_rc() -> Trainer.Config:
    """qwen35_27b_cord_tp4dp2 with every block recomputed (FullAC)."""
    config = qwen35_27b_cord_tp4dp2()
    config.activation_checkpoint = FullAC.Config()
    return config


def _with_layout(
    config: Trainer.Config, parallelism: ParallelismConfig, data_parallel: int, flavor: str = "27B"
) -> Trainer.Config:
    """config on another layout (and model flavor): the model rebuilt for it, Bridge's rows
    dealt to the data-parallel copies as Megatron deals them, ROWS_PER_STEP rows of 2048
    tokens a step."""
    config.parallelism = parallelism
    config.model = build_model_config(flavor, seq_len=SEQ, parallelism=parallelism)
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(config.model))
    )
    config.dataloader = exported_dataloader(os.environ["EXPORTED"], data_parallel=data_parallel)
    config.training.num_tokens_per_train_step = ROWS_PER_STEP * SEQ
    return config


def qwen35_27b_cord_tp2pp2dp2() -> Trainer.Config:
    """TP2 x PP2 x data parallel 2 (FSDP over the 2 copies), as Bridge and rdsp ran this
    layout: 32 pipeline microbatches of one row per copy, the vision encoder on stage 0,
    layers 32 | 32 (TorchTitan's default split). No recompute: a fit check."""
    return _with_layout(
        qwen35_27b_cord_tp4pp2(),
        ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            enable_sequence_parallel=True,
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=ROWS_PER_STEP // 2,
        ),
        data_parallel=2,
    )


def qwen35_27b_cord_tp2pp2dp2_rc() -> Trainer.Config:
    """qwen35_27b_cord_tp2pp2dp2 with every block recomputed (FullAC)."""
    config = qwen35_27b_cord_tp2pp2dp2()
    config.activation_checkpoint = FullAC.Config()
    return config


def qwen35_27b_cord_tp2pp4() -> Trainer.Config:
    """TP2 x PP4, no data parallelism, as Bridge and rdsp ran this layout: 64 pipeline
    microbatches of one row, the vision encoder on stage 0, layers 16 | 17 | 16 | 15
    (TorchTitan's default split: the embedding and the output layer each count as one
    layer). No recompute: a fit check."""
    return _with_layout(
        qwen35_27b_cord_tp4pp2(),
        ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=2,
            enable_sequence_parallel=True,
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=ROWS_PER_STEP,
        ),
        data_parallel=1,
    )


def qwen35_27b_cord_tp2pp4_rc() -> Trainer.Config:
    """qwen35_27b_cord_tp2pp4 with every block recomputed (FullAC)."""
    config = qwen35_27b_cord_tp2pp4()
    config.activation_checkpoint = FullAC.Config()
    return config


def qwen35_27b_cord_fsdp8() -> Trainer.Config:
    """TorchTitan's own regime: FSDP over all 8 GPUs (dp_shard 8), no tensor or pipeline
    parallelism. Weights, gradients and Adam are split eight ways, and each layer is
    gathered whole just before it runs. 8 accumulation steps of one row per GPU per step:
    64 rows, Bridge's rows dealt to the 8 copies as Megatron deals them. No recompute: a
    fit check."""
    return _with_layout(
        qwen35_27b_cord_tp4pp2(),
        ParallelismConfig(
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
        data_parallel=8,
    )


def qwen35_27b_cord_fsdp8_rc() -> Trainer.Config:
    """qwen35_27b_cord_fsdp8 with every block recomputed (FullAC)."""
    config = qwen35_27b_cord_fsdp8()
    config.activation_checkpoint = FullAC.Config()
    return config


def qwen35_27b_cord_fsdp16_rc() -> Trainer.Config:
    """Two nodes: FSDP over all 16 GPUs (dp_shard 16), every block recomputed (FullAC), as
    qwen35_27b_cord_fsdp8_rc on one node. Run with ROWS=128: 8 accumulation steps of one row
    per GPU per step, the rows dealt to the 16 copies as Megatron deals them."""
    config = _with_layout(
        qwen35_27b_cord_tp4pp2(),
        ParallelismConfig(
            data_parallel_shard_degree=16,
            tensor_parallel_degree=1,
        ),
        data_parallel=16,
    )
    config.activation_checkpoint = FullAC.Config()
    return config


def _without_router_balancing_loss(config: Trainer.Config) -> Trainer.Config:
    """No load-balancing loss on the experts' router, as in the Bridge and rdsp runs: the
    systems each define it differently (TorchTitan's flavor sets a batch-wise one at 1e-3),
    so only "none" is the same everywhere. The loss is cross-entropy alone."""
    for layer in config.model.layers:
        if getattr(layer, "moe", None) is not None:
            layer.moe.router.aux_loss = None
    return config


def qwen35_35b_a3b_cord_fsdp16() -> Trainer.Config:
    """Qwen3.5-35B-A3B (mixture of experts: 256 experts, 8 per token, plus a shared expert)
    on two nodes: FSDP over all 16 GPUs, no expert parallelism, so each GPU holds 1/16 of
    every expert's weights and each layer is gathered whole before it runs. Run with ROWS=128
    and HF_ASSETS at the 35B-A3B snapshot: 8 accumulation steps of one row per GPU per step.
    No router load-balancing loss. No recompute: a fit check."""
    return _without_router_balancing_loss(
        _with_layout(
            qwen35_27b_cord_tp4pp2(),
            ParallelismConfig(
                data_parallel_shard_degree=16,
                tensor_parallel_degree=1,
            ),
            data_parallel=16,
            flavor="35B-A3B",
        )
    )


def qwen35_35b_a3b_cord_fsdp16_rc() -> Trainer.Config:
    """qwen35_35b_a3b_cord_fsdp16 with every block recomputed (FullAC)."""
    config = qwen35_35b_a3b_cord_fsdp16()
    config.activation_checkpoint = FullAC.Config()
    return config


def qwen35_35b_a3b_cord_fsdp16_ep8() -> Trainer.Config:
    """qwen35_35b_a3b_cord_fsdp16 with expert parallelism 8: each node's 8 GPUs split the
    256 experts (32 each) and send tokens to them with all-to-all inside the node; the expert
    weights are FSDP-split over the 2 nodes, everything else over all 16 GPUs. The model is
    built for the EP layout (the flavor reads it). No router load-balancing loss. No
    recompute: a fit check."""
    return _without_router_balancing_loss(
        _with_layout(
            qwen35_27b_cord_tp4pp2(),
            ParallelismConfig(
                data_parallel_shard_degree=16,
                tensor_parallel_degree=1,
                expert_parallel_degree=8,
            ),
            data_parallel=16,
            flavor="35B-A3B",
        )
    )


def qwen35_35b_a3b_cord_fsdp16_ep8_rc() -> Trainer.Config:
    """qwen35_35b_a3b_cord_fsdp16_ep8 with every block recomputed (FullAC; the all-to-alls
    are replayed in the backward pass)."""
    config = qwen35_35b_a3b_cord_fsdp16_ep8()
    config.activation_checkpoint = FullAC.Config()
    return config


def _bf16_reduce(config: Trainer.Config) -> Trainer.Config:
    """Gradients reduce-scattered in bf16 instead of fp32, as Megatron-Bridge and rdsp send
    theirs (TorchTitan still accumulates the microbatches in fp32). Half the bytes of each
    gradient reduce-scatter."""
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def qwen35_27b_cord_fsdp8_rc_bf16() -> Trainer.Config:
    """qwen35_27b_cord_fsdp8_rc with gradients reduced in bf16."""
    return _bf16_reduce(qwen35_27b_cord_fsdp8_rc())


def qwen35_27b_cord_fsdp8_ffnrc_bf16() -> Trainer.Config:
    """qwen35_27b_cord_fsdp8 with only each block's MLP recomputed (RegionAC: every declared
    region kept except feed_forward.*) and gradients reduced in bf16. Between no recompute
    (out of memory at 73 GiB) and FullAC."""
    config = _bf16_reduce(qwen35_27b_cord_fsdp8())
    config.activation_checkpoint = RegionAC.Config(
        save_regions=["*"], recompute_regions=["feed_forward.*"]
    )
    return config


def qwen35_35b_a3b_cord_fsdp16_ep8_bf16() -> Trainer.Config:
    """qwen35_35b_a3b_cord_fsdp16_ep8 with gradients reduced in bf16."""
    return _bf16_reduce(qwen35_35b_a3b_cord_fsdp16_ep8())


def qwen35_27b_cord_fsdp16_bf16() -> Trainer.Config:
    """Two nodes, FSDP over all 16 GPUs, no recompute, gradients reduced in bf16 (ROWS=128).
    Half the state per GPU of FSDP8, so it may fit where FSDP8 without recompute did not."""
    return _bf16_reduce(
        _with_layout(
            qwen35_27b_cord_tp4pp2(),
            ParallelismConfig(
                data_parallel_shard_degree=16,
                tensor_parallel_degree=1,
            ),
            data_parallel=16,
        )
    )


def qwen35_27b_cord_fsdp16_ffnrc_bf16() -> Trainer.Config:
    """qwen35_27b_cord_fsdp16_bf16 with only each block's MLP recomputed (RegionAC)."""
    config = qwen35_27b_cord_fsdp16_bf16()
    config.activation_checkpoint = RegionAC.Config(
        save_regions=["*"], recompute_regions=["feed_forward.*"]
    )
    return config


def qwen35_27b_cord_fsdp16_rc_bf16() -> Trainer.Config:
    """qwen35_27b_cord_fsdp16_rc with gradients reduced in bf16."""
    return _bf16_reduce(qwen35_27b_cord_fsdp16_rc())


def qwen35_35b_a3b_cord_fsdp8_ep8_rc_bf16() -> Trainer.Config:
    """Qwen3.5-35B-A3B on one node: FSDP over 8 GPUs with expert parallelism 8 (32 experts per
    GPU), every block recomputed, gradients reduced in bf16, no router load-balancing loss
    (ROWS=64). TorchTitan keeps fp32 weights and Adam moments: ~16 bytes a parameter, about
    70 GB a GPU before activations, so this is a fit check."""
    config = _without_router_balancing_loss(
        _with_layout(
            qwen35_27b_cord_tp4pp2(),
            ParallelismConfig(
                data_parallel_shard_degree=8,
                tensor_parallel_degree=1,
                expert_parallel_degree=8,
            ),
            data_parallel=8,
            flavor="35B-A3B",
        )
    )
    config.activation_checkpoint = FullAC.Config()
    return _bf16_reduce(config)


# rdsp's measured cut of the 27B over 8 pipeline stages (decoder layers per stage), given to
# TorchTitan and Megatron-Bridge alike
PP8_CUT_27B = (1, 9, 10, 9, 9, 9, 10, 7)


def _module_split(cut: tuple[int, ...]) -> list[list[str]]:
    """Per-stage module names for a layer cut: the vision encoder and embedding on the first
    stage, the final norm and output layer on the last."""
    parts, start = [], 0
    for i, n in enumerate(cut):
        names = [f"layers.{j}" for j in range(start, start + n)]
        start += n
        if i == 0:
            names = ["vision_encoder", "tok_embeddings"] + names
        if i == len(cut) - 1:
            names += ["norm", "lm_head"]
        parts.append(names)
    return parts


def _from_dcp(config: Trainer.Config) -> Trainer.Config:
    """Weights from a DCP copy of the Hugging Face checkpoint (TT_DCP, made with TorchTitan's
    scripts/checkpoint_conversion/convert_from_hf.py): TorchTitan's Qwen3.5 HF loader needs the
    embedding or the output layer on every pipeline stage (state_dict_adapter.py from_hf), so a
    middle stage of 3 or more stages cannot load the HF files."""
    config.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=False,
        initial_load_path=os.environ["TT_DCP"],
        load_only=True,
        initial_load_model_only=True,
    )
    return config


def qwen35_27b_cord_pp8_rc() -> Trainer.Config:
    """TP1 x PP8 on one node, 1F1B, 64 microbatches of one row, at rdsp's measured cut
    (PP8_CUT_27B), every block recomputed (FullAC), gradients reduced in bf16 (no data
    parallel here, so no reduce), weights from DCP."""
    config = _with_layout(
        qwen35_27b_cord_tp4pp2(),
        ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=8,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=ROWS_PER_STEP,
            pipeline_parallel_module_fqns_per_model_part=_module_split(PP8_CUT_27B),
        ),
        data_parallel=1,
    )
    config.activation_checkpoint = FullAC.Config()
    return _from_dcp(_bf16_reduce(config))


def qwen35_27b_cord_pp8dp2_rc() -> Trainer.Config:
    """Two nodes (ROWS=128): PP8 x FSDP over 2 copies, 64 microbatches of one row per copy,
    otherwise as qwen35_27b_cord_pp8_rc."""
    config = _with_layout(
        qwen35_27b_cord_tp4pp2(),
        ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=8,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=ROWS_PER_STEP // 2,
            pipeline_parallel_module_fqns_per_model_part=_module_split(PP8_CUT_27B),
        ),
        data_parallel=2,
    )
    config.activation_checkpoint = FullAC.Config()
    return _from_dcp(_bf16_reduce(config))


def qwen35_35b_a3b_cord_fsdp16_bf16() -> Trainer.Config:
    """qwen35_35b_a3b_cord_fsdp16 with gradients reduced in bf16."""
    return _bf16_reduce(qwen35_35b_a3b_cord_fsdp16())
