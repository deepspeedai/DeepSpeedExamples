"""
ANNOTATED: DeepSpeed-Chat Supervised Fine-Tuning (SFT)

Original File: applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py

This script demonstrates a production-ready training pipeline with advanced features:
1. Dynamic DeepSpeed config generation
2. LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
3. ZeRO-3 model saving utilities
4. Distributed data loading with proper sampling
5. Gradient checkpointing integration
6. Evaluation loop with perplexity calculation

KEY PRODUCTION PATTERNS:
- Config generation based on arguments (not static JSON)
- Conditional CPU/GPU optimizer selection
- ZeRO-3 checkpoint saving (special handling required)
- LoRA layer conversion
- Proper distributed evaluation

RLHF CONTEXT:
This is Step 1 of the RLHF (Reinforcement Learning from Human Feedback) pipeline:
Step 1: Supervised Fine-Tuning (this script)
Step 2: Reward Model Training
Step 3: PPO Training
"""

import argparse
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, get_scheduler
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# ============================================================================
# DEEPSPEED CONFIG GENERATION (Production Pattern)
# ============================================================================

def get_train_ds_config(offload: bool, dtype: str, stage: int,
                        enable_tensorboard: bool, tb_path: str, tb_name: str):
    """
    [ANNOTATION] **DYNAMIC CONFIG GENERATION**

    This is a production pattern: Generate DeepSpeed config programmatically
    instead of using static JSON files.

    Benefits:
    - Config adapts to runtime arguments (offload, dtype, stage)
    - Easier to maintain (one function vs many JSON files)
    - Can be version controlled as code
    - Type checking and validation
    """

    device = "cpu" if offload else "none"

    # [ANNOTATION] Base configuration
    ds_config = {
        "train_batch_size": "auto",  # Calculated from per_device_batch * world_size * grad_accum
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",

        # [ANNOTATION] Mixed precision - only one enabled at a time
        "fp16": {
            "enabled": dtype == "fp16",
            "loss_scale_window": 100
        },
        "bf16": {
            "enabled": dtype == "bf16"
        },

        # [ANNOTATION] ZeRO configuration
        "zero_optimization": {
            "stage": stage,

            # [ANNOTATION] Conditional offloading based on argument
            "offload_optimizer": {
                "device": device,  # "cpu" if offload else "none"
                "pin_memory": True
            },

            # [ANNOTATION] For ZeRO-3, also offload parameters
            "offload_param": {
                "device": device,
                "pin_memory": True
            } if stage == 3 else {},

            # Communication optimizations
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        },

        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }

    # [ANNOTATION] Conditional TensorBoard logging
    if enable_tensorboard:
        ds_config["tensorboard"] = {
            "enabled": True,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }

    return ds_config


# ============================================================================
# OPTIMIZER SELECTION (CPU vs GPU)
# ============================================================================

def create_optimizer(model, args, ds_config):
    """
    [ANNOTATION] **CONDITIONAL OPTIMIZER SELECTION**

    Critical decision: Which optimizer to use based on offloading.

    DeepSpeedCPUAdam:
    - Required when offload_optimizer.device = "cpu"
    - Runs optimizer step on CPU
    - Slower but saves GPU memory

    FusedAdam:
    - Used when optimizer stays on GPU
    - Fused kernels for better performance
    - Requires GPU memory for optimizer states
    """

    # Get grouped parameters (with weight decay applied correctly)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate
    )

    # [ANNOTATION] Select optimizer based on offload setting
    if args.offload:
        # CPU offload: Use DeepSpeedCPUAdam
        AdamOptimizer = DeepSpeedCPUAdam
    else:
        # GPU: Use FusedAdam (faster)
        AdamOptimizer = FusedAdam

    optimizer = AdamOptimizer(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95)  # Standard values for LLM training
    )

    return optimizer


# ============================================================================
# LORA (LOW-RANK ADAPTATION) INTEGRATION
# ============================================================================

def setup_lora(model, args):
    """
    [ANNOTATION] **LoRA FOR PARAMETER-EFFICIENT FINE-TUNING**

    LoRA (Low-Rank Adaptation):
    - Add small trainable matrices to frozen model
    - Only train LoRA parameters (<<< full model parameters)
    - Merge back to original model for inference

    Example: 7B model
    - Without LoRA: Train all 7B parameters
    - With LoRA (rank=8): Train ~8M parameters (0.1% of model!)

    Benefits:
    - Much less memory for optimizer states
    - Faster training
    - Can train on smaller GPUs
    - Multiple LoRA adapters for different tasks
    """

    if args.lora_dim > 0:
        # [ANNOTATION] Convert linear layers to LoRA layers
        model = convert_linear_layer_to_lora(
            model,
            args.lora_module_name,  # Which modules to apply LoRA (e.g., "decoder.layers.")
            args.lora_dim            # Rank of LoRA matrices
        )

        if args.only_optimize_lora:
            # [ANNOTATION] Freeze all parameters except LoRA
            model = only_optimize_lora_parameters(model)

            # Make compatible with gradient checkpointing
            model = make_model_gradient_checkpointing_compatible(model)

    return model


# ============================================================================
# DISTRIBUTED DATA LOADING
# ============================================================================

def create_dataloaders(args, tokenizer):
    """
    [ANNOTATION] **DISTRIBUTED DATA LOADING PATTERN**

    Critical for multi-GPU training:
    - Each GPU must process different samples
    - Use DistributedSampler to partition data
    - Avoid duplicate computation across GPUs
    """

    # Create dataset
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase=1,
        seed=args.seed,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        sft_only_data_path=args.sft_only_data_path
    )

    # [ANNOTATION] **CRITICAL**: Use DistributedSampler for multi-GPU
    if args.local_rank == -1:
        # Single GPU: Use RandomSampler
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        # Multi-GPU: Use DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    # [ANNOTATION] DistributedSampler ensures:
    # - Each GPU gets different subset of data
    # - No overlap between GPUs
    # - Balanced load (approximately equal samples per GPU)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size
    )

    return train_dataloader, eval_dataloader


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    args = parse_args()

    # [ANNOTATION] **DISTRIBUTED INITIALIZATION**
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        # Set device for this process
        get_accelerator().set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        # Initialize distributed backend
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # [ANNOTATION] Generate DeepSpeed config
    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="step1_model"
    )

    # Set batch sizes
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = (
        args.per_device_train_batch_size *
        torch.distributed.get_world_size() *
        args.gradient_accumulation_steps
    )

    # [ANNOTATION] Barrier before data loading (avoid race conditions)
    torch.distributed.barrier()

    # Load tokenizer and model
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path,
                           tokenizer, ds_config, dropout=args.dropout)

    # [ANNOTATION] Apply LoRA if specified
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name, args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare data
    train_dataloader, eval_dataloader = create_dataloaders(args, tokenizer)

    # Create optimizer
    optimizer = create_optimizer(model, args, ds_config)

    # Create learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    # [ANNOTATION] **DEEPSPEED INITIALIZATION**
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )

    # [ANNOTATION] Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model_engine.gradient_checkpointing_enable()


    # ============================================================================
    # TRAINING LOOP
    # ============================================================================

    for epoch in range(args.num_train_epochs):
        model_engine.train()

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = to_device(batch, device)

            # Forward pass
            outputs = model_engine(**batch, use_cache=False)
            loss = outputs.loss

            # [ANNOTATION] DeepSpeed backward and step
            model_engine.backward(loss)
            model_engine.step()

        # [ANNOTATION] **DISTRIBUTED EVALUATION**
        model_engine.eval()
        perplexity, eval_loss = evaluation(model_engine, eval_dataloader, device)

        # Only rank 0 logs
        if args.global_rank == 0:
            print(f"Epoch {epoch+1}: Perplexity = {perplexity}, Loss = {eval_loss}")


    # ============================================================================
    # ZERO-3 MODEL SAVING (Special Handling Required)
    # ============================================================================

    if args.output_dir is not None:
        print_rank_0('Saving the final model ...', args.global_rank)

        # [ANNOTATION] Convert LoRA back to linear if used
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            # [ANNOTATION] For ZeRO-1 and ZeRO-2: Standard saving
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # [ANNOTATION] **CRITICAL FOR ZERO-3**
            # ZeRO-3 partitions parameters across GPUs
            # Need special saving logic to gather full model

            save_zero_three_model(
                model_engine,
                args.global_rank,
                args.output_dir,
                zero_stage=args.zero_stage
            )

            # [ANNOTATION] What save_zero_three_model does:
            # 1. Gather parameters from all GPUs (All-Gather)
            # 2. Rank 0 saves the full model
            # 3. Other ranks wait at barrier
            # 4. Ensures model can be loaded for inference


def evaluation(model, eval_dataloader, device):
    """
    [ANNOTATION] **DISTRIBUTED EVALUATION PATTERN**

    Important considerations:
    - Each GPU evaluates on different subset (via DistributedSampler)
    - Need to reduce losses across all GPUs
    - Use get_all_reduce_mean() to average losses
    """
    model.eval()
    losses = 0

    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses += loss.float()

    # Average over steps
    losses = losses / (step + 1)

    # [ANNOTATION] **ALL-REDUCE MEAN**: Average losses across all GPUs
    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass

    # Calculate perplexity
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")

    return perplexity, losses.item()


# ============================================================================
# KEY TAKEAWAYS FOR DEEPSPEED-CHAT SFT
# ============================================================================

"""
1. PRODUCTION CONFIG GENERATION:
   - Generate config dynamically based on arguments
   - Easier to maintain than multiple JSON files
   - Type-safe and version controlled

2. CONDITIONAL OPTIMIZER:
   - DeepSpeedCPUAdam when offloading to CPU
   - FusedAdam when keeping optimizer on GPU
   - Critical for performance

3. LORA INTEGRATION:
   - Parameter-efficient fine-tuning
   - Only train 0.1-1% of parameters
   - Huge memory savings for optimizer states
   - Can train on much smaller hardware

4. DISTRIBUTED DATA LOADING:
   - MUST use DistributedSampler for multi-GPU
   - Each GPU processes different subset
   - Avoid duplicate computation

5. ZERO-3 SAVING:
   - Cannot use standard torch.save()
   - Must use save_zero_three_model()
   - Gathers partitioned parameters before saving

6. DISTRIBUTED EVALUATION:
   - Each GPU evaluates on different subset
   - Use all_reduce to aggregate metrics
   - Only rank 0 should log/save

7. GRADIENT CHECKPOINTING:
   - Enable with model.gradient_checkpointing_enable()
   - Compatible with LoRA (with special handling)
   - 30-40% memory savings

8. PROPER BARRIERS:
   - torch.distributed.barrier() before data loading
   - Prevents race conditions
   - Critical for multi-node training
"""


# ============================================================================
# CONFIGURATION EXAMPLE
# ============================================================================

"""
# Generated config for ZeRO-3 with CPU offload:

{
  "train_batch_size": 128,  # 16 per GPU × 8 GPUs × 1 grad_accum
  "train_micro_batch_size_per_gpu": 16,
  "gradient_accumulation_steps": 1,

  "bf16": {"enabled": true},

  "zero_optimization": {
    "stage": 3,

    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },

    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },

    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 10,

  "tensorboard": {
    "enabled": true,
    "output_path": "./tensorboard_logs/",
    "job_name": "step1_model_tensorboard"
  }
}
"""


# ============================================================================
# USAGE
# ============================================================================

"""
# Single-node, 8 GPUs, ZeRO-3 with CPU offload:
deepspeed --num_gpus=8 main.py \\
    --model_name_or_path meta-llama/Llama-2-7b-hf \\
    --data_path Dahoas/rm-static \\
    --per_device_train_batch_size 16 \\
    --per_device_eval_batch_size 16 \\
    --max_seq_len 512 \\
    --learning_rate 1e-5 \\
    --weight_decay 0.0 \\
    --num_train_epochs 1 \\
    --gradient_accumulation_steps 1 \\
    --lr_scheduler_type cosine \\
    --num_warmup_steps 0 \\
    --seed 1234 \\
    --zero_stage 3 \\
    --offload \\
    --dtype bf16 \\
    --output_dir ./output \\
    --gradient_checkpointing

# With LoRA (parameter-efficient):
deepspeed --num_gpus=8 main.py \\
    --model_name_or_path meta-llama/Llama-2-7b-hf \\
    --data_path Dahoas/rm-static \\
    --per_device_train_batch_size 16 \\
    --zero_stage 3 \\
    --offload \\
    --lora_dim 128 \\
    --lora_module_name "model.layers." \\
    --only_optimize_lora \\
    --output_dir ./output_lora
"""

# [ANNOTATION] See applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py
