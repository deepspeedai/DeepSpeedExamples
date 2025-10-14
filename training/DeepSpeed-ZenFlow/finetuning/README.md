
# ZenFlow Llama-2 Fine-Tuning Example

This project demonstrates how to fine-tune a [Llama-2](https://huggingface.co/meta-llama) model using [DeepSpeed](https://www.deepspeed.ai/) with **ZenFlow**, a stall-free offloading engine for large-scale model training.

## Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Configure training**

Edit `zf_config.json` to enable ZenFlow:

```json
"zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
    },
    "zenflow": {
        "topk_ratio": 0.1,
        "update_interval": 4,
        "full_warm_up_rounds": 0,
        "overlap_step": true
    }
}
```

3. **Run fine-tuning**

```bash
bash finetune_llama.sh
```

This runs LLaMA-2 fine-tuning on Alpaca-52K using DeepSpeed + ZenFlow, saving checkpoints to `./alpaca_output`.

## Example Output

Below is a sample log showing step time and loss values. You can see significant speedup after the first full step:

```
ZenFlowCPUAdam initialized with overlap step.
Step 5, Loss: 1.2599, Time: 719.58ms 
Step 6, Loss: 0.9847, Time: 702.81ms <-- gradient accumulation with overlapped update
Step 7, Loss: 0.6220, Time: 705.50ms
Step 8, Loss: 0.5173, Time: 1912.92ms <-- full optimizer step of remaining part and update parameters
Step 9, Loss: 0.4557, Time: 890.60ms
Step 10, Loss: 0.3882, Time: 740.11ms
Step 11, Loss: 0.3627, Time: 731.95ms
Step 12, Loss: 0.3341, Time: 2221.18ms
Step 13, Loss: 0.2453, Time: 1061.80ms
```

## Key Insight
Steps like 5，6 and 7 are accumulation steps where ZenFlow overlaps part of the optimizer step in the background. These steps remain fast (~700ms).

Step 8 performs the remaining part of optimizer step and updates parameters to the GPU (2–2.2s).

Without ZenFlow, a full update would take nearly 4 seconds, and ZenFlow distributes half of this cost across earlier accumulation steps via asynchronous overlap.

This demonstrates how ZenFlow hides much of the CPU offload cost, enabling near stall-free training. Crucially, ZenFlow not only overlaps the CPU optimizer step but also maintains training progress on the GPU by immediately updating the most important gradients.

## Notes

- To change model, batch size, or epochs, modify `finetune_llama.sh`.
- All DeepSpeed and ZenFlow options are controlled via `zf_config.json`.

## Citation

To cite ZenFlow, please cite our [arxiv report](https://arxiv.org/abs/2505.12242):

```bib
@misc{lan2025zenflowenablingstallfreeoffloading,
      title={ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates}, 
      author={Tingfeng Lan and Yusen Wu and Bin Ma and Zhaoyuan Su and Rui Yang and Tekin Bicer and Masahiro Tanaka and Olatunji Ruwase and Dong Li and Yue Cheng},
      year={2025},
      eprint={2505.12242},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2505.12242}, 
}
```
