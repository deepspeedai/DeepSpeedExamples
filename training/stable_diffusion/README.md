# Lora-enhanced distillation on Stable Diffusion model

This repository contains the implementation of Lora-enhanced distillation applied to the Stable Diffusion (SD) model. By combining the LoRA technique with distillation, we've achieved remarkable results, including a significant reduction in inference time and a 50% decrease in memory consumption. Importantly, this integration of LoRA-enhanced distillation maintains image quality and alignment with the provided prompt. For additional details on this work, please consult our technical report [TODO: add link].

In this implementation, we have adapted the dreambooth finetuning [code](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#dreambooth-training-example) as our baseline. Below, you'll find information regarding input data, training, and inference.

## Installation

You need to have huggingface [diffusers](https://github.com/huggingface/diffusers) installed on your machine. Then install the requirements:

<pre>
pip install -r requirements.txt
</pre>

## Training

### Training Data
Our training data includes a significant dataset of pre-generated images by [SD](https://github.com/poloclub/diffusiondb). You are not required to download the input data. Instead, you can specify or modify it within the training code (`train_sd_distill_lora.py`) as needed.To train the model, follow these steps:

### Training Script

1. Run the `mytrainbash.sh` file.
2. The finetuned model will be saved inside the output directory.

Here's an example command to run the training script:

<pre>
bash mytrainbash.sh
</pre>

Make sure to customize the training parameters in the script to suit your specific requirements.

## Inference

For inference, you can use the `inf-loop.py` Python code. Follow these steps:

1. Provide your desired prompts as input in the script.
2. Run the `inf_txt2img_loop.py` script.

Here's an example command to run the inference script:

<pre>
deepspeed inf_txt2img_loop.py
</pre>