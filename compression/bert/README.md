#### Install

``pip install -r requirements.txt``

You will also need to install updated DeepSpeed version (>0.7.0), which contains the compression library.

#### Key File: run_glue_no_trainer.py

The python code is modified based on [HuggingFace&#39;s PyTorch text_classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). The key added feature is the implementation of knowledge distillation (KD)（--distill_method one_stage). If no KD, run (--distill_method zero_stage).

#### Folders (config, huggingface_transformer, bash_script)

* **config:** This folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.
* **huggingface_transformer:** This folder serves the implementation of knowledge distillation. It's based on [HuggingFace&#39;s transformer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
  The change is line 383, where we output attention_scores instead of attention_prob.
* **bash_script**  This folder contains many bash scripts for various kinds of compression. See more descriptions and results in our [tutorial page](https://www.deepspeed.ai/).

