#### Example of fine-tuning GPT using random-LTD (https://arxiv.org/abs/2211.11586) and the composed data efficiency library (https://arxiv.org/abs/2212.03597).

#### Install

``pip install -r requirement.txt``

You will also need to install updated DeepSpeed version (>=0.8.0), which contains the random-ltd library.

#### Key File: run_clm_no_trainer.py

The python code is modified based on huggingface (https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py). The key added feature is our random-ltd.

#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including the schedules of sequence-length and the layers applied by random-ltd.

#### bash script

* **run_base_random_ltd.sh/run_medium_random_ltd.sh**  This bash script contains jobs for training with random-ltd
* Run the job under the gpt_finetuning directory:

 ``DeepSpeedExamples/random_ltd/gpt_finetuning$ . ./bash_script/run_base_random_ltd.sh``


 ``DeepSpeedExamples/random_ltd/gpt_finetuning$ . ./bash_script/run_medium_random_ltd.sh``
 See more descriptions and results in our [tutorial page](https://www.deepspeed.ai/).

#### Reproducing results in data efficiency paper (https://arxiv.org/abs/2212.03597).

```finetune/ds_finetune_gpt2_run.sh``` is the script to reproduce the PTB finetuning results (random-LTD, curriculum learning, and CL+random-LTD) in our [data efficiency paper](https://arxiv.org/abs/2212.03597). For CL metrics that require data analysis (e.g., the vocabulary rarity metric), you need to first use ```finetune/ds_analyze_gpt_data_*``` to analyze and index the dataset. Additional tutorial can be found at [DeepSpeed website](https://www.deepspeed.ai/tutorials/data-efficiency/).
