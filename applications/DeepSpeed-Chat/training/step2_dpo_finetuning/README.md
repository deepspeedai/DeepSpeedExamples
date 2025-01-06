# 🐕 Direct Preference Optimization (DPO) finetuning
[Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) is a novel approach to preference learning, which directly optimizes the policy without explicit reward modeling or reinforcement learning. It leverages a specific parameterization of the reward model that enables the extraction of the corresponding optimal policy in closed form. By using a simple classification loss, DPO aligns language models with human preferences, avoiding the complexity and instability often associated with RLHF.

As the paper says, "Your Language Model is Secretly a Reward Model." Therefore, the training arguments and the training process of DPO are mostly the same as the reward model, as shown in [step2 "Reward Model (RM) finetuning"](../step2_reward_model_finetuning/README.md). After the training of DPO, you will get a model that has been aligned with human preferences.

## 🏃 How to train the model

We provide the script for OPT-350m, which you can test by launching the command

```bash
training_scripts/opt/single_node/run_350m.sh
```

We also provide the script for llama2, which you can test by launching the command

```bash
training_scripts/llama2/run_llama2_7b.sh
```

## 🏃 How to evaluate the DPO checkpoint?

The checkpoint of DPO is exactly the language model that can be evaluated as [step1 "Supervised Finetuning"](../step1_supervised_finetuning/README.md).

## 💁 Datasets

Because DPO treats the language model as a reward model, the dataset for DPO is in the same format as that used for reward model fine-tuning. Each item in the dataset includes one "chosen" and one "rejected" output for the same input.
