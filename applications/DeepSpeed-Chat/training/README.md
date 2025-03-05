# Training Experiences
It's important to note that training large language models (LLMs) and Reinforcement Learning from Human Feedback (RLHF) are still open problems with many unknowns. DeepSpeed-Chat aims to provide an end-to-end RLHF training pipeline with efficient and fast system support, rather than a comprehensive solution for RLHF training. As this field is relatively new, there are various unknowns for both users and developers.

Users are encouraged to experiment with the pipeline, provide feedback, and make suggestions. Contributions to the project are welcome when users find something useful and tested for their use cases. By working together, we can advance the development of this project and improve our understanding of LLMs and RLHF training.

## Three Training Steps Discussion
### Step 1: Supervised Finetuning
Supervised fine-tuning (SFT) has indeed made significant progress in the field of large language models (LLMs). However, unexpected behaviors such as repeating content generation and inconsistency between perplexity (PPL) scores and generation capabilities can still occur.

Based on our testing, there are several terms that affect the generation behavior:
* ```weight decay```: OPT models are pretrained with weight decay. Following that, finetuning normally inherits this setting. However, it may not produce the desired model. Particularly, for our OPT-1.3B example, we disabled weight decay.
* ```dropout```: Similar as above, dropout is used in OPT pretraining. However, SFT may not necessarily need it. Particularly, for our OPT-1.3B example, we enabled dropout.
* ```dataset```: Using more data usually provides better model quality. But if the sources of datasets are too different, it may hurt the performance. For our OPT-1.3B example, we use the following four datasets: ```Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets```.
* ```training epochs``` Normally, to avoid overfitting, we choose smaller training epochs instead of longer epochs if smaller epochs can achieve similar model quality (in this case, we use PPL as an indicator). However, similar to InstructGPT pointed, we found even though we got overfitting due to longer training, it is still recommended to use longer training epochs to get better generation quality. Particularly, for our OPT-1.3B example, we use 16 epochs even though we found that 1 or 2 epochs training can reach the same PPL score.

### Step 2: Reward Model Finetuning
Reward model (RM) fine-tuning is indeed similar to SFT, with the main differences being: (1) the training datasets are different - RM requires both good responses and bad responses to the same query; (2) the training loss is different - RM requires pair ranking loss as the optimizing objective.

We provide two metrics for the reward model: (1) the reward score for accepted responses (and bad responses), and (2) the accuracy, i.e., when accepted responses can get higher scores than rejected responses. Sometimes, we observe that the accuracy is very high, but the average reward score for accepted answers is negative, or the rejected answer's score is similar to accepted answers. Would this affect the step-3 model quality? If we use the metric reward score gain for step-3, this probably won't have any issue. However, this machine learning metric (reward score gain/increasing) cannot really reflect the step-3 model generation quality. As such, we do not have a definitive answer yet.

Here, we share more about what we observed during our exploration:
* ```weight decay```: For our OPT-350m example, we enabled weight decay with 0.1.
* ```dropout```: For our OPT-350m example, we disabled dropout.
* ```dataset```: For our OPT-350m example, we use the following four datasets: ```Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets```.
* ```training epochs``` InstructGPT suggests to finetune the model with 1 epoch since overfitting hurts the step 3 performance. During our exploration, we did not see overfitting behavior when we increased the training epochs. However, to follow the instructions from the authors. We set training epoch to be 1.

Also, we provide more explorations here even though we have not set them as an option or included them in our current pipeline
* ```multiple answers for one prompt``` In InstructGPT, authors specifically mentioned that using paird rejected and accepted answers for one prompt is not suitable for reward model training. Therefore, InstructGPT constructs the dataset with 4--9 answers per prompt. However, we did not find good datasets with this feature.
* ```initialize RM with SFT or Pretrained checkpoint``` We internally tested this but did not see a big difference for either accuracy or reward score. Also, in InstructGPT, the authors have the same finding. However, we encourage users to try it for their own usage.
* ```Reward score calculation``` We use the final token (or the first padding token) to get the reward score. However, it might not be the optimal choice. For instance, users can try the average score for the entire answer, etc.
* ```Reward loss objective``` We simply use the ranking loss to be the objective. However, others, like MSE, can also be an option.


### Step 3: RLHF finetuning
The RLHF finetuning is the most complicated step among the three-step training. Similar to SFT, the reward score cannot really reflect the model generation quality. Also, we sometimes observed that the reward score drops to the initial phase at a certain point and then quickly recovers. To make things worse, we also see the training can easily get divergence. We here share our settings and observations.

* ```weight decay```: For our OPT-1.3B/350m (actor/critic) example, we disabled weight decay for both models.
* ```dropout```: We disabled droppout for OPT-1.3B and enabled it for OPT-350m.
* ```dataset```: We use the following single dataset: ```Dahoas/rm-static```.
* ```training epochs``` The reward score quickly becomes platou. Therefore, we set the training epoch to be 1 for our OPT-1.3B/350m (actor/critic) example. However, longer training may bring better model quality as SFT.
* ```ema checkpoint``` We observe ema checkpoint can generally bring better model generation quality as stated in InstructGPT.
* ```PPO related hyperparameters``` PPO training has a lot of hyperparameters, see [here](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py#L61-L66). For now, we hard-coded them for users but you may want to adjust them for you own usage.
* ```mix unsupervised training``` InstructGPT suggests mixing PPO and unsupervised training to prevent the loss of the model's benchmark quality. However, when we directly apply the hyperparameter from Instruct, the model cannot converge. Therefore, we stop exploring this. However, users are encouraged to test it and tune the hyperparameter for their own usage.
* ```diverging issue``` We have found that it is very unstable to use different generation training batch sizes (`--per_device_generation_batch_size`) and PPO training batch sizes (`--per_device_training_batch_size`), more than one PPO training epoch (`--ppo_epochs`), or more than one generation batch (`--generation_batches 1`). These all point to the same problem: we are not able to update the actor model multiple times after generating experimental data. Therefore, in all of our successful runs, we have set `per_device_generation_batch_size=per_device_training_batch_size` and `ppo_epochs=generation_batches=1`. This is unexpected for a standard RL training pipeline, and we have tried different methods to overcome this, but all have failed. One of the most likely reasons for this instability is that we found the `log_probs` and `old_log_probs` used in the `actor_loss_fn` function can quickly diverge even within two consecutive iterations, which causes the corresponding `ratio` to be huge. Setting a strict upper bound can alleviate this problem, but it cannot fully resolve the convergence issue.

### About our testing
We did most of our accuracy/quality testing on OPT-1.3B (SFT and Actor model) and OPT-350m (RW and Critic model). Particularly, we used the 16 V100-32G (DGX-2 node) GPUs to run our experiments.

The hyperparameters included in our scripts are based on our own testing. Therefore, it may not work for your case when (but not limited to): (1) a different number of GPUs, (2) different model sizes, (3) different model families, etc.

Also note that you could find even better training configurations/recipes than what we provided. We did not extensively test all hyperparameter combinations due to resource constraints.

### Training logs
We are sharing our training logs for all three steps for an OPT-1.3b actor and OPT-350m critic trained with x16-V100-32GB GPUs:

| Step         | Run Script     | Training Log |
|--------------|-----------|------------|
| 1 | [opt/single_node/run_1.3b.sh](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_node/run_1.3b.sh) | [opt-1.3b-globalBatchSize128.log](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_log_output/opt-1.3b-globalBatchSize128.log) |
| 2 | [opt/single_node/run_350m.sh](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_node/run_350m.sh) |  [opt-350m_globalBatchSize-64.log](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_log_output/opt-350m_globalBatchSize-64.log) |
| 3 | [opt/single_node/run_1.3b.sh](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/single_node/opt/run_1.3b.sh) | [actor_opt-1.3b_critic_opt-350m_globalBatchSize64.log](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_log_output/actor_opt-1.3b_critic_opt-350m_globalBatchSize64.log) |

### Characterization Scripts
Scripts for sweeping training across various parameters (Zero Stage, Offload, Lora, etc) are available for Step 1, 2, and 3. These scripts can be further extended to sweep across additional parameters such as learning rate.

| Step         | Sweep Script     | README |
|--------------|-----------|-----------|
| 1 | [run_step1_sweep.sh](./step1_supervised_finetuning/training_scripts/opt/single_node/sweep/run_step1_sweep.sh) | [README](./step1_supervised_finetuning/training_scripts/opt/single_node/sweep/README.md) |
| 2 | [run_step2_sweep.sh](./step2_reward_model_finetuning/training_scripts/opt/single_node/sweep/run_step2_sweep.sh) | [README](./step2_reward_model_finetuning/training_scripts/opt/single_node/sweep/README.md) |
| 3 | [run_step3_sweep.sh](./step3_rlhf_finetuning/training_scripts/opt/single_node/sweep/run_step3_sweep.sh) | [README](./step3_rlhf_finetuning/training_scripts/opt/single_node/sweep/README.md) |

### Others
RLHF (Reinforcement Learning for Human Feedback) training is still an open problem, and DeepSpeed-Chat is designed to be a starting point for researchers and practitioners to work on it with an efficient and fast training experience. The Hybrid-Engine and other efficient components, like LoRA, can be inherited from DeepSpeed-Chat, allowing you to develop your own RLHF training pipeline for exploration, research, and other purposes.

Contributions from users are highly appreciated to build a more successful, easier-to-use, and more stable RLHF training pipeline together.
