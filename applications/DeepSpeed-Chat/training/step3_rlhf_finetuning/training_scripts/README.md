### 💁For each folder, the bash scripts are examples of "facebook/opt" family.

If you want to change your model such as EleutherAI/gpt-j-6b, you may simply update
``` --actor_model_name_or_path ${step1_path} --critic_model_name_or_path ${step2_path} ```.

If you don't have step 1 and step 2 models. You may simply try
``` bash
--actor_model_name_or_path facebook/opt-1.3b --critic_model_name_or_path facebook/opt-350m
```
⚡⚡⚡ When you use above script, please make sure you comment out the following such that it won't load the model weight from previous paths.
```bash
applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L60
```

For the models we support, please see [our landing page](./../../../README.md#-supported-models-)
