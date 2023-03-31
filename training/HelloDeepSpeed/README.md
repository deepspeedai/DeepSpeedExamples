# Training a Masked Language Model with PyTorch and DeepSpeed

In this tutorial, we will create and train a Transformer encoder on the Masked Language Modeling (MLM) task. Then we will show the changes necessary to integrate DeepSpeed, and show some of the advantages of doing so.

Table of contents
=================

<!--toc-start-->
  * [(1) Training a Transformer Encoder (BERT / Roberta) model for MLM](#1-training-a-transformer-encoder-bert--roberta-model-for-mlm)
    * [1.0 Some Good Practices](#10-some-good-practices)
    * [1.1 The Masked Language Modeling Task](#11-the-masked-language-modeling-task)
    * [1.2 Creating a Transformer model](#12-creating-a-transformer-model)
    * [1.3 Training the Model](#13-training-the-model)
  * [(2) Integrating DeepSpeed For More Efficient Training](#2-integrating-deepspeed-for-more-efficient-training)
    * [2.0 Core DeepSpeed Code Changes](#20-core-deepspeed-code-changes)
    * [2.1 Launching Training](#21-launching-training)
    * [2.2 Mixed Precision Training (fp16)](#22-mixed-precision-training-fp16)
    * [2.3 Zero Redundancy Optimizer (ZeRO)](#23-zero-redundancy-optimizer-zero)
  * [References](#references)
<!--toc-end-->

## 1. Training a Transformer Encoder (BERT / Roberta) model for MLM

### 1.0 Some Good Practices

### Version Control and Reproducibility

One of the most important parts of training ML models is for the experiments to be reproducible (either by someone else, or by you 3 months later). Some steps that help with this are:

* Use some form of version control (eg: `git`). Additionally, make sure to save the `gitdiff` and `githash`, so that anyone trying to replicate the experiment can easily do so

* Save all the hyperparameters associated with the experiment (be it taken from a config or parsed from the command line)

* Seed your random generators. Some useful tips can be found [here](https://pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility).

* Specify all the packages and their versions. This can be a `requirements.txt` file, a conda `env.yaml` file or a `pyproject.toml` file. If you want complete reproducibility, you can also include a `Dockerfile` to specify the environment to run the experiment in.

In this example, the checkpoint directory has the following format:

```bash
{exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    `-experiment-name
        |- hparams.json
        |- githash.log
        |- gitdiff.log
        `- tb_dir/
```
This ensures that if you revisit this experiment, it is easy to understand what the experiment was about, when was it run, with what hyperparameters and what was the status of the code when the run was executed.

### Writing Unit Tests

Unit tests can help catch bugs early, and also set up a fast feedback loop. Since some experiments can take days or even weeks to run, both of these things are quite invaluable.

In this tutorial, two primary parts are data creating and model training. Hence, we test both these parts (see [here](./tests/test_train_bert.py)):

1. Testing Data creation: Data creation for the MLM task involves randomly masking words to generate model inputs. To test for correctness, we test if the fraction of masked tokens matches what we expect from our `DataLoader`. For more details, please take a look at

```python
def test_masking_stats(tol: float = 1e-3):
    """Test to check that the masking probabilities
    match what we expect them to be.
    """
    ...
```

2. Model training and checkpointing: Since pretraining experiments are usually quite expensive (take days / weeks to complete), chances are that you might run into some hardware failure before the training completes. Thus, it is crucial that the checkpointing logic is correct to allow a model to resume training. One way to do this is to train a small model for a few iterations and see if the model can resume training and if the checkpoints are loaded correctly. See `test_model_checkpointing` for an example test.

---

💡 **_Tip:_** While saving checkpoints, make sure to also save the optimizer states along with the model parameters !

---

### 1.1 The Masked Language Modeling Task

The main idea behind the MLM task is to get the model to fill in the blanks based on contextual clues present **both before and after** the blank. Consider, for example, the following sentence:

> In the beautiful season of ____ the ____ shed their leaves.

Given the left context `season` and the right context `shed their leaves`, one can guess that the blanks are `Autumn` and `trees` respectively. This is exactly what we want the model to do: utilize both the left and right context to fill in the blanks.

In order to do that, we do the following:

1. Tokenize a sentence into word(pieces)
2. Randomly select some words to mask, and replace them with a special \<Mask\> token
    * Of the masked tokens, it is common to replace a fraction of them with a random token, and leave a fraction of them unchanged.
3. Collect the actual words that were masked, and use them as targets for the model to predict against:
    * From the model's perspective, this is a simple `CrossEntropy` loss over the vocabulary of the model.

In this tutorial, we use the [wikitext-2-v1](https://huggingface.co/datasets/wikitext) dataset from [HuggingFace datasets](https://github.com/huggingface/datasets). To see how this is done in code, take a look at `masking_function` in [train_bert.py](./train_bert.py).


### 1.2 Creating a Transformer model

A Transformer model repeatedly applies a (Multi-Headed) Self-Attention block and a FeedForward layer to generate contextual representations for each token. Thus, the key hyperparameters for a Transformer model usually are

1. The number of Self-Attention + FeedForward blocks (depth)
2. The size of the hidden representation
3. The number of Self Attention Heads
4. The size of the intermediate representation between the FeedForward block

Check out the `create_model` function in [train_bert.py](./train_bert.py) to see how this is done in code. In this example, we create a Roberta model [[3](#3)]

---
📌 **Note:** You can check out [[1](#1), [2](#2)] as a starting point for better understanding Transformers. Additionally, there are a number of blogs that do a nice deep dive into the workings of these models (eg: [this](https://nlp.seas.harvard.edu/2018/04/03/attention.html), [this](https://jalammar.github.io/illustrated-bert/) and [this](https://jalammar.github.io/illustrated-transformer/)).

---

### 1.3 Training the Model

In order to train the model, you can run the following command

```bash
python train_bert.py --checkpoint_dir ./experiments
```
This will create a model with the default parameters (as specified by the arguments to the `train` function), and train it on the wikitext dataset. Other parameters can be configured from the command line as:

```bash
python train_bert.py --checkpoint_dir ./experiments \
    --mask_prob ${mask_prob} \
    --random_replace_prob ${random_replace_prob} \
    --unmask_replace_prob ${unmask_replace_prob} \
    --max_seq_length ${max_seq_length} \
    --tokenizer ${tokenizer} \
    --num_layers ${num_layers} \
    --num_heads ${num_heads} \
    --ff_dim ${ff_dim} \
    --h_dim ${h_dim} \
    --dropout ${dropout} \
    --batch_size ${batch_size} \
    --num_iterations ${num_iterations} \
    --checkpoint_every ${checkpoint_every} \
    --log_every ${log_every} \
    --local_rank ${local_rank}

```

The parameters are explained in more details in the docstring of `train`.

---
💡 **_Tip:_** If you have a GPU available, you can considerably speedup your training by running it on the GPU. Simply set the `local_rank` to the GPU you want to run it on. Eg: for a single GPU machine, this would look like
```bash
--local_rank 0
```

---

## 2. Integrating DeepSpeed For More Efficient Training

In this next section we'll add DeepSpeed to the model presented in Section 1 and turn on several features.

## 2.0 Core DeepSpeed Code Changes

Please see the [Writing DeepSpeed Models](https://www.deepspeed.ai/getting-started/#writing-deepspeed-models) instructions written on modifying an existing model to use DeepSpeed. Also we will heavily rely on the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest/) and [config JSON documentation](https://www.deepspeed.ai/docs/config-json/) going forward.

Please install DeepSpeed via `pip install deepspeed` if you haven't already done so, after installing you can check if your current version and other information via `ds_report`. For this tutorial we assume a DeepSpeed version of >= 0.5.4 and a torch version >= 1.6. Please upgrade via `pip install --upgrade deepspeed` if you are running an older version of DeepSpeed.

### Add deepspeed.initialize + config

Our first task is to identify where to add `deepspeed.initialize()` to the existing code in order to use the DeepSpeed training engine. Please see the [deepspeed.initialize API documentation](https://deepspeed.readthedocs.io/en/latest/initialize.html#training-initialization) for more details. This needs to be done after the model has been created and before the training loop has started. Most of our edits will be inside the `train` function inside [train_bert.py](./train_bert.py).

After the model is created and before the optimizer is created we want to add the following lines:

```python
ds_config = {
  "train_micro_batch_size_per_gpu": batch_size,
  "optimizer": {
      "type": "Adam",
      "params": {
          "lr": 1e-4
      }
  },
}
model, _, _, _ = deepspeed.initialize(model=model,
                                      model_parameters=model.parameters(),
                                      config=ds_config)
```

This will create the DeepSpeed training engine based on the previously instantiated model and the new `ds_config` dictionary. We can now also remove the previous lines of code that created an Adam optimizer, this will now be done via the DeepSpeed engine. It should be noted, you can optionally created your own optimizer and pass it into `deepspeed.initialize` however DeepSpeed is able to make further performance optimizations by instantiating its own optimizers.

### Update the training-loop

Next we want to update our training-loop to use the new model engine with the following changes:

* `model.to(device)` can be removed
  * DeepSpeed will be careful on when to move the model to GPU to reduce GPU memory usage (e.g., converts to half on CPU then moves to GPU)
* `optimizer.zero_grad()` can be removed
  * DeepSpeed will do this for you at the right time.
* Replace `loss.backward()` with `model.backward(loss)`
  * There are several cases where the engine will properly scale the loss when using certain features (e.g., fp16, gradient-accumulation).
* Replace `optimizer.step()` with `model.step()`
  * The optimizer step is handled by the engine now and is responsible for dispatching to the right optimizer depending on certain features.

### Update checkpoint save and load

Immediately after our new `deepspeed.initialize` you will see a checkpoint load and in the training-loop you will see a few checkpoint save calls. DeepSpeed handles the complexities of checkpoint saving for you so we can simplify these codepaths in the following way. Please refer to the [model checkpoint API documentation](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) for more details.

__Checkpoint saving__: DeepSpeed will construct and save the state_dict for you, we can replace the *two* checkpoint saving snippets (i.e., `state_dict` construction and `torch.save`)  and replace them with the snippet below. The `client_state` being passed in here is an example of state outside the view of DeepSpeed that will be saved with the checkpoint.

```python
model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step})
```

__Checkpoint loading__: The checkpoint loading is happening right before the training-loop starts. It invokes the `load_model_checkpoint` function which consists of around 30 lines of code. We can replace the `load_model_checkpoint(load_checkpoint_dir, model, optimizer)` call with the following snippet:

```python
_, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
checkpoint_step = client_state['checkpoint_step']
```

---
📌 **Note:** You may also want/need to make additional changes to your code if you run on multiple GPUs as DeepSpeed will launch multiple processes. You will want to avoid potential race conditions with creating directories or writing to file and restrict logging to a single process. Take a look at `train_bert_ds.py` for an example of how to do this.

---

## 2.1 Launching Training

We are now ready to launch our training! As a convenience, DeepSpeed provides its own launcher that is seamlessly compatible with clusters that provide a `/job/hostfile` containing all available machines in your job. You can now try running your model on your available GPU(s) with the command below. By default this will attempt to run distributed data-parallel (DDP) training across all available GPUs on the current machine + any external machines listed in your `/job/hostfile`. Please read [more details about the DeepSpeed launcher](https://www.deepspeed.ai/getting-started/#launching-deepspeed-training) and its assumptions on our website.

```bash
deepspeed train_bert.py --checkpoint_dir .
```

---
📌 **Note:** If using the deepspeed launcher you should not pass the `--local_rank` explicitly. This will be done by the launcher in the same way as if you launched with `torch.distributed.launch` from PyTorch.

---

## 2.2 Mixed Precision Training (fp16)

Now that we are setup to use the DeepSpeed engine with our model we can start trying out a few different features of DeepSpeed. One feature is mixed precision training that utilizes half precision (floating-point 16 or fp16) data types. If you want to learn more about how mixed precision training works please refer to the Mixed Precision Training paper [[3]](https://arxiv.org/pdf/1710.03740v3.pdf) from Baidu and NVIDIA on the topic.

To enable this mode in DeepSpeed we need to update our `ds_config` before the engine is created. Please see [fp16 training options](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) in the config documentation for more information. In our case let's simple enable it by adding the following to our `ds_config` dictionary:

```python
  "fp16": {
    "enabled": True
  }
```

The memory reduction by switching from fp32 to fp16 results in the *model parameters* using half the amount of GPU memory, however the overall GPU memory reduction is not as simple. Since fp16 has half the available bits as fp32 it is not able to represent the same expressiveness as fp32, which can result in numeric instabilities during training. We are able to get around these instabilities in most cases by keeping some states in fp16 and others remain in fp32 (see Section 3 in [[3]](https://arxiv.org/pdf/1710.03740v3.pdf) if you'd like to learn more).

The primary reason to utilize fp16 training is due to *Tensor Cores*. If you are training with NVIDIA V100 or A100 GPUs they include Tensor Cores which in some cases can accelerate computation by as much as 8x if certain conditions are met. One of the most important conditions is that your model parameters are stored as fp16. For more details on other conditions and tips to better utilize these cores please see this guide from NVIDIA on [Tips for Optimizing GPU Performance Using Tensor Cores](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/).

---
📌 **Note:** At the start of training you will probably see several log messages about loss scaling and overflows, this is normal. In order for fp16 training to be numerically stable we utilize a common technique called "loss scaling" (similar to Section 3.2 in [[3]](https://arxiv.org/pdf/1710.03740v3.pdf)). This attempts to find a scaling value to mitigate gradient over/under-flows during training.

---

## 2.3 Zero Redundancy Optimizer (ZeRO)

ZeRO leverages the aggregate computation and memory resources of data parallelism to reduce the memory and compute requirements of each device (GPU) used for model training. ZeRO reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs) in the distributed training hardware. Concretely, ZeRO is  implemented as incremental stages of optimizations, where optimizations in earlier stages are available in the later stages. There are 3 different stages of ZeRO, Stage 1: optimizer state partitioning, Stage 2: optimizer state + gradient partitioning, and Stage 3: optimizer state + gradient + weight partitioning. We will focus on two features of ZeRO here, ZeRO Stage 1 and ZeRO-Offload. For further information, please refer to our [tutorial deep diving ZeRO](https://www.deepspeed.ai/tutorials/zero/) and our [tutorial deep diving ZeRO Offload](https://www.deepspeed.ai/tutorials/zero-offload/) on our website. To deep dive into ZeRO, please see our three papers [[4](https://arxiv.org/pdf/1910.02054.pdf), [5](https://www.usenix.org/system/files/atc21-ren-jie.pdf), [6](https://arxiv.org/abs/2104.07857)] that explore different optimizations in this space.

* ZeRO Stage 1: The optimizer states (e.g., for the Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
* ZeRO-Offload: Supports efficiently offloading optimizer memory and computation from the GPU to the host CPU. ZeRO-Offload enables large models with up to 13 billion parameters to be trained on a single GPU.

To enable ZeRO Stage 1 in DeepSpeed we need to again update our `ds_config` before the engine is created. Please see [ZeRO optimizations](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training) in the DeepSpeed config documentation for more information. In our case let's simply enable stage 1 it by adding the following to our `ds_config` dictionary:

```python
  "zero_optimization": {
    "stage": 1
  }
```

We can re-run our training now with ZeRO stage 1 enabled and will see a per-GPU memory reduction as we scale up the total number of GPUs. Typically you can now use this extra GPU memory to either scale up your model size or scale up your per-GPU training batch size. However, if we only have 1 GPU available we probably want to enable ZeRO-Offload to allow us to train larger model sizes. Please update your `ds_config` to include the following:

```python
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
```

This config will now allow us to train a much larger model than we were previously able to do. For example on a single P40 GPU with 24GB of memory we are unable to train a 2 billion parameter model (i.e., `--num_layers 24 --h_dim 4096`), however with ZeRO-Offload we now can!

```bash
deepspeed train_bert.py --checkpoint_dir . --num_layers 24 --h_dim 4096
```

---
📌 **Note:** Earlier on when we setup `deepspeed.initialize` we chose not to explicitly pass an optimizer and instead let the DeepSpeed engine instantiate one for us. This is especially useful now that we are using ZeRO-Offload. DeepSpeed includes a highly optimized version of Adam that executes purely on CPU. This means that DeepSpeed will detect if you are using ZeRO-Offload w. Adam and switch to optimized CPUAdam variant.

---

## References
> <a id="1">[1]</a>
[Vaswani et al. Attention is all you need.
In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)](https://arxiv.org/pdf/1706.03762.pdf)
>
> <a id="2">[2]</a>
[J. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT'19)](https://aclanthology.org/N19-1423.pdf)
>
> <a id="3">[3]</a>
[P. Micikevicius et al. Mixed Precision Training (ICLR'18)](https://arxiv.org/pdf/1710.03740v3.pdf)
>
> <a id="4">[4]></a>
[S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. ZeRO: memory optimizations toward training trillion parameter models. (SC‘20)](https://arxiv.org/pdf/1910.02054.pdf)
>
> <a id="5">[5]</a>
[J. Ren, S. Rajbhandari, R. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, Y. He. ZeRO-Offload: Democratizing Billion-Scale Model Training. (ATC'21)](https://www.usenix.org/system/files/atc21-ren-jie.pdf)
>
> <a id="1">[6]</a>
[S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, Y. He. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning (SC'21)](https://arxiv.org/abs/2104.07857)
