# Non-Persistent Pipeline Examples

The `pipeline.py` script can be used to run any of the [supported
models](https://github.com/microsoft/DeepSpeed-mii#supported-models). Provide
the HuggingFace model name, maximum generated tokens, and prompt(s). The
generated responses will be printed in the terminal:

```shell
$ python pipeline.py --model "mistralai/Mistral-7B-v0.1" --max-new-tokens 128 --prompts "DeepSpeed is" "Seattle is"
```

Tensor-parallelism can be controlled using the `deepspeed` launcher and setting
`--num_gpus`:

```shell
$ deepspeed --num_gpus 2 pipeline.py
```

## Model-Specific Examples

For convenience, we also provide a set of scripts to quickly test the MII
Pipeline with some popular text-generation models: 

| Model | Launch command |
|-------|----------------|
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b) | `$ python llama2.py` |
| [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) | `$ python falcon.py` |
| [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | `$ deepspeed --num_gpus 2 mixtral.py` |