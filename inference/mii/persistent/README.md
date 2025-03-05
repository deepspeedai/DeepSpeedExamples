# Persistent Deployment Examples

The `serve.py` script can be used to create an inference server for any of the
[supported models](https://github.com/deepspeedai/DeepSpeed-mii#supported-models).
Provide the HuggingFace model name and tensor-parallelism (use the default
values and run `$ python serve.py` for a single-GPU
[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
deployment):

```shell
$ python serve.py --model "mistralai/Mistral-7B-v0.1" tensor-parallel 1
```

Connect to the persistent deployment and generate text with `client.py`. Provide
the HuggingFace model name, maximum generated tokens, and prompt(s) (or if you
are using the default values, run `$ python client.py`):

```shell
$ python client.py --model "mistralai/Mistral-7B-v0.1" --max-new-tokens 128 --prompts "DeepSpeed is" "Seattle is"
```

Shutdown the persistent deployment with `terminate.py`. Provide the HuggingFace
model name (or if you are using the default values, run `$ python
terminate.py`):

```shell
$ python terminate.py --model "mistralai/Mistral-7B-v0.1
```