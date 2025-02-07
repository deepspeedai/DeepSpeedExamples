
# DeepSpeed Huggingface Inference Examples

# Contents
   * [Setup](#setup)
   * [Usage](#usage)
   * [Additional Resources](#additional-resources)
       * [DeepSpeed Inference](#deepspeed-inference)
       * [Benchmarking](#benchmarking)

# Setup
The Python dependencies for each example are captured in `requirements.txt` in the corresponding ML task directory (e.g. [./text-generation](./text-generation)).

Python dependencies can be installed using:
<pre>
pip install -r requirements.txt
</pre>

For the [`./automatic-speech-recognition/test-wav2vec.py`](./automatic-speech-recognition/test-wav2vec.py) speech model example, you may also need to install the `libsndfile1-dev` generic library:
<pre>
sudo apt-get install libsndfile1-dev
</pre>

# Usage
The DeepSpeed huggingface inference examples are organized into their corresponding ML task directories (e.g. [./text-generation](./text-generation)). Each ML task directory contains a `README.md` and a `requirements.txt`.

| Task | README | requirements |
|:---|:---|:---|
| [`automatic-speech-recognition`](./automatic-speech-recognition/) | [`README`](./automatic-speech-recognition/README.md) | [`requirements`](./automatic-speech-recognition/requirements.txt) |
| [`fill-mask`](./fill-mask/) | [`README`](./fill-mask/README.md) | [`requirements`](./fill-mask/requirements.txt) |
| [`text-generation`](./text-generation/) | [`README`](./text-generation/README.md) | [`requirements`](./text-generation/requirements.txt) |
| [`text-generation/run-generation-script`](./text-generation/run-generation-script/) | [`README`](./text-generation/run-generation-script/README.md) | [`requirements`](./text-generation/run-generation-script/requirements.txt) |
| [`text2text-generation`](./text2text-generation/) | [`README`](./text2text-generation/README.md) | [`requirements`](./text2text-generation/requirements.txt) |
| [`translation`](./translation/) | [`README`](./translation/README.md) | [`requirements`](./translation/requirements.txt) |
| [`stable-diffusion`](./stable-diffusion/) | [`README`](./stable-diffusion/README.md) | [`requirements`](./stable-diffusion/requirements.txt) |

Most examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] test-[model].py</pre>

# Additional Resources
Information about DeepSpeed can be found at the [deepspeed.ai](https://www.deepspeed.ai) website.

## DeepSpeed Inference
Additional information on DeepSpeed inference can be found here:
* [Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)

## Benchmarking
DeepSpeed inference benchmarking can be found in the DeepSpeed repository:
* [DeepSpeed Inference Benchmarking](https://github.com/deepspeedai/DeepSpeed/tree/master/benchmarks/inference)
