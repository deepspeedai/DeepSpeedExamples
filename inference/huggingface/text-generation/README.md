
# DeepSpeed Huggingface Text Generation Examples

# Contents
   * [Setup](#setup)
   * [Inference Test](#inference-test)
        * [Usage](#usage)
        * [Single-batch Example](#single-batch-example)
        * [Multi-batch Example](#multi-batch-example)
        * [`DSPipeline` utility class](#dspipeline-utility-class)
   * [DeepSpeed HuggingFace Compare](#deepspeed-huggingface-compare)

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>
If you are using conda, the following works:
<pre>
conda create -c conda-forge -n deepspeed python=3.10
conda activate deepspeed
pip install -r requirements.txt
deepspeed --num_gpus 1 inference-test.py --model bigscience/bloom-3b --batch_size 2
</pre>

# Inference Test

The script inference-test.py can be used to test DeepSpeed with AutoTP (--num_gpus > 1), kernel injection (--use_kernel), batching (--batch_size > 1), meta tensors (--use_meta), and checkpoints using the DS Pipeline utility class. 

The default setting (no user supplied arguments except --model to inference-test.py) runs the model with deepspeed.init_inference wrapper. However, the exact optimizations will need to be supplied by the user based on the model they want to run.

1. AutoTP will be applied if the --num_gpus > 1. This is inferred through the world_size argument
2. Kernel injection will only happen if user passes --use_kernel as a command-line flag. This option can work in tandem with --num_gpus > 1 for some models
3. Meta tensor feature enables fast loading of checkpoints for large models. 

## Example usage with kernel injection and meta tensor support for a large model
<pre>
deepspeed --num_gpus 1 inference-test.py --model bigscience/bloom-3b --use_meta --use_kernel
</pre>

The above command will offer the best performance for the Bloom-3B model. For other models, users need to experiment with various features to get best performance. 

## Usage
Examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] inference-test.py --model [model name/path] --batch_size [batch] --dtype [data type]
</pre>

## Single-batch Example
Command:
<pre>
deepspeed --num_gpus 1 inference-test.py --model facebook/opt-125m
</pre>

Output:
<pre>
in=DeepSpeed is a machine learning framework                   
out=DeepSpeed is a machine learning framework based on TensorFlow. It was first released in 2015, then improved on 2016, and is now a major addition to the deep learning tools on GitHub.                                                                     
------------------------------------------------------------    
</pre>

## Multi-batch Example
Command:
<pre>
deepspeed --num_gpus 1 inference-test.py --model bigscience/bloom-3b --batch_size 2
</pre>

Output:
<pre>
in=DeepSpeed is a machine learning framework                                 
out=DeepSpeed is a machine learning framework that takes a machine learning algorithm and then uses those algorithms to find out how the user interacts with the environment. The company announced in July 2017 that it is ready for release - in 2018. It has
 been working on deep learning for about 6 years,                                                                                                                         
------------------------------------------------------------                                                                                                                                        
in=He is working on                                                      
out=He is working on the new video game 'Bloodborne's' expansion pack. Check out the trailer here: Bloodborne's expansion pack includes a complete remaster of the original game, including over 120 maps, playable characters, new quests, and the possibility
 to bring Blood
------------------------------------------------------------     
</pre>

## `DSPipeline` utility class
The text-generation examples make use of the [`DSPipeline`](utils.py) utility class, a class that helps with loading DeepSpeed meta tensors and is meant to mimic the Hugging Face transformer pipeline.

The BLOOM model is quite large and the way DeepSpeed loads checkpoints for this model is a little different than other HF models. Specifically, we use meta tensors to initialize the model before loading the weights:

<pre>
with deepspeed.OnDevice(dtype=self.dtype, device="meta"):
</pre>

This reduces the total system/GPU memory needed to load the model across multiple GPUs and makes the checkpoint loading faster.
The DSPipeline class helps to load the model and run inference on it, given these differences.

# DeepSpeed HuggingFace Compare

The ds-hf-compare script can be used to compare the text generated outputs of DeepSpeed with kernel injection and HuggingFace inference of a model with the same parameters on a single GPU.
(p.s. kernel injection will not be used by default and is only enabled when the "--use_kernel" argument is provided.)

## Usage
Examples can be run as follows:
<pre>deepspeed --num_gpus 1 ds-hf-compare.py --model [model name/path] --dtype [data type] --num_inputs [number of test inputs] --print_outputs --use_kernel[enable kernel injection]
</pre>