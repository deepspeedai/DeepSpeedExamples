
# DeepSpeed Huggingface Text Generation Examples

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>

# Usage
Examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] inference_test.py --name [model name/path] --batch_size [batch] --dtype [data type] 
</pre>
# Single-batch Example
Command:
<pre>
deepspeed --num_gpus 1 inference_test.py --name facebook/opt-125m
</pre>

Output:
<pre>
in=DeepSpeed is a machine learning framework                   
out=DeepSpeed is a machine learning framework based on TensorFlow. It was first released in 2015, then improved on 2016, and is now a major addition to the deep learning tools on GitHub.                                                                     
------------------------------------------------------------    
</pre>

# Multi-batch Example
Command:
<pre>
deepspeed --num_gpus 1 inference_test.py --name bigscience/bloom-3b --batch_size 2
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