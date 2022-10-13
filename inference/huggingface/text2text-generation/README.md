
# DeepSpeed Huggingface Text2Text Generation Examples

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>

# Usage
Examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] test-[model].py</pre>

# Example Output
Command:
<pre>
deepspeed --num_gpus 1 test-t5.py
</pre>

Output:
<pre>
[{'generated_text': 'd review: this is the best cast iron skillet. Great review! Great review! Great'}]
</pre>
