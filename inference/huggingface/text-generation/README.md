
# DeepSpeed Huggingface Text Generation Examples

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
deepspeed --num_gpus 1 test-opt.py
</pre>

Output:
<pre>
[{'generated_text': 'DeepSpeed is a bit better than i2f. It works and is faster than i2f'}]
</pre>
