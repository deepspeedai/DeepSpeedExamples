
# DeepSpeed Huggingface Translation Examples

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
deepspeed --num_gpus 1 test-t5-base.py
</pre>

Output:
<pre>
[{'translation_text': 'Le renard brun rapide saute au-dessus du chien lazy.'}]
</pre>
