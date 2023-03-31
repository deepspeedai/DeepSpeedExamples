
# DeepSpeed Stable Diffusion Example

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>

# Usage
Examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] test-[model].py</pre>

NOTE: Local CUDA graphs for replaced SD modules will only be enabled when `mp_size==1`.

# Example Output
Command:
<pre>
deepspeed --num_gpus 1 test-stable-diffusion.py
</pre>

Output:
<pre>
./baseline.png
./deepspeed.png
</pre>
