
# DeepSpeed Huggingface Text Generation Script

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>

# Usage
The [`test-run-generation.py`](./test-run-generation.py) example can be run using [test-gpt.sh](./test-gpt.sh), which serves as an example of how to run the script.
<pre>
deepspeed --num_nodes 1 --num_gpus 1 test-run-generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-xl \
    --sample_input single_query.txt \
    --fp16 \
    --ds-inference
</pre>

# Example Output
Command:
<pre>
deepspeed --num_nodes 1 --num_gpus 1 test-run-generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-xl \
    --sample_input single_query.txt \
    --fp16 \
    --ds-inference
</pre>

Output:
<pre>
=== GENERATED SEQUENCE 1 ===
What is DeepSpeed?

DeepSpeed is a multi-dimensional data compression framework designed to achieve high compression ratio on human readable
</pre>
