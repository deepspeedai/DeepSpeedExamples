
# DeepSpeed Huggingface Fill Mask Examples

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
deepspeed --num_gpus 1 test-roberta.py
</pre>

Output:
<pre>
[{'score': 0.40290409326553345, 'token': 3742, 'token_str': ' Internet', 'sequence': 'The invention of the Internet revolutionized the way we communicate with each other.'}, {'score': 0.20314466953277588, 'token': 7377, 'token_str': ' telephone', 'sequence': 'The invention of the telephone revolutionized the way we communicate with each other.'}, {'score': 0.17653286457061768, 'token': 2888, 'token_str': ' internet', 'sequence': 'The invention of the internet revolutionized the way we communicate with each other.'}, {'score': 0.06900821626186371, 'token': 4368, 'token_str': ' smartphone', 'sequence': 'The invention of the smartphone revolutionized the way we communicate with each other.'}, {'score': 0.03270129859447479, 'token': 3034, 'token_str': ' computer', 'sequence': 'The invention of the computer revolutionized the way we communicate with each other.'}]
</pre>
