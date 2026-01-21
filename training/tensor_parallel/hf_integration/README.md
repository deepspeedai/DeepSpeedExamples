# tensor parallel example (Hugging Face Trainer + AutoTP)
This project is adapted from https://github.com/tatsu-lab/stanford_alpaca.
It uses Hugging Face `Trainer` with a DeepSpeed config that enables AutoTP via `tensor_parallel.autotp_size`.
We only modified the DeepSpeed config and logging, as an example use case.

**Script**

``` bash run.sh ``` or ```bash run.sh MODE``` 


