# HumanEval Evaluation Script for DeepSpeed-FastGen

## DISCLAIMER

This human-eval evaluation will execute untrusted model-generated code. As per the OpenAI warning, we
strongly recommend you sandbox your environment as described in the [human-eval paper](https://arxiv.org/pdf/2107.03374.pdf).

## Setup

Running the human-eval evaluation requires installation of `human_eval` with the execution code enabled,
which requires local changes to `execution.py`. The following steps will setup `human-eval` for execution:

```bash
git clone https://github.com/openai/human-eval.git
sed -i '/exec(check_program, exec_globals)/ s/^# //' human-eval/human_eval/execution.py
cd human-eval
python -m pip install -e .
```

This evaluation also requires the installation of DeepSpeed-MII:

```bash
python -m pip install deepspeed-mii
```

Additional DeepSpeed-MII installation details can be found [here](https://github.com/deepspeedai/DeepSpeed-MII#installation).

## Run the Evaluation

The following command shows how to run a benchmark using the `codellama/CodeLlama-7b-Python-hf` model:

```bash
python run_human_eval.py --model codellama/CodeLlama-7b-Python-hf --max-tokens 512 --num-samples-per-task 20
```

## Run Evaluation on Samples

Once samples have been generated, they can be evaluated independently using the `evaluate_functional_correctness` command.
For example, the following command will evaluate `mii_samples.jsonl`:

```bash
evaluate_functional_correctness mii_samples.jsonl
```

The evaluation results will be saved to `mii_samples.jsonl_results.jsonl`.
