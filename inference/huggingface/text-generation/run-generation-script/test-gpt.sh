deepspeed --num_nodes 1 --num_gpus 1 test-run-generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-xl \
    --sample_input single_query.txt \
    --fp16 \
    --ds-inference
