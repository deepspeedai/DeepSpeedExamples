### ☀️Evaluation
We provide a few examples to test the quality of the models.
To run the tests, use the `batch_generation.py` script, which will call the JSON file located in  `eval_data/*.json`.
You will need to specify the model path where you've saved your checkpoints. For example, if you've saved your model checkpoint at $YOUR_CHECKPOINT_PATH/epoch-5/pytorch_model.bin, then pass the following arguments: 
```
--checkpoint_path $YOUR_CHECKPOINT_PATH --checkpoint_names epoch-5
```

##### 🏃 Run the Code
NOTE: Before you run the code `run_batch.sh`, please read it carefully. This bash script creates a folder eval/results/eval_comprehensive if you use the json evaluation "eval_comprehensive". It will write to "eval/results/eval_comprehensive/{args.output_filename}.csv" file with four columns. The generation output is in the last column. Please read one of our examples such as `eval/results/eval_comprehensive/ours-set1_final.csv`.
To run the code, you need to go to outside the current folder
```
cd DeeSpeedExamples/applications/DeepSpeed-VisualChat
bash eval/run_batch.sh
```


#### 🐕 Our Model Results Overview
We present the outcomes from our three distinct models, each trained with vision encoders: `qwen-clip` and `Llama-2-13b-hf`.

###### Results Directories and Training Details:
- **results/eval_single:**  
  This directory contains results from the model trained with LoRA, featuring a dimension size of 128.

- **results/eval_comprehensive** and **results/eval_robustness:**  
  These directories host results from two models:
  - One model is trained excluding the Sparkles dataset (referred to as `ours-set1`).
  - The other incorporates Sparkles dataset in the training (denoted as `ours-set2`).
