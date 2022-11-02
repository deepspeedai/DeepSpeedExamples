'''
Helper classes and functions for examples
'''

import os
import io
from pathlib import Path
import json
import deepspeed
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class Pipeline():
    '''Example helper class, meant to mimic HF pipelines'''
    def __init__(self,
                 model_name='bigscience/bloom-3b',
                 dtype=torch.float16,
                 is_meta=True
                 ):
        self.model_name = model_name
        self.dtype = dtype

        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
        self.tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if (is_meta):
            '''When meta tensors enabled, use checkpoints'''
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.repo_root, self.checkpoints_json = self._generate_json()

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.model.eval()


    def __call__(self,
                inputs=["test"],
                num_tokens=100,
                do_sample=False):
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        outputs = self.generate_outputs(input_list, num_tokens=num_tokens, do_sample=do_sample)
        return outputs


    def _generate_json(self):
        repo_root = snapshot_download(self.model_name, allow_patterns=["*"], local_files_only=False, revision=None)

        if (self.model_name in self.tp_presharded_models):
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"

            with io.open(checkpoints_json, "w", encoding="utf-8") as f:
                file_list = [str(entry) for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
                data = {"type": self.config.model_type, "checkpoints": file_list, "version": 1.0}
                json.dump(data, f)

        return repo_root, checkpoints_json


    def generate_outputs(self,
                         inputs=["test"],
                         num_tokens=100,
                         do_sample=False):
        generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=do_sample)

        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

        outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs
