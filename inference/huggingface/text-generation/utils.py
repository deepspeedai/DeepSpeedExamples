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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast
from deepspeed.accelerator import get_accelerator

class DSPipeline():
    '''
    Example helper class for comprehending DeepSpeed Meta Tensors, meant to mimic HF pipelines.
    The DSPipeline can run with and without meta tensors.
    '''
    def __init__(self,
                 model_name='bigscience/bloom-3b',
                 dtype=torch.float16,
                 is_meta=True,
                 device=-1,
                 checkpoint_path=None,
                 trust_remote_code=False,
                 ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(get_accelerator().device_name(device))

        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
        self.tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if (is_meta):
            '''When meta tensors enabled, use checkpoints'''
            self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            self.repo_root, self.checkpoints_json = self._generate_json(checkpoint_path)

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=trust_remote_code)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)

        self.model.eval()

        if self.dtype == torch.float16:
            self.model.half()


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


    def _generate_json(self, checkpoint_path=None):
        if checkpoint_path is None:
            repo_root = snapshot_download(self.model_name,
                                      allow_patterns=["*"],
                                      cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                                      ignore_patterns=["*.safetensors"],
                                      local_files_only=False,
                                      revision=None)
        else:
            assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
            repo_root = checkpoint_path

        if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        elif (self.model_name in self.tp_presharded_models):
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"

            with io.open(checkpoints_json, "w", encoding="utf-8") as f:
                file_list = [str(entry).split('/')[-1] for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
                data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
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
                input_tokens[t] = input_tokens[t].to(self.device)

        self.model.to(self.device)

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            # NOTE: Check if Llamma can work w/ **input_tokens
            #       'token_type_ids' kwarg not recognized in Llamma generate function
            outputs = self.model.generate(input_tokens.input_ids, **generate_kwargs)
        else:
            outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs

class Performance():

    def print_perf_stats(latency_set, config, dtype, batch_size, warmup=3):
        # trim warmup queries
        latency_set = list(latency_set)
        latency_set = latency_set[warmup:]
        count = len(latency_set)

        if count > 0:
            latency_set.sort()
            avg = sum(latency_set) / count
            num_layers = getattr(config, "num_layers", config.num_hidden_layers)
            num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
            if dtype == "float16":
                num_bytes = 2
            elif dtype == "float32":
                num_bytes = 4
            else:
                num_bytes = 1
            print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
            print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
            print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * batch_size / 1e12))
