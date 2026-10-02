# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

# The stand-in implements the launcher's CLI contract, including process exit status.
# It never launches training or queries a GPU; real training needs separate device validation.
LAUNCHER = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
config_arg = next(arg for arg in args if arg.startswith('--deepspeed_config='))
config = json.loads(Path(config_arg.split('=', 1)[1]).read_text())
with open(os.environ['CAPTURE'], 'a') as capture:
    capture.write(json.dumps({'args': args, 'config': config}) + '\\n')
mode = 'reflow' if 'reflow' in config['zero_optimization'] else 'zerooffload'
steps = int(args[args.index('--bench_steps') + 1])
scenario = os.environ.get('SCENARIO', '')
if scenario == 'truncated':
    steps -= 1
for step in range(1, steps + 1):
    value = 1.0
    if scenario == 'mismatch' and mode == 'zerooffload':
        value = 2.0
    print(f'2026-10-02 - finetune_zero3 - INFO - BITLOSS step {step} hex={value.hex()} dec={value}')
    label = str(step) if scenario == 'legacy' else f'{step:4d}'
    print(f'2026-10-02 - finetune_zero3 - INFO - Step {label} | Loss: 1.0 | Time: {step * 10}ms | TFLOPS(w/ recompute): 1.0')
if scenario != 'legacy':
    print('2026-10-02 - finetune_zero3 - INFO - Mean step time after 1 warmup steps: 123.45ms')
if scenario == 'failure':
    sys.exit(7)
'''


class LauncherTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='reflow example ')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.example = self.directory / 'example scripts'
        self.example.mkdir()
        for source in ROOT.glob('*.sh'):
            shutil.copy(source, self.example)
        shutil.copy(ROOT / 'finetune_zero3.py', self.example)
        self.bin_dir = self.directory / 'bin'
        self.bin_dir.mkdir()
        launcher = self.bin_dir / 'deepspeed'
        launcher.write_text(LAUNCHER)
        launcher.chmod(0o755)
        # Public nvidia-smi CSV contract used by the comparison sampler.
        nvidia = self.bin_dir / 'nvidia-smi'
        nvidia.write_text('#!/bin/sh\necho 100\n')
        nvidia.chmod(0o755)
        self.capture = self.directory / 'capture.jsonl'
        self.env = dict(os.environ,
                        PATH=str(self.bin_dir) + os.pathsep + os.environ['PATH'],
                        CAPTURE=str(self.capture),
                        OUT_DIR=str(self.directory / 'results'),
                        SAMPLE_SEC='0.01',
                        FORCE='1',
                        ATTN_IMPLEMENTATION='eager')

    def run_script(self, script, *args, **env):
        return subprocess.run(['bash', str(self.example / script), *args],
                              cwd=self.directory,
                              env=dict(self.env, **env),
                              text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              timeout=30)

    def launches(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_all_launchers_generate_matching_cpu_bf16_configs(self):
        # Catch per-rank/global batch confusion, invalid runtime options and mode drift.
        from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
        for script in sorted(self.example.glob('finetune_*.sh')):
            with self.subTest(script=script.name):
                pair = []
                for mode in ('reflow', 'zerooffload'):
                    result = self.run_script(script.name, mode, '16', BENCH_STEPS='100', WARMUP_STEPS='10')
                    self.assertEqual(result.returncode, 0, result.stdout)
                    launch = self.launches()[-1]
                    args, config = launch['args'], launch['config']
                    DeepSpeedZeroConfig(**config['zero_optimization'])
                    gpu_arg = next(arg for arg in args if arg.startswith('--num_gpus='))
                    world_size = int(gpu_arg.split('=')[1])
                    microbatch = int(args[args.index('--batch_size') + 1])
                    self.assertEqual(microbatch * world_size, config['train_batch_size'])
                    self.assertEqual(config['train_batch_size'], 16)
                    self.assertTrue(config['bf16']['enabled'])
                    self.assertEqual(config['gradient_clipping'], 0)
                    self.assertEqual(config['zero_optimization']['offload_optimizer']['device'], 'cpu')
                    self.assertEqual(args[args.index('--bench_steps') + 1], '100')
                    self.assertEqual(args[args.index('--warmup_steps') + 1], '10')
                    self.assertEqual(args[args.index('--attn_implementation') + 1], 'eager')
                    self.assertIn(str(self.example / 'finetune_zero3.py'), args)
                    pair.append(config)
                reflow = pair[0]['zero_optimization'].pop('reflow')
                self.assertEqual(reflow['bucketwise_cores_per_worker'], 8)
                self.assertEqual(pair[0], pair[1])

    def test_invalid_mode_and_global_batch_do_not_launch(self):
        for args in ((), ('unknown', ), ('reflow', '0'), ('reflow', '7'), ('reflow', 'abc')):
            with self.subTest(args=args):
                result = self.run_script('finetune_opt-30b_8gpu.sh', *args)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.capture.exists())

    def test_core_overrides_reach_generated_configuration(self):
        result = self.run_script('finetune_opt-350m-1gpu.sh', 'reflow', WORKER_CORES='4', MAIN_CORES='3')
        self.assertEqual(result.returncode, 0, result.stdout)
        reflow = self.launches()[-1]['config']['zero_optimization']['reflow']
        self.assertEqual(reflow['bucketwise_cores_per_worker'], 4)
        self.assertEqual(reflow['main_thread_cores'], 3)

    def test_bitexact_requires_complete_successful_matching_runs(self):
        for scenario, success in (('', True), ('mismatch', False), ('truncated', False), ('failure', False)):
            with self.subTest(scenario=scenario):
                result = self.run_script('check_bitexact.sh', '8', SCENARIO=scenario)
                self.assertEqual(result.returncode == 0, success, result.stdout)
                self.assertEqual('BIT-IDENTICAL' in result.stdout, success)
        config = self.launches()[0]['config']
        self.assertEqual(config['train_micro_batch_size_per_gpu'], 4)

    def test_compare_propagates_partial_training_failure(self):
        result = self.run_script('run_compare.sh', 'finetune_opt-350m-1gpu.sh', SCENARIO='failure')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('FAILED', result.stdout)
        self.assertNotIn('SPEEDUP', result.stdout)
        self.assertNotIn('cannot open', result.stdout)

    def test_compare_uses_reported_warmup_summary(self):
        result = self.run_script('run_compare.sh', 'finetune_opt-350m-1gpu.sh')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertRegex(result.stdout, r'step time \(ms\)\s+123.45\s+123.45')

    def test_compare_preserves_numeric_step_order_in_legacy_logs(self):
        # Lexicographic sorting drops step 10 as warmup instead of step 1.
        result = self.run_script('run_compare.sh', 'finetune_opt-350m-1gpu.sh', SCENARIO='legacy')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertRegex(result.stdout, r'step time \(ms\)\s+60\s+60')


class PreprocessingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('These checks must run without initializing CUDA')
        spec = importlib.util.spec_from_file_location('reflow_example', ROOT / 'finetune_zero3.py')
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'hello': 2, 'world': 3}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]')

    def test_padding_and_raw_dataset_fields_do_not_enter_training_loss(self):
        # Catch unmasked padded targets and source strings leaking into the model batch.
        import torch
        from datasets import Dataset, DatasetDict
        from transformers import OPTConfig, OPTForCausalLM, default_data_collator
        records = [{'instruction': 'hello', 'input': '', 'output': 'world'}] * 2
        data = DatasetDict(train=Dataset.from_list(records))
        with patch.object(self.example, 'load_dataset', return_value=data):
            processed = self.example.load_and_preprocess_dataset('offline', 100, self.tokenizer, 32,
                                                                 self.example.setup_logger())
        batch = default_data_collator([processed[0], processed[1]])
        self.assertEqual(set(batch), {'input_ids', 'attention_mask', 'labels'})
        padding = batch['attention_mask'] == 0
        self.assertTrue(padding.any())
        self.assertTrue((batch['labels'][padding] == -100).all())
        self.assertTrue(torch.equal(batch['labels'][~padding], batch['input_ids'][~padding]))
        model = OPTForCausalLM(
            OPTConfig(vocab_size=4,
                      hidden_size=16,
                      ffn_dim=32,
                      num_hidden_layers=1,
                      num_attention_heads=2,
                      max_position_embeddings=32,
                      pad_token_id=1))
        output = model(**batch)
        expected = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 4),
                                                     batch['labels'][:, 1:].reshape(-1),
                                                     ignore_index=-100)
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.model.decoder.embed_tokens.weight.grad).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_empty_dataset_selection_is_rejected(self):
        from datasets import Dataset, DatasetDict
        data = DatasetDict(train=Dataset.from_list([{'instruction': 'hello', 'input': '', 'output': 'world'}]))
        with patch.object(self.example, 'load_dataset', return_value=data):
            with self.assertRaises(ValueError):
                self.example.load_and_preprocess_dataset('offline', 1, self.tokenizer, 32, self.example.setup_logger())

    def test_invalid_benchmark_window_is_rejected(self):
        for extra in (['--bench_steps', '0'], ['--bench_steps', '10', '--warmup_steps', '20'], ['--log_interval',
                                                                                                '0']):
            with self.subTest(extra=extra):
                args = self.example.create_argument_parser().parse_args(
                    ['--model_name', 'offline', '--lr', '1e-5', '--batch_size', '1', '--output_dir', 'out'] + extra)
                with self.assertRaises(ValueError):
                    self.example.validate_arguments(args)


if __name__ == '__main__':
    unittest.main()
