# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure actual online OPSD or replay of previously generated responses.

The upstream baseline imports a separately frozen, unchanged Examples tree.
The gathered reference retains candidate head layouts, gathers teacher
logits before D2H, and uses dense KL with local-slice gather backward.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.utils import groups, safe_get_full_fp32_param


def full_logits(local, layout):
    if len(layout.ranks) == 1:
        return local
    widths = [end - start for start, end in layout.ranges]
    padded = F.pad(local.contiguous(), (0, max(widths) - local.shape[-1]))
    values = [torch.empty_like(padded) for _ in widths]
    dist.all_gather(values, padded, group=layout.group)
    return torch.cat([value[..., :width] for value, width in zip(values, widths)], -1)


class DenseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, layout):
        ctx.start, ctx.end = layout.start, layout.end
        return full_logits(local, layout)

    @staticmethod
    def backward(ctx, gradient):
        return gradient[..., ctx.start:ctx.end].contiguous(), None


def prepare_benchmark_inputs(implementation, config_path, output_jsonl):
    with Path(config_path).open() as source:
        config = json.load(source)
    sharded = implementation in ('vocab_parallel', 'gathered_reference')
    expected_backend = 'vocab_parallel' if sharded else 'dense'
    backend = config.get('distillation', {}).get('loss_backend', 'dense')
    if backend != expected_backend:
        raise ValueError(f'{implementation} requires distillation.loss_backend={expected_backend!r}; got {backend!r}')
    with Path(config['deepspeed_config']).open() as source:
        ds_config = json.load(source)
    if (implementation == 'dense_default'
            and ds_config.get('tensor_parallel', {}).get('vocab_parallel_lm_head') is True):
        raise ValueError('dense_default requires full-vocabulary heads; disable vocab_parallel_lm_head')
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fail before model initialization if the result path cannot be written.
    with output.open('a'):
        pass


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['online', 'replay'], required=True)
    parser.add_argument('--implementation', choices=['upstream_dense', 'dense_default', 'gathered_reference', 'vocab_parallel'],
                        required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fixture', required=True, help='Per-rank real-rollout fixture directory')
    parser.add_argument('--upstream-opsd-dir')
    parser.add_argument('--warmup-updates', type=int, default=5)
    parser.add_argument('--measure-updates', type=int, default=30)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--save-checkpoint')
    parser.add_argument('--sdpa-backend', choices=['default', 'math'], default='default',
                        help='Use the same PyTorch SDPA backend for every measured implementation')
    parser.add_argument('--profile', action='store_true', help='Additional phase synchronization; separate from throughput')
    args = parser.parse_args()
    try:
        prepare_benchmark_inputs(args.implementation, args.config, args.output_jsonl)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.implementation == 'upstream_dense':
        if not args.upstream_opsd_dir:
            parser.error('upstream_dense needs the unchanged frozen upstream OPSD directory')
        sys.path.insert(0, args.upstream_opsd_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import main as application
    from teacher import TeacherLogitCache
    from losses import streamed_distillation_loss
    initialization_overrides = []
    original_initialize = application.deepspeed.initialize
    if args.implementation == 'upstream_dense':
        def full_vocabulary_initialize(*parameters, **keywords):
            config = keywords.get('config')
            if isinstance(config, dict) and config.get('tensor_parallel', {}).get('autotp_size', 1) > 1:
                config = dict(config)
                config['tensor_parallel'] = dict(config['tensor_parallel'], vocab_parallel_lm_head=False)
                keywords['config'] = config
                initialization_overrides.append(config['tensor_parallel'])
            return original_initialize(*parameters, **keywords)
        # The unchanged upstream TeacherWrapper does not expose a head flag.
        # Set the documented full-logits configuration on both initializers;
        # preserve its model, cache, loss, backward and optimizer code.
        application.deepspeed.initialize = full_vocabulary_initialize

    class MeasuredTrainer(application.OPSDTrainer):
        def __init__(self, *parameters, **keywords):
            super().__init__(*parameters, **keywords)
            self.rank = dist.get_rank()
            self.tp_rank = groups.get_tensor_model_parallel_rank()
            self.samples, self.fixture_rows, self.measured = [], [], []
            self.elapsed = 0.0
            self.tokens = 0
            self.initial_updates = self.student_engine.global_steps
            self.fixture_path = Path(args.fixture) / ('rank-' + str(self.rank) + '.pt')
            self.replay = torch.load(self.fixture_path, map_location='cpu', weights_only=True) if args.mode == 'replay' else None
            self.initial = self._parameter_digest()
            self.initial_optimizer = self._optimizer_digest()
            self.initial_skipped = self.student_engine.skipped_steps
            original_forward = self.teacher.forward_to_cache
            self.raw_teacher_forward = original_forward
            original_generate = self.rollout.generate

            def observed_generate(*parameters, **keywords):
                result = original_generate(*parameters, **keywords)
                self.last_rollout = result
                return result

            self.rollout.generate = observed_generate

            def observed_teacher(input_ids, attention_mask):
                if args.mode == 'online':
                    from utils import build_response_mask
                    mask = build_response_mask(self.last_rollout.response_start_idx.to(input_ids.device), attention_mask)
                    self.fixture_rows.append({'input_ids': input_ids.detach().cpu(),
                        'attention_mask': attention_mask.detach().cpu(), 'response_mask': mask.detach().cpu(),
                        'source_microstep': self.step})
                if args.profile:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                if args.implementation == 'gathered_reference':
                    with torch.no_grad():
                        logits = self.teacher._callable(input_ids=input_ids, attention_mask=attention_mask).logits
                        cache = TeacherLogitCache.from_gpu_logits(full_logits(logits, self.teacher.vocab_layout))
                else:
                    cache = original_forward(input_ids, attention_mask)
                if args.profile:
                    torch.cuda.synchronize()
                self.last_teacher = {'host_cache_bytes': cache.cpu_logits.numel() * cache.cpu_logits.element_size(),
                                     'cache_shape': list(cache.shape), 'teacher_profile_s': time.perf_counter() - started}
                return cache

            self.teacher.forward_to_cache = observed_teacher

        @torch.no_grad()
        def _fixed_response_loss(self, fixture):
            self.student_engine.eval()
            ids, attention, mask = [fixture[name].to(self.device)
                                    for name in ['input_ids', 'attention_mask', 'response_mask']]
            logits = self.student_engine(input_ids=ids, attention_mask=attention).logits[:, :-1]
            cache = self.raw_teacher_forward(ids, attention)
            if args.implementation == 'gathered_reference':
                logits = full_logits(logits, self.vocab_layout)
                target = full_logits(cache.cpu_logits.to(logits.device), self.teacher.vocab_layout)
                cache.free()
                cache = TeacherLogitCache.from_gpu_logits(target)
            backend = {}
            if args.implementation == 'vocab_parallel':
                backend = {'loss_backend': 'vocab_parallel', 'vocab_layout': self.vocab_layout}
            try:
                loss = streamed_distillation_loss(logits,
                    lambda start, end: cache.chunk_to_device(start, end, logits.device, logits.dtype), mask[:, 1:],
                    loss_type=self.cfg.distillation.loss_type, temperature=self.cfg.distillation.temperature,
                    chunk_size=self.cfg.distillation.chunk_size, **backend)
            finally:
                cache.free()
            value = loss.detach().clone()
            dist.all_reduce(value)
            value /= dist.get_world_size()
            assert torch.isfinite(value)
            return float(value)

        def _parameter_digest(self):
            digest = hashlib.sha256()
            for name, parameter in self.student_engine.module.named_parameters():
                value = safe_get_full_fp32_param(parameter)
                if value is None:
                    value = parameter.detach().float()
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        def _optimizer_digest(self):
            # Unfused low-precision optimizers do not expose their masters
            # through safe_get_full_fp32_param; public param_groups contain
            # the actual FP32 parameters passed to the underlying optimizer.
            digest = hashlib.sha256()
            for group_index, group in enumerate(self.student_engine.optimizer.param_groups):
                for parameter_index, parameter in enumerate(group['params']):
                    if parameter.dtype != torch.float32:
                        raise RuntimeError('This benchmark requires FP32 optimizer master parameters')
                    digest.update(str((group_index, parameter_index, tuple(parameter.shape))).encode())
                    digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        def _student_update(self, input_ids, attention_mask, response_mask, cache):
            if args.implementation in ('vocab_parallel', 'dense_default'):
                return super()._student_update(input_ids, attention_mask, response_mask, cache)
            started = time.perf_counter()
            self.student_engine.train()
            local = self.student_engine(input_ids=input_ids, attention_mask=attention_mask).logits
            if args.implementation == 'gathered_reference':
                layout = self.vocab_layout
                assert local.shape[-1] == layout.local_size and cache.shape == (*local.shape[:2], layout.vocab_size)
                local = DenseGather.apply(local, layout)
            # Replay of A uses the unchanged upstream loss function with its
            # original full logits, next-token shift and Engine backward.
            logits = local[:, :-1]
            mask = response_mask[:, 1:]
            loss = streamed_distillation_loss(logits,
                lambda start, end: cache.chunk_to_device(start, end, logits.device, logits.dtype), mask,
                loss_type=self.cfg.distillation.loss_type, temperature=self.cfg.distillation.temperature,
                chunk_size=self.cfg.distillation.chunk_size)
            self.student_engine.backward(loss)
            self.student_engine.step()
            return loss, int(mask.sum().item()), time.perf_counter() - started

        def _train_step(self, batch):
            gas = self.cfg.training.gradient_accumulation_steps
            if self.step % gas == 0:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.elapsed, self.tokens = 0.0, 0
            torch.cuda.synchronize()
            started = time.perf_counter()
            if args.mode == 'online':
                metrics = super()._train_step(batch)
            else:
                fixture = self.replay[self.step]
                ids, attention, mask = [fixture[name].to(self.device)
                                        for name in ['input_ids', 'attention_mask', 'response_mask']]
                if getattr(self, 'vocab_layout', None) is not None:
                    from vocab_parallel import validate_tp_batch
                    validate_tp_batch(self.vocab_layout, ids, attention, mask)
                cache = self.teacher.forward_to_cache(ids, attention)
                try:
                    loss, tokens, student_s = self._student_update(ids, attention, mask, cache)
                finally:
                    cache.free()
                averaged = loss.detach().clone()
                dist.all_reduce(averaged)
                averaged /= dist.get_world_size()
                metrics = {'loss': float(averaged.item()), 'response_tokens': tokens, 'rollout_s': 0.0,
                           'teacher_s': self.last_teacher['teacher_profile_s'], 'student_s': student_s, 'step_s': 0.0}
            torch.cuda.synchronize()
            self.elapsed += time.perf_counter() - started
            self.tokens += metrics['response_tokens'] if self.tp_rank == 0 else 0
            assert torch.isfinite(torch.tensor(metrics['loss']))
            if (self.step + 1) % gas == 0:
                assert self.student_engine.skipped_steps == self.initial_skipped, 'Optimizer skipped an update'
                update = self.student_engine.global_steps - self.initial_updates
                row = {'rank': self.rank, 'mode': args.mode, 'implementation': args.implementation,
                       'update': update, 'optimizer_skipped_steps': self.student_engine.skipped_steps - self.initial_skipped,
                       'sdpa_backend': args.sdpa_backend, 'loss': metrics['loss'], 'elapsed_s': self.elapsed,
                       'unique_response_tokens': self.tokens,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(), **self.last_teacher}
                ranks = [None] * dist.get_world_size()
                dist.all_gather_object(ranks, row)
                if update > args.warmup_updates:
                    self.measured.append(row)
                    if self.rank == 0:
                        elapsed = max(item['elapsed_s'] for item in ranks)
                        tokens = sum(item['unique_response_tokens'] for item in ranks)
                        summary = {'update': update, 'elapsed_s': elapsed, 'unique_response_tokens': tokens,
                                   'tokens_per_s': tokens / elapsed, 'ranks': ranks,
                                   'mode': args.mode, 'implementation': args.implementation,
                                   'profile': args.profile}
                        with Path(args.output_jsonl).open('a') as output:
                            output.write(json.dumps(summary) + '\n')
            return metrics

        def train(self):
            expected = args.warmup_updates + args.measure_updates
            assert self.cfg.training.max_steps == expected * self.cfg.training.gradient_accumulation_steps
            if args.mode == 'replay':
                fixed_response = self.replay[0]
            else:
                from contextlib import nullcontext
                from deepspeed.runtime.rollout import RolloutRequest, SamplingConfig
                from utils import build_response_mask
                batch = next(iter(self.dataloader))
                self.student_engine.eval()
                layout = getattr(self, 'vocab_layout', None)
                scope = nullcontext()
                if layout is not None:
                    from vocab_parallel import gathered_rollout_logits
                    scope = gathered_rollout_logits(self.student_engine.module, layout)
                with torch.no_grad(), scope:
                    response = self.rollout.generate(RolloutRequest(batch['prompt_ids'].to(self.device),
                        batch['prompt_attention_mask'].to(self.device)),
                        SamplingConfig(max_new_tokens=self.cfg.rollout.max_response_length, temperature=0.0))
                fixed_response = {'input_ids': response.input_ids.detach().cpu(),
                            'attention_mask': response.attention_mask.detach().cpu(),
                            'response_mask': build_response_mask(response.response_start_idx,
                                                                response.attention_mask).detach().cpu()}
            fixed_response_initial = self._fixed_response_loss(fixed_response)
            super().train()
            assert self.student_engine.global_steps - self.initial_updates == expected
            assert len(self.measured) == args.measure_updates
            final = self._parameter_digest()
            final_optimizer = self._optimizer_digest()
            fixed_response_final = self._fixed_response_loss(fixed_response)
            assert final_optimizer != self.initial_optimizer, 'No actual optimizer master update'
            assert self.student_engine.skipped_steps == self.initial_skipped
            assert all(not p.requires_grad and p.grad is None for p in self.teacher._callable.module.parameters())
            if args.mode == 'online':
                self.fixture_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.fixture_rows, self.fixture_path)
            if args.save_checkpoint:
                self.student_engine.save_checkpoint(args.save_checkpoint, tag='completed',
                    client_state={'opsd_microstep': self.step, 'optimizer_updates': expected})
            resolved = {'status': 'PASS', 'rank': self.rank, 'mode': args.mode, 'implementation': args.implementation,
                        'optimizer_updates': expected, 'measured_updates': len(self.measured),
                        'initial_parameter_sha256': self.initial, 'final_parameter_sha256': final,
                        'config': self.cfg.to_dict(), 'fixture_sha256': hashlib.sha256(self.fixture_path.read_bytes()).hexdigest(),
                        'initial_optimizer_sha256': self.initial_optimizer, 'final_optimizer_sha256': final_optimizer,
                        'optimizer_skipped_steps': self.student_engine.skipped_steps - self.initial_skipped,
                        'model_weights_changed': final != self.initial, 'sdpa_backend': args.sdpa_backend,
                        'optimizer_class': type(self.student_engine.optimizer).__name__,
                        'teacher_frozen': True, 'source_dir': str(Path(application.__file__).parent)}
            resolved.update(fixed_response_loss_before=fixed_response_initial, fixed_response_loss_after=fixed_response_final,
                            fixed_response_scope='Initial response reused after training; separate from timed workload, no convergence claim',
                            full_vocabulary_initialization_overrides=initialization_overrides)
            Path(args.output_jsonl + '.rank-' + str(self.rank) + '.json').write_text(json.dumps(resolved, indent=2))

    application.OPSDTrainer = MeasuredTrainer
    sys.argv = ['main.py', '--config', args.config]
    try:
        if args.sdpa_backend == 'math':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                application.main()
        else:
            application.main()
    finally:
        application.deepspeed.initialize = original_initialize
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    run()
