# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Actual OPSD trainer, Engine optimizer, DP/TP and restart integration.

The random tiny Qwen2 fixtures test application mechanics, not pretrained
distillation quality. The independent reference uses dense PyTorch KL on
the same physical TP layout, with local-slice gather backward semantics.
"""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.runtime.rollout import build_rollout
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast

from config import OPSDConfig, TeacherConfig
from data import LeftPaddedPromptCollator, PromptDataset
from teacher import TeacherWrapper
from trainer import OPSDTrainer
from vocab_parallel import VocabShardLayout, prepare_vocab_parallel
from run_vocab_parallel_probe import compare, make_model


def gather(logits, layout):
    if len(layout.ranks) == 1:
        return logits
    widths = [end - start for start, end in layout.ranges]
    padded = F.pad(logits.contiguous(), (0, max(widths) - logits.shape[-1]))
    outputs = [torch.empty_like(padded) for _ in widths]
    dist.all_gather(outputs, padded, group=layout.group)
    return torch.cat([value[..., :width] for value, width in zip(outputs, widths)], -1)


class DenseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, layout):
        ctx.start, ctx.end = layout.start, layout.end
        return gather(logits, layout)

    @staticmethod
    def backward(ctx, gradient):
        return gradient[..., ctx.start:ctx.end].contiguous(), None


def master(parameter):
    value = safe_get_full_fp32_param(parameter)
    return parameter.detach().float() if value is None else value.detach().float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--tp', type=int, choices=[1, 2], required=True)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--phase', choices=['continuous', 'save', 'resume'], default='continuous')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--student-zero', type=int, choices=[0, 2], default=0)
    parser.add_argument('--vocab-size', type=int, default=64)
    parser.add_argument('--empty-mask', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False,
                              timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world % args.tp == 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1781)
    student = make_model(dtype, args.tied, 32, 1, args.vocab_size)
    reference = copy.deepcopy(student)
    teacher_model = make_model(dtype, args.tied, 64, 2, args.vocab_size)
    fixture = out / 'fixture'
    if rank == 0:
        fixture.mkdir()
        teacher_model.save_pretrained(fixture / 'teacher')
        tokens = ['[PAD]', '[EOS]', '[BOS]', '[UNK]'] + ['t' + str(i) for i in range(4, args.vocab_size)]
        backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(args.vocab_size))), unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]',
                                            bos_token='[BOS]', unk_token='[UNK]', padding_side='left')
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        tokenizer.save_pretrained(fixture / 'tokenizer')
        prompts = [' '.join('t' + str(4 + (i + j) % 24) for j in range(2 + i % 5)) for i in range(40)]
        (fixture / 'prompts.jsonl').write_text(''.join(json.dumps({'prompt': p}) + '\n' for p in prompts))
    dist.barrier()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(fixture / 'tokenizer')
    config = {'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2,
              'tensor_parallel': {'autotp_size': args.tp, 'vocab_parallel_lm_head': True},
              'zero_optimization': {'stage': args.student_zero}, 'gradient_clipping': 0.0,
              'zero_allow_untested_optimizer': True,
              'bf16': {'enabled': dtype == torch.bfloat16},
              'optimizer': {'type': 'SGD', 'params': {'lr': 0.05, 'momentum': 0.9}}}
    candidate, *_ = deepspeed.initialize(model=student, model_parameters=student.parameters(), config=config)
    dense, *_ = deepspeed.initialize(model=reference, model_parameters=reference.parameters(), config=config)
    teacher = TeacherWrapper(TeacherConfig(str(fixture / 'teacher'), dtype=args.dtype,
                                           autotp_size=args.tp, offload_to_cpu=False),
                             world_size=world, vocab_parallel=True)
    layout, teacher.vocab_layout = prepare_vocab_parallel(candidate.module, teacher._callable.module,
                                                         tokenizer, tokenizer)
    candidate._opsd_vocab_layout = layout
    reference_layout = VocabShardLayout.from_model(dense.module)
    layout.require_compatible(reference_layout)
    handle = dense.module.get_output_embeddings().register_forward_hook(
        lambda module, inputs, output: DenseGather.apply(output, reference_layout))
    cfg = OPSDConfig.from_dict({'student': {'model_name_or_path': 'random-tiny-Qwen2', 'dtype': args.dtype},
        'teacher': {'model_name_or_path': str(fixture / 'teacher'), 'dtype': args.dtype,
                    'autotp_size': args.tp, 'offload_to_cpu': False},
        'distillation': {'loss_backend': 'vocab_parallel', 'loss_type': 'reverse_kl',
                         'temperature': 0.5, 'chunk_size': 3},
        'rollout': {'max_prompt_length': 16, 'max_response_length': 3},
        'training': {'max_steps': 4 if args.phase == 'save' else 10, 'logging_steps': 1, 'save_steps': 1000},
        'data': {'path': str(fixture / 'prompts.jsonl'), 'shuffle': False}})
    cfg.validate()
    sampler = DistributedSampler(PromptDataset(cfg.data.path, tokenizer, 16), num_replicas=world // args.tp,
                                 rank=dist.get_rank(candidate.data_parallel_group), shuffle=False)
    dataset = sampler.dataset
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=True,
                        collate_fn=LeftPaddedPromptCollator(tokenizer, 16))
    parameters = dict(candidate.module.named_parameters())
    references = dict(dense.module.named_parameters())
    assert parameters.keys() == references.keys()
    initial = {name: master(p).clone() for name, p in parameters.items()}
    arrays, records = {}, []
    identity = id(candidate.module.get_output_embeddings().weight)

    class AuditedTrainer(OPSDTrainer):
        def _student_update(self, input_ids, attention_mask, response_mask, cache):
            # A DP replica with no valid response positions must still join
            # the TP KL collectives and Engine's GAS/DP reductions.
            if args.empty_mask and self.step == 2 and dist.get_rank(candidate.data_parallel_group) == 1:
                response_mask = torch.zeros_like(response_mask)
            assert cache.cpu_logits.device.type == 'cpu' and cache.shape[-1] == layout.local_size
            assert cache.cpu_logits.numel() == input_ids.numel() * layout.local_size
            assert all(p.grad is None for p in teacher._callable.module.parameters())
            assert id(candidate.module.get_output_embeddings().weight) == identity
            assert not getattr(candidate.module.get_output_embeddings(), '_opsd_rollout_gather_active', False)
            dense.train()
            logits = dense(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
            target = gather(cache.cpu_logits.to(device=logits.device, dtype=dtype), layout)[:, :-1].float()
            mask = response_mask[:, 1:].float()
            tau = cfg.distillation.temperature
            lp, lq = F.log_softmax(logits / tau, -1), F.log_softmax(target / tau, -1)
            dense_loss = ((lp.exp() * (lp - lq)).sum(-1) * mask).sum() * tau**2 / mask.sum().clamp_min(1)
            dense.backward(dense_loss)
            gradient_errors = {}
            original_step = candidate.step

            def checked_step(*step_args, **step_kwargs):
                # ZeRO-2 exposes assembled gradients at the GAS boundary.
                # Earlier microsteps retain internal accumulation fragments.
                names = parameters if args.student_zero == 0 or candidate.is_gradient_accumulation_boundary() else []
                for name in names:
                    parameter = parameters[name]
                    actual, expected = safe_get_full_grad(parameter), safe_get_full_grad(references[name])
                    assert actual is not None and expected is not None, name
                    gradient_errors[name] = compare(actual, expected, dtype, 'gradient ' + name)
                    arrays[f'micro{self.step}.grad.{name}'] = actual.detach().float().cpu().numpy().copy()
                    arrays[f'micro{self.step}.reference_grad.{name}'] = expected.detach().float().cpu().numpy().copy()
                return original_step(*step_args, **step_kwargs)

            candidate.step = checked_step
            try:
                result = super()._student_update(input_ids, attention_mask, response_mask, cache)
            finally:
                candidate.step = original_step
            dense.step()
            loss_error = compare(result[0], dense_loss, dtype, 'trainer loss')
            parameter_errors = {name: compare(master(p), master(references[name]), dtype, 'master ' + name)
                                for name, p in parameters.items()}
            records.append({'microstep': self.step, 'global_steps': candidate.global_steps,
                            'response_tokens': result[1], 'input_ids': input_ids.cpu().tolist(),
                            'cache_shape': list(cache.shape), 'cache_bytes': cache.cpu_logits.numel() * 2,
                            'loss_error': loss_error, 'max_gradient_error': max(gradient_errors.values(), default=0.0),
                            'gradients_checked': len(gradient_errors),
                            'max_master_parameter_error': max(parameter_errors.values()),
                            'parameters_checked': len(parameters)})
            return result

    trainer = AuditedTrainer(cfg, candidate, teacher, tokenizer,
                             build_rollout(cfg.rollout, student_engine=candidate, tokenizer=tokenizer), loader)
    checkpoint = Path(args.checkpoint)
    if args.phase == 'resume':
        for engine, name in [(candidate, 'candidate'), (dense, 'reference')]:
            path, state = engine.load_checkpoint(str(checkpoint / name), tag='two-updates')
            assert path and state['opsd_microstep'] == 4 and engine.global_steps == 2
        trainer.step = 4
        # Preserve the actual ordered DP data continuation across processes.
        trainer.dataloader = DataLoader(Subset(dataset, list(iter(sampler))[4:]), batch_size=1,
                                       collate_fn=LeftPaddedPromptCollator(tokenizer, 16))
    trainer.train()
    if args.phase == 'save':
        for engine, name in [(candidate, 'candidate'), (dense, 'reference')]:
            engine.save_checkpoint(str(checkpoint / name), tag='two-updates', client_state={'opsd_microstep': 4})
    for name, parameter in parameters.items():
        arrays['final.' + name] = master(parameter).cpu().numpy()
        arrays['reference_final.' + name] = master(references[name]).cpu().numpy()
        arrays['delta.' + name] = (master(parameter) - initial[name]).cpu().numpy()
    np.savez(out / ('rank-' + str(rank) + '.npz'), **arrays)
    result = {'status': 'PASS', 'scope': 'Random tiny model actual trainer integration', 'rank': rank,
              'world': world, 'tp': args.tp, 'dp': world // args.tp, 'phase': args.phase,
              'optimizer_updates': candidate.global_steps, 'dtype': args.dtype, 'tied': args.tied,
              'student_zero_stage': args.student_zero, 'vocab_size': args.vocab_size,
              'parameters_checked': len(parameters), 'records': records,
              'teacher_zero_stage': teacher._callable.zero_optimization_stage(),
              'teacher_frozen': all(not p.requires_grad and p.grad is None for p in teacher._callable.module.parameters())}
    (out / ('rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2))
    handle.remove()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
