# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CUDA KL matrix and uniform distributed rejection before the first loss."""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import deepspeed
import deepspeed.comm as dist
import torch

from losses import streamed_distillation_loss
from teacher import TeacherLogitCache
from vocab_parallel import VocabShardLayout, prepare_vocab_parallel, validate_tp_batch
from run_vocab_parallel_probe import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False, timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    group = dist.new_group([0, 1])
    records = []
    for dtype in [torch.float32, torch.bfloat16]:
        for vocab in [64, 67]:
            sizes = [(vocab + 1) // 2, vocab // 2]
            start = sum(sizes[:rank]);end = start + sizes[rank]
            layout = VocabShardLayout(vocab, start, end, (0, 1), ((0, sizes[0]), (sizes[0], vocab)), group)
            torch.manual_seed(710)
            full = torch.randn(2, 7, vocab, device='cuda', dtype=dtype)
            teacher = torch.randn_like(full)
            mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1]], device='cuda')
            for direction in ['forward_kl', 'reverse_kl']:
                for tau in [0.5, 1.0, 2.0]:
                    for chunk in [1, 3, 12]:
                        student = full[..., start:end].clone().requires_grad_()
                        reference = full.double().detach().requires_grad_()
                        target = teacher.to(torch.bfloat16).double()
                        lp, lq = torch.log_softmax(reference / tau, -1), torch.log_softmax(target / tau, -1)
                        tokens = (lq.exp() * (lq - lp)).sum(-1) if direction == 'forward_kl' else (lp.exp() * (lp - lq)).sum(-1)
                        expected = (tokens * mask).sum() * tau**2 / mask.sum()
                        cache = TeacherLogitCache.from_gpu_logits(teacher[..., start:end], layout=layout)
                        actual = streamed_distillation_loss(student,
                            lambda a, b: cache.chunk_to_device(a, b, student.device, student.dtype), mask,
                            loss_type=direction, temperature=tau, chunk_size=chunk,
                            loss_backend='vocab_parallel', vocab_layout=layout)
                        actual.backward();expected.backward()
                        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
                        tolerance = (1e-6, 1e-5) if dtype == torch.float32 else (2e-4, 1e-2)
                        torch.testing.assert_close(student.grad.double(), reference.grad[..., start:end],
                                                   atol=tolerance[0], rtol=tolerance[1])
                        cache.free()
                        records.append({'dtype': str(dtype), 'vocab': vocab, 'direction': direction,
                                        'temperature': tau, 'chunk': chunk, 'status': 'PASS'})
    # Undefined padded distributions must contribute neither NaN loss nor
    # gradient. The independent oracle excludes padding before log-softmax.
    torch.manual_seed(711)
    full = torch.randn(2, 7, 67, device='cuda')
    teacher = torch.randn_like(full)
    mask = torch.ones(2, 7, device='cuda')
    mask[1, 0] = 0
    full[1, 0] = -torch.inf
    teacher[1, 0] = -torch.inf
    start, end = (0, 34) if rank == 0 else (34, 67)
    layout = VocabShardLayout(67, start, end, (0, 1), ((0, 34), (34, 67)), group)
    student = full[..., start:end].clone().requires_grad_()
    reference = full.double().detach().requires_grad_()
    valid = mask.bool()
    lp = torch.log_softmax(reference[valid], -1)
    lq = torch.log_softmax(teacher.to(torch.bfloat16).double()[valid], -1)
    expected = (lp.exp() * (lp - lq)).sum(-1).mean()
    cache = TeacherLogitCache.from_gpu_logits(teacher[..., start:end], layout=layout)
    actual = streamed_distillation_loss(student, lambda a, b: cache.chunk_to_device(a, b, student.device),
                                       mask, chunk_size=3, loss_backend='vocab_parallel', vocab_layout=layout)
    actual.backward();expected.backward()
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(student.grad.double(), reference.grad[..., start:end], atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(student.grad[1, 0]) == 0
    cache.free()
    records.append({'case': 'masked_undefined_distribution', 'status': 'PASS'})
    torch.manual_seed(1781)
    model = make_model(torch.float32, False, 32, 1)
    engine, *_ = deepspeed.initialize(model=model, config={
        'train_micro_batch_size_per_gpu': 1, 'tensor_parallel': {'autotp_size': 2, 'vocab_parallel_lm_head': True}})
    head = engine.module.get_output_embeddings()
    original_end = head.vocab_end_index
    if rank == 1:
        head.vocab_end_index -= 1
    rejected = None
    try:
        VocabShardLayout.from_model(engine.module)
    except ValueError as failure:
        rejected = str(failure)
    assert rejected and 'Vocabulary' in rejected
    head.vocab_end_index = original_end
    # Malformed local metadata must fail on both peers without a timeout.
    if rank == 1:
        del head.vocab_end_index
    try:
        VocabShardLayout.from_model(engine.module)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed head was accepted')
    head.vocab_end_index = original_end
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 0, 'b': 1}, get_added_vocab=lambda: {}, pad_token_id=0)
    different = SimpleNamespace(get_vocab=lambda: {'a': 1, 'b': 0} if rank else {'a': 0, 'b': 1},
                                get_added_vocab=lambda: {}, pad_token_id=0)
    try:
        prepare_vocab_parallel(engine.module, engine.module, tokenizer, different)
    except ValueError as failure:
        assert 'semantics' in str(failure)
    else:
        raise AssertionError('Tokenizer mismatch was accepted')
    layout = VocabShardLayout.from_model(engine.module)
    try:
        ids = torch.tensor([[rank + 1, 3]], device='cuda')
        validate_tp_batch(layout, ids, torch.ones_like(ids), torch.ones_like(ids))
    except ValueError:
        pass
    else:
        raise AssertionError('Different TP prompts were accepted')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    (out / ('rank-' + str(rank) + '.json')).write_text(json.dumps({
        'status': 'PASS', 'numeric_cases': records, 'uniform_negative_cases': 4,
        'scope': 'Actual CUDA PR1 KL plus real AutoTP head preflight; controlled errors, no watchdog success'}, indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':
    main()
