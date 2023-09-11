import torch
import time

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12


global torch_linear_init_backup
global torch_layer_norm_init_backup

def get_quant_config(model_config, bits: int, group_size: int):
    qaunt_config = {
        'weight_quantization': {
            'quantized_initialization' : {
                'num_bits': bits,
                'group_size': group_size,
                "group_dim": 1,
                "symmetric": False
            }
        }
    }

    return qaunt_config

def model_bytes(config):
    h = config.hidden_size
    return 	2 * (config.num_hidden_layers * (
    # config-attention
    h * (3 * h + 1) + h * (h + 1) +
    # mlp
    h * (4 * h + 1) + h * 4 * (h + 1) +
    # layer norm
    h * 4) +
    # embedding
    config.vocab_size * (h + 1))

def cache_bytes(config, batch_size, seq_len):
    return 2 * batch_size * seq_len * config.num_hidden_layers * config.hidden_size * 2

def hidden_bytes(config, batch_size, seq_len):
    return batch_size * seq_len * config.hidden_size * 2



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda config: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda config: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)

def disable_hf_bloom_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.bloom.modeling_bloom.BloomPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str

def get_filename(model_name, batch_size, prompt_len, gen_len,
                 cpu_offload, disk_offload, num_nodes, num_gpus_per_node,
                 kv_offload, weight_quantize):
    simple_name = model_name.split('/')[-1]
    filename = "ds-"
    filename += f"{simple_name}-bs{batch_size}-prompt{prompt_len}-gen{gen_len}-"
    filename += f"n{num_nodes}x{num_gpus_per_node}-"
    if cpu_offload:
        filename += "cpu"
    elif disk_offload:
        filename += "disk"
    else:
        filename += "gpu"
    if kv_offload:
        filename += "-kv_offload"
    if weight_quantize:
        filename += "-w_quant"
        
    return filename


def realize_meta_module(module, dtype=None, device=None):
    for name, child in module.named_children():
        realize_meta_module(child, dtype, device)

    keys = list(module._parameters.keys())
    for k in keys:
        v = module._parameters[k]
        if v is not None:
            module._parameters[k] = torch.nn.Parameter(
                torch.empty(*v.shape, dtype=dtype or v.dtype,
                    device=device or v.device))

    keys = list(module._buffers.keys())
    for k in keys:
        v = module._buffers[k]
        assert v is None


def meta_to_cpu(container, dtype=None):
    if isinstance(container, torch.Tensor):
        return torch.empty(*container.shape, dtype=dtype or container.dtype)
    elif isinstance(container, tuple):
        return tuple(meta_to_cpu(x, dtype) for x in container)
    elif isinstance(container, dict):
        return dict((k, meta_to_cpu(v, dtype)) for k, v in container.items())
    else:
        raise ValueError(f"Invalid type: {container}")

# add timing hooks
def add_model_hooks(model: torch.nn.Module):

    def start_time_hook(module, input):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__start_time__ = time.time()

    def end_time_hook(module, input, output):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__duration__ = time.time() - module.__start_time__
            module.stage = "decode"

    if not hasattr(model, '__start_time_hook_handle'):
        model.__start_time_hook_handle__ = model.register_forward_pre_hook(
            start_time_hook, )

    if not hasattr(model, '__end_time_hook_handle__'):
        model.__end_time_hook_handle__ = model.register_forward_hook(
            end_time_hook, )
