from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from jiwer import wer
import os
import deepspeed
from deepspeed import module_inject
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer
from deepspeed.accelerator import get_accelerator

librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
device = torch.device(get_accelerator().device_name(local_rank))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

model = deepspeed.init_inference(model,
                                 mp_size=world_size,
                                 dtype=torch.float,
                                 injection_policy={Wav2Vec2EncoderLayer: ('attention.out_proj','feed_forward.output_dense')},
                                 replace_with_kernel_inject=False)
model.to(device)
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

librispeech_eval = librispeech_eval.map(map_to_array)

def map_to_pred(batch):
    input_values = processor(batch["speech"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

print("WER:", wer(result["text"], result["transcription"]))
