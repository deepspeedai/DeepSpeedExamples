import torch
from domino.arguments import get_args 
from domino.language_model import parallel_lm_logits
from domino.modules.enums import AttnMaskType
from domino.modules.module import DominoModule
from domino.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from domino.language_model import get_language_model


def post_language_model_processing(lm_output, labels, logit_weights, parallel_output):
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)
    labels = labels.transpose(0, 1).contiguous()
    loss = vocab_parallel_cross_entropy(output.float(), labels)
    loss = loss.transpose(0, 1).contiguous()
    return loss


class LLaMAModel(DominoModule):
    """LLaMA Language model."""

    def __init__(
        self,
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
    ):
        args = get_args()
        super(LLaMAModel, self).__init__(
            config=config,
            share_embeddings_and_output_weights=True)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.padded_vocab_size = args.padded_vocab_size
        self.language_model = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self.initialize_word_embeddings()
        self.lm_head = torch.nn.Linear(
            args.hidden_size, args.padded_vocab_size, bias=False
        )

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def _causal_lm_process(self, lm_output, labels):
        lm_output = lm_output.transpose(0, 1)
        logits = self.lm_head(lm_output)
        loss = None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., :-1].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        shift_logits = shift_logits.view(-1, self.padded_vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        retriever_input_ids=None,
        retriever_position_ids=None,
        retriever_attn_mask=None,
        labels=None,
        inference_params=None,
    ):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params,
        )

        if self.post_process:
            return self._causal_lm_process(lm_output=lm_output, labels=labels)
        else:
            return lm_output
