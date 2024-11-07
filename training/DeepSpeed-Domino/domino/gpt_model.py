from domino.modules.module import DominoModule
from domino.modules.enums import AttnMaskType
from domino.language_model import parallel_lm_logits
from domino.language_model import get_language_model
from domino.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy


def post_language_model_processing(lm_output, labels, logit_weights, parallel_output):
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)
    labels = labels.transpose(0, 1).contiguous()
    loss = vocab_parallel_cross_entropy(output.float(), labels)
    loss = loss.transpose(0, 1).contiguous()
    return loss


class GPTModel(DominoModule):
    def __init__(
        self,
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
    ):
        super().__init__(config=config)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
        inference_params=None,
    ):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params,
        )

        if self.post_process:
            return post_language_model_processing(
                lm_output,
                labels,
                self.shared_embedding_or_output_weight(),
                self.parallel_output,
            )
        else:
            return lm_output
