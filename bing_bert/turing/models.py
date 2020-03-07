import argparse
import logging
import random
import numpy as np
import os
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, MSELoss

from turing.dataset import BatchType
from turing.utils import TorchTuple

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel #, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainingHeads, PreTrainedBertModel, BertPreTrainingHeads
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from nvidia.modeling import BertForPreTraining, BertConfig

class BertPretrainingLoss(PreTrainedBertModel):
    def __init__(self, bert_encoder, config):
        super(BertPretrainingLoss, self).__init__(config)
        self.bert = bert_encoder
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.cls.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertClassificationLoss(PreTrainedBertModel):
    def __init__(self, bert_encoder, config, num_labels: int = 1):
        super(BertClassificationLoss, self).__init__(config)
        self.bert = bert_encoder
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        scores = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(scores.view(-1, self.num_labels), labels.view(-1,1))
            return loss
        else:
            return scores


class BertRegressionLoss(PreTrainedBertModel):
    def __init__(self, bert_encoder, config):
        super(BertRegressionLoss, self).__init__(config)
        self.bert = bert_encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))
            return loss
        else:
            return logits


class MTLRouting(nn.Module):
    def __init__(self, encoder: BertModel, args, loss_calculation):
        super(MTLRouting, self).__init__()
        #self.bert_encoder = encoder
        self.batch_loss_calculation = loss_calculation
        #self._batch_loss_calculation = nn.ModuleDict()
        #self._batch_counter = {}
        #self._batch_module_name = {}
        # self._batch_loss_calculation = {}
        #self._batch_name = {}
        self.logger = args.logger
        self.args = args

    def register_batch(self, batch_type, module_name, loss_calculation: nn.Module):
        raise NotImplementedError()
        assert isinstance(loss_calculation, nn.Module)
        # print(str(batch_type))
        self._batch_loss_calculation[str(batch_type.value)] = loss_calculation
        self._batch_counter[batch_type] = 0
        self._batch_module_name[batch_type] = module_name

    def log_summary_writer(self, batch_type, logs: dict, base='Train'):
        if (not self.args.no_cuda and dist.get_rank() == 0) or (self.args.no_cuda and self.args.local_rank == -1):
            counter = self._batch_counter[batch_type]
            module_name = self._batch_module_name.get(
                batch_type, self._get_batch_type_error(batch_type))
            for key, log in logs.items():
                self.args.summary_writer.add_scalar(
                    f'{base}/{module_name}/{key}', log, counter)
            self._batch_counter[batch_type] = counter + 1

    def _get_batch_type_error(self, batch_type):
        def f(*args, **kwargs):
            message = f'Misunderstood batch type of {batch_type}'
            self.logger.error(message)
            raise ValueError(message)
        return f

    def forward(self, batch, log=True):
        batch_type = batch[0][0].item()

        # Pretrain Batch
        assert batch_type == BatchType.PRETRAIN_BATCH

        #loss_function = self._batch_loss_calculation[str(batch_type)]
        loss_function = self.batch_loss_calculation

        loss = loss_function(input_ids=batch[1],
                             token_type_ids=batch[3],
                             attention_mask=batch[2],
                             masked_lm_labels=batch[5],
                             next_sentence_label=batch[4])
        #if log:
        #    self.log_summary_writer(
        #        batch_type, logs={'pretrain_loss': loss.item()})
        return loss

        ## QP Batch
        #elif batch_type == BatchType.QP_BATCH:
        #    loss_function = self._batch_loss_calculation[str(batch_type)]
        #    loss = loss_function(input_ids=batch[1],
        #                         token_type_ids=batch[3],
        #                         attention_mask=batch[2],
        #                         labels=batch[4] if not self.args.fp16 or batch[4] is None else batch[4].half())
        #    if batch[4] is not None:
        #        print(f"QP Loss:{loss.item()}")
        #        self.log_summary_writer(batch_type, logs={'qp_loss': loss.item()})
        #    return loss

        ## Ranking Batch
        #elif batch_type == BatchType.RANKING_BATCH:
        #    loss_function = self._batch_loss_calculation[str(batch_type)]
        #    loss = loss_function(input_ids=batch[1],
        #                         token_type_ids=batch[3],
        #                         attention_mask=batch[2],
        #                         labels=batch[4] if not self.args.fp16 else batch[4].half())
        #    self.log_summary_writer(
        #        batch_type, logs={'ranking_loss': loss.item()})
        #    return loss


class BertMultiTask:
    def __init__(self, args):
        self.config = args.config

        if not args.use_pretrain:

            bert_config = BertConfig(**self.config["bert_model_config"])
            bert_config.vocab_size = len(args.tokenizer.vocab)

            # Padding for divisibility by 8
            if bert_config.vocab_size % 8 != 0:
                bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
            print("VOCAB SIZE:", bert_config.vocab_size)

#            self.bert_encoder = BertModel(bert_config)
            self.network = BertForPreTraining(bert_config, args)
        # Use pretrained bert weights
        else:
            self.bert_encoder = BertModel.from_pretrained(self.config['bert_model_file'], cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
            bert_config = self.bert_encoder.config

        #self.network = MTLRouting(self.bert_encoder, args)
        #self.network = MTLRouting(self.bert_encoder, args, loss_calculation=BertPretrainingLoss(self.bert_encoder, bert_config))

        # for param in self.bert_encoder.parameters():
        #     param.required_grad = False
        config_data=self.config['data']

        self.device=args.device
        return

        # QA Dataset
        if config_data["flags"].get("qp_dataset", False):
            self.network.register_batch(BatchType.QP_BATCH, "qa_dataset", loss_calculation=BertClassificationLoss(
                self.bert_encoder, bert_config, 1))

        # Pretrain Dataset
        if config_data["flags"].get("pretrain_dataset", False):
            self.network.register_batch(BatchType.PRETRAIN_BATCH, "pretrain_dataset",
                                        loss_calculation=BertPretrainingLoss(self.bert_encoder, bert_config))

        # Ranking Dataset
        if config_data["flags"].get("ranking_dataset", False):
            self.network.register_batch(BatchType.RANKING_BATCH, "ranking_dataset", loss_calculation=BertRegressionLoss(
                self.bert_encoder, bert_config))

        # self.network = self.network.float()
        # print(f"Bert ID: {id(self.bert_encoder)}  from GPU: {dist.get_rank()}")

    def save(self, filename: str):
        network=self.network.module
        return torch.save(network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(torch.load(model_state_dict, map_location=lambda storage, loc: storage))

    def move_batch(self, batch: TorchTuple, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()
