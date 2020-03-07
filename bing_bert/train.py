import sys
import logging
import pdb
import numpy as np
import random
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import argparse
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import time

from apex import amp
from timer import ThroughputTimer as tt
from turing.logger import Logger
from turing.utils import get_sample_writer
from turing.models import BertMultiTask
from turing.sources import PretrainingDataCreator, TokenInstance, WikiNBookCorpusPretrainingDataCreator, CleanBodyDataCreator
from turing.sources import WikiPretrainingDataCreator
from turing.dataset import QADataset, RankingDataset, PreTrainingDataset, QAFinetuningDataset
from turing.dataset import QABatch, RankingBatch, PretrainBatch, PretrainDataType
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear, warmup_linear_decay_exp
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from utils import get_argument_parser, is_time_to_exit
global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0

def checkpoint_model(PATH, model, optimizer, epoch, last_global_step, last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    logging.info(f"Checkpoint path = {PATH}")
    checkpoint_state_dict = {'epoch': epoch,
                             'last_global_step': last_global_step,
                             'model_state_dict': model.network.module.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'last_global_data_samples': last_global_data_samples}
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)
    torch.save(checkpoint_state_dict, PATH)
    return


def load_training_checkpoint(args, model, optimizer, PATH, load_optimizer_state):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    logger = args.logger
    checkpoint_state_dict = torch.load(PATH, map_location=torch.device("cpu"))
    model.network.module.load_state_dict(
        checkpoint_state_dict['model_state_dict'])
    if load_optimizer_state:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer_state_dict'])
    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict['last_global_data_samples']
    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples)

def get_effective_batch(args, total):
    if args.local_rank != -1:
        return total//dist.get_world_size()//args.train_batch_size//args.gradient_accumulation_steps//args.refresh_bucket_size
    else:
        return total//args.train_batch_size//args.gradient_accumulation_steps//args.refresh_bucket_size


def get_dataloader(args, dataset: Dataset, eval_set=False):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in DataLoader(dataset, batch_size=args.train_batch_size//2 if eval_set else args.train_batch_size, sampler=train_sampler, num_workers=args.config['training']['num_workers']))


def pretrain_validation(args, index, model):
    config = args.config
    logger = args.logger

    model.eval()
    dataset = PreTrainingDataset(args.tokenizer, config['validation']['path'], args.logger,
                                 args.max_seq_length, index, PretrainDataType.VALIDATION, args.max_predictions_per_seq)
    data_batches = get_dataloader(args, dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0
    for batch in tqdm(data_batches):
        batch = tuple(t.to(args.device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Validation/Loss', eval_loss, index+1)
    return


def evaluate_tp1pp_set(args, model, eval_file):
    dataset = QAFinetuningDataset(
        args.tokenizer, eval_file, args.logger, args.max_seq_length)
    data_batches = DataLoader(
        dataset, batch_size=args.train_batch_size, drop_last=False)
    scores = []
    labels = []
    for batch in tqdm(data_batches):
        batch_labels = batch[4].view(-1).tolist()
        # This is important for the model to return scores and not the loss
        batch[4] = None
        batch = tuple(t.to(args.device) if t is not None else t for t in batch)
        batch_scores = model.network(batch).view(-1).tolist()
        labels.extend(batch_labels)
        scores.extend(batch_scores)

    precision, recall, threshold = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    eval_auc = auc(fpr, tpr)
    args.logger.info(f"TP1++ evaluation file: {eval_file}; AUC: {eval_auc}")
    return eval_auc


def eval_tp1pp(args, index, model):
    model.eval()
    tp1pp_evaluations = {}
    finetune_sets = args.config["data"]["tp1pp_evalsets"]
    for key, data_path in finetune_sets.items():
        tp1pp_evaluations[key] = evaluate_tp1pp_set(args, model, data_path)
    return eval_tp1pp

    #! Need to add to logs

def master_process(args):
    return (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1)

def get_train_dataset(args, index, finetune=False, shuffle=True):
    i = 0
    dataloaders = {}
    datalengths = []
    batchs_per_dataset = []
    batch_mapping = {}

    config = args.config
    dataset_paths = config["data"]["datasets"]
    dataset_flags = config["data"]["flags"]

    if finetune:
        qp_finetune_dataset = QAFinetuningDataset(
            args.tokenizer, dataset_paths["qp_finetuning_dataset"], args.logger, args.max_seq_length)
        datalengths.append(len(qp_finetune_dataset))
        dataloaders[i] = get_dataloader(args, qp_finetune_dataset)
        batch_mapping[i] = QABatch
        batchs_per_dataset.append(
            get_effective_batch(args, len(qp_finetune_dataset)))
        i += 1

    else:
        # QP dataset
        if dataset_flags.get("qp_dataset", False):
            qp_dataset = QADataset(
                args.tokenizer, dataset_paths["qp_dataset"], args.logger, args.max_seq_length, index)
            datalengths.append(len(qp_dataset))
            dataloaders[i] = get_dataloader(args, qp_dataset)
            batch_mapping[i] = QABatch
            batchs_per_dataset.append(get_effective_batch(args, len(qp_dataset)))
            i += 1

        # Pretraining dataset
        if dataset_flags.get("pretrain_dataset", False):
            pretrain_type = dataset_flags.get("pretrain_type")

            # CLEAN BODY Data Load
            if pretrain_type == "clean_body":
                cb_pretrain_dataset = PreTrainingDataset(
                    args.tokenizer, dataset_paths['cb_pretrain_dataset'], args.logger, args.max_seq_length, index, PretrainDataType.NUMPY)
                datalengths.append(len(cb_pretrain_dataset))
                dataloaders[i] = get_dataloader(args, cb_pretrain_dataset)
                batch_mapping[i] = PretrainBatch
                batchs_per_dataset.append(
                    get_effective_batch(args, len(cb_pretrain_dataset)))
                i += 1

            elif pretrain_type == "wiki_bc":
                # Load Wiki Dataset
                wiki_pretrain_dataset = PreTrainingDataset(
                    args.tokenizer,
                    dataset_paths['wiki_pretrain_dataset'],
                    args.logger,
                    args.max_seq_length,
                    index,
                    PretrainDataType.NUMPY,
                    args.max_predictions_per_seq)
                datalengths.append(len(wiki_pretrain_dataset))
                dataloaders[i] = get_dataloader(args, wiki_pretrain_dataset)
                batch_mapping[i] = PretrainBatch
                batchs_per_dataset.append(
                    get_effective_batch(args, len(wiki_pretrain_dataset)))
                i += 1

                bc_pretrain_dataset = PreTrainingDataset(
                    args.tokenizer,
                    dataset_paths['bc_pretrain_dataset'],
                    args.logger,
                    args.max_seq_length,
                    index,
                    PretrainDataType.NUMPY,
                    args.max_predictions_per_seq
                )
                datalengths.append(len(bc_pretrain_dataset))
                dataloaders[i] = get_dataloader(args, bc_pretrain_dataset)
                batch_mapping[i] = PretrainBatch
                batchs_per_dataset.append(
                    get_effective_batch(args, len(bc_pretrain_dataset)))
                i += 1

        # Ranking Dataset
        if dataset_flags.get("ranking_dataset", False):
            ranking_dataset = RankingDataset(
                args.tokenizer, dataset_paths['ranking_dataset'], args.logger, args.max_seq_length, index, args.fp16)
            datalengths.append(len(ranking_dataset))
            dataloaders[i] = get_dataloader(args, ranking_dataset)
            batch_mapping[i] = RankingBatch
            batchs_per_dataset.append(
                get_effective_batch(args, len(ranking_dataset)))
            i += 1

    dataset_batches = []
    for i, batch_count in enumerate(batchs_per_dataset):
        dataset_batches.extend([i] * batch_count)

    # shuffle
    if shuffle:
        random.shuffle(dataset_batches)

    dataset_picker = []
    for dataset_batch_type in dataset_batches:
        dataset_picker.extend([dataset_batch_type] *
                              args.gradient_accumulation_steps * args.refresh_bucket_size)

    return dataset_picker, dataloaders, sum(datalengths)

def train(args, index, model, optimizer, finetune=False):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    dataset_picker, dataloaders, total_length = get_train_dataset(args, index, finetune)
    current_data_sample_count = global_data_samples
    global_data_samples += total_length
    config = args.config
    logger = args.logger
    print('total_length', total_length, 'global_data_samples', global_data_samples)

    model.train()

    epoch_step = 0
    for step, dataset_type in enumerate(tqdm(dataset_picker, smoothing=1)):
        try:
            batch = next(dataloaders[dataset_type])
            if args.n_gpu == 1:
                batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

            # Calculate forward pass
            loss = model.network(batch)
            unscaled_loss = loss.item()
            current_data_sample_count += (args.train_batch_size * dist.get_world_size())
            if args.n_gpu > 1:
                # this is to average loss for multi-gpu. In DistributedDataParallel
                # setting, we get tuple of losses form all proccesses
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Enabling optimized reduction
            # Reduction only happens in backward if this method is called before
            if args.local_rank != -1 and (step + 1) % args.gradient_accumulation_steps == 0:
                model.network.enable_need_reduction()
            else:
                model.network.disable_need_reduction()
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = update_learning_rate(config, global_step, optimizer)

                report_step_metrics(args, lr_this_step, unscaled_loss, global_step, current_data_sample_count)

                optimizer.step()
                optimizer.zero_grad()

                report_lamb_coefficients(args, optimizer)
                global_step += 1
                epoch_step += 1
        except StopIteration:
            continue

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args,
                           epoch_steps=epoch_step,
                           global_steps=current_global_step):
            print(f'Warning: Early epoch termination due to max steps limit, epoch step ={epoch_step}, global step = {current_global_step}, epoch = {index+1}')
            break

    # Run Validation Loss
    if not finetune and args.max_seq_length == 512:
        logger.info(f"TRAIN BATCH SIZE: {args.train_batch_size}")
        pretrain_validation(args, index, model)

    if finetune:
        eval_tp1pp(args, index, model)

def update_learning_rate(config, current_global_step, optimizer):
    global last_global_step_from_restore

    global_step_for_lr = current_global_step - last_global_step_from_restore
    lr_this_step = config["training"]["learning_rate"] * warmup_linear_decay_exp(global_step_for_lr,
                                                                                config["training"]["decay_rate"],
                                                                                config["training"]["decay_step"],
                                                                                config["training"]["total_training_steps"],
                                                                                config["training"]["warmup_proportion"])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    return lr_this_step

def report_step_metrics(args, lr, loss, step, data_sample_count):
    ##### Record the LR against global_step on tensorboard #####
    if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
        args.summary_writer.add_scalar(
            f'Train/lr', lr, step)

        args.summary_writer.add_scalar(
                f'Train/Samples/train_loss', loss, data_sample_count)

        args.summary_writer.add_scalar(
                f'Train/Samples/lr', lr, data_sample_count)
    ##### Recording  done. #####

    if (step + 1) % args.print_steps == 0 and master_process(args):
        print('bing_bert_progress: step={}, loss={}, lr={}, sample_count={}'
        .format(step + 1, loss, lr, data_sample_count))

def report_lamb_coefficients(args, optimizer):
    if master_process(args):
        if (args.fp16 and args.use_lamb):
            #print("Lamb Coeffs", optimizer.optimizer.get_lamb_coeffs())
            lamb_coeffs=optimizer.optimizer.get_lamb_coeffs()
            lamb_coeffs=np.array(lamb_coeffs)
            if lamb_coeffs.size > 0:
                args.summary_writer.add_histogram(
                        f'Train/lamb_coeffs', lamb_coeffs, global_step)

def get_arguments():
    parser = get_argument_parser()
    args = parser.parse_args()

    return args

def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    args.config = config

    job_name = config['name'] if args.job_name is None else args.job_name
    print("Running Config File: ", job_name)
    # Setting the distributed variables
    print("Args = {}".format(args))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                            and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        start_time = time.time()
        torch.distributed.init_process_group(backend='nccl')
        end_time = time.time()
        logger.info("Init_process_group takes %f sec" % (end_time - start_time))

        if args.fp16:
            logger.info(
                "16-bits distributed training not officially supported but seems to be working.")
            args.fp16 = True  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory () already exists and is not empty.")

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(
        args.output_dir, "saved_models/", job_name)

    # Prepare Summary Writer and saved_models path
    if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
        summary_writer = get_sample_writer(
            name=job_name, base=args.output_dir)
        args.summary_writer = summary_writer
        os.makedirs(args.saved_model_path, exist_ok=True)

    # set device
    args.device = device
    args.n_gpu = n_gpu

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning('Early training exit is set after {} global steps'.format(args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(args.max_steps_per_epoch))

    return args

def prepare_optimizer_parameters(args, model):
    config = args.config

    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters

def prepare_model_optimizer(args):
    # Loading Model
    model = BertMultiTask(args)

    if args.fp16:
        model.half()
    model.to(args.device)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # Prepare Optimizer
    config = args.config
    logger = args.logger
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer, FP16_UnfusedOptimizer, FusedAdam, FusedLamb
        except:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        if args.use_lamb:
            logger.info("Using Lamb optimizer min_coeff={}, max_coeff={}".format(args.min_lamb, args.max_lamb))
            optimizer = FusedLamb(optimizer_grouped_parameters,
                                  lr=config["training"]["learning_rate"],
                                  bias_correction=False,
                                  max_grad_norm=1.0,
                                  max_coeff=args.max_lamb,
                                  min_coeff=args.min_lamb)
        else:
            logger.info("Using adam optimizer")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=config["training"]["learning_rate"],
                                  bias_correction=False,
                                  max_grad_norm=1.0)
        logger.info(f"unwrapped optimizer_state = {optimizer.state_dict()}")
        if args.use_lamb:
            optimizer = FP16_UnfusedOptimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config["training"]["learning_rate"],
                             warmup=config["training"]["warmup_proportion"],
                             t_total=config["training"]["total_training_steps"])
    if args.local_rank != -1:
        try:
            logger.info(
                "***** Using Default Apex Distributed Data Parallel *****")
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        torch.cuda.set_device(args.local_rank)
        model.network = DDP(model.network, delay_allreduce=args.delay_allreduce, message_size=250000000)
    elif args.n_gpu > 1:
        model.network = DDP(model.network, delay_allreduce=args.delay_allreduce, message_size=250000000)
    return model, optimizer

def load_checkpoint(args, model, optimizer):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}")
    start_epoch, global_step, global_data_samples = load_training_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        PATH=args.load_training_checkpoint,
        load_optimizer_state=args.use_lamb)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step} and global data samples at {global_data_samples}")

    # restore global data samples in model
    model.network.sample_count = global_data_samples
    if args.rewarmup:
        logger.info(
            f"Rewarmup learning rate with last_global_step_from_restore = {global_step}")
        last_global_step_from_restore = global_step

    lr_this_step = config["training"]["learning_rate"] * warmup_linear_decay_exp(global_step,
                                                                                 config["training"]["decay_rate"],
                                                                                 config["training"]["decay_step"],
                                                                                 config["training"]["total_training_steps"],
                                                                                 config["training"]["warmup_proportion"])
    logger.info(f"Restart training with lr = {lr_this_step}")

    # Run validation for checkpoint before training
    if not args.finetune and args.max_seq_length == 512:
        logger.info(f"Validation Loss of Checkpoint {start_epoch} before pretraining")
        logger.info(f"TRAIN BATCH SIZE: {args.train_batch_size}")
        index = start_epoch - 1 if start_epoch > 0 else start_epoch
        pretrain_validation(args, index, model)

    return start_epoch

def run(args, model, optimizer, start_epoch):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    if args.finetune:
        for index in range(config["training"]["num_epochs"]):
            logger.info(f"Finetuning Epoch: {index + 1}")

            train(args, index, model, optimizer, finetune=True)

            if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
                model.save(os.path.join(args.saved_model_path,
                                        "model_finetuned_epoch_{}.pt".format(index + 1)))
    else:
        for index in range(start_epoch, config["training"]["num_epochs"]):
            logger.info(f"Training Epoch: {index + 1}")
            pre = time.time()
            train(args, index, model, optimizer)
            if (not args.no_cuda and dist.get_rank() == 0) or (args.no_cuda and args.local_rank == -1):
                logger.info(
                    f"Saving a checkpointing of the model for epoch: {index+1}")
                saved_model_path = os.path.join(args.saved_model_path, "training_state_checkpoint_{}.tar".format(index + 1))
                checkpoint_model(PATH=saved_model_path,
                                 model=model,
                                 optimizer=optimizer,
                                 epoch=index+1,
                                 last_global_step=global_step,
                                 last_global_data_samples=global_data_samples)

            post = time.time()
            logger.info(f"Time for shard {index + 1}: {post-pre} seconds")

            current_global_step = global_step - last_global_step_from_restore
            if is_time_to_exit(args=args,
                               global_steps=current_global_step):
                print(f'Warning: Early training termination due to max steps limit, epoch={index+1}, global_step={current_global_step}')
                break


def main():
    start = time.time()
    args = construct_arguments()
    model, optimizer = prepare_model_optimizer(args)
    start_epoch = 0
    if args.load_training_checkpoint is not None:
        start_epoch = load_checkpoint(args, model, optimizer)
    run(args, model, optimizer, start_epoch)
    elapsed = time.time() - start
    logger = args.logger
    logger.info(f"Elapsed time: {elapsed} seconds")

if __name__ == "__main__":
    main()
