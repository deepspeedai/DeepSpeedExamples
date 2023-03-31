import torch
from torch.nn import CrossEntropyLoss, MSELoss
import datasets
from datasets import load_dataset, load_metric
import copy
import os
import json
import transformers
from transformers import AutoConfig, PretrainedConfig
import huggingface_transformer
from huggingface_transformer.modeling_bert import BertForSequenceClassification
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)
acc_tasks = ["mnli", "mrpc", "sst2", "qqp", "qnli", "rte"]
corr_tasks = ["stsb"]
mcc_tasks = ["cola"]
output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst2": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification"
}
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_rank(args):
    def _print_rank_0(msg):
        if args.local_rank <= 0:
            print(msg)
    return _print_rank_0
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def check_and_identify_compresssion(args, ds_config):
    assert args.per_device_train_batch_size == ds_config["train_micro_batch_size_per_gpu"]
    assert args.gradient_accumulation_steps == ds_config["train_batch_size"] / ds_config["train_micro_batch_size_per_gpu"]
    quantization_enabled, prune_enabled, layer_reduction_enabled = False, False, False
    if ds_config["compression_training"]["layer_reduction"]["enabled"]:
        layer_reduction_enabled = True

    if ds_config["compression_training"]["sparse_pruning"]["shared_parameters"]["enabled"] or \
        ds_config["compression_training"]["row_pruning"]["shared_parameters"]["enabled"] or \
            ds_config["compression_training"]["head_pruning"]["shared_parameters"]["enabled"]:
        prune_enabled = True

    if ds_config["compression_training"]["weight_quantization"]["shared_parameters"]["enabled"] or \
        ds_config["compression_training"]["activation_quantization"]["shared_parameters"]["enabled"]:
        quantization_enabled = True

    return layer_reduction_enabled, prune_enabled, quantization_enabled

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (-targets_prob * student_likelihood).mean()


# Some models have set the order of the labels to use, so let's make sure we do use it.
def replace_config(args, config, model_tmp, label_list, num_labels=2,label_to_id=None, is_regression=False):
    if (model_tmp.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and args.task_name is not None and not is_regression):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v
            for k, v in model_tmp.config.label2id.items()
        }
        if list(sorted(label_name_to_id.keys())) == list(
                sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!")
            label_to_id = {
                i: label_name_to_id[label_list[i]]
                for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    if label_to_id is not None:
        model_tmp.config.label2id = label_to_id
        model_tmp.config.id2label = {
            id: label
            for label, id in config.label2id.items()
        }
    elif args.task_name is not None and not is_regression:
        model_tmp.config.label2id = {
            l: i
            for i, l in enumerate(label_list)
        }
        model_tmp.config.id2label = {
            id: label
            for label, id in config.label2id.items()
        }


def do_eval(args, model, eval_dataloader, mm_eval_dataloader, device, is_regression=False):
    model.eval()
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(predictions=predictions, references=batch["labels"])
    eval_metric = metric.compute()
    
    eval_metric1 = None
    if args.task_name == 'mnli':
        metric1 = load_metric("accuracy")
        for step, batch in enumerate(mm_eval_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric1.add_batch(predictions=predictions, references=batch["labels"])
        eval_metric1 = metric1.compute()
    model.train()
    return eval_metric, eval_metric1    

def arrange_output(task_name, results, previous_best, best_dev_acc):
    result = results[0]
    result1 = results[1]
    save_model = False
    if task_name in acc_tasks:
        if task_name in ['sst2', 'qnli', 'rte']:
            current_result = f"acc:{result['accuracy']}"
        elif task_name == 'mnli':
            current_result = f"acc/mm-acc:{result['accuracy']}/{result1['accuracy']}"
        elif task_name in ['mrpc', 'qqp']:
            current_result = f"f1/acc:{result['f1']}/{result['accuracy']}"
        
        if result['accuracy'] > best_dev_acc:
            save_model = True
            best_dev_acc = result['accuracy']
            previous_best = current_result

    elif task_name in corr_tasks:
        current_result = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"
        if result['pearson'] > best_dev_acc:
            best_dev_acc = result['pearson']
            save_model = True
            previous_best = current_result
    elif task_name in mcc_tasks:
        current_result = f"mcc:{result['matthews_correlation']}"
        if result['matthews_correlation'] > best_dev_acc:
            best_dev_acc = result['matthews_correlation']
            save_model = True
            previous_best = current_result
    return current_result, previous_best, best_dev_acc, save_model


def forward_loss(args, ds_config, output_mode):
    assert args.distill_method in ['zero_stage', 'one_stage']
   
    if args.distill_method == 'zero_stage':   
        def _simple_function(batch, model, teacher_model=None):      
            outputs = model(**batch)
            return [outputs.loss, 0, 0 ,0 ]
        return _simple_function

    elif args.distill_method == 'one_stage':  
        loss_mse =  MSELoss()
        if output_mode == "classification":
            cls_loss_func = soft_cross_entropy
        elif output_mode == "regression":
            cls_loss_func = loss_mse

        def _kd_function(batch, model, teacher_model):   
            att_loss, rep_loss, loss,  = 0., 0., 0.
            outputs = model(**batch, output_attentions=True, output_hidden_states=True)
            student_logits, student_reps, student_atts = outputs.logits, outputs.hidden_states, outputs.attentions
            with torch.no_grad():
                outputs_teacher = teacher_model(**batch, output_attentions=True, output_hidden_states=True)
            teacher_logits, teacher_reps, teacher_atts = outputs_teacher.logits, outputs_teacher.hidden_states, outputs_teacher.attentions
            cls_loss = cls_loss_func(student_logits, teacher_logits)
            loss += cls_loss
            teacher_layer_num, student_layer_num = len(teacher_atts), len(student_atts)
            if args.layer_reduction_enabled:
                teacher_layers = [x for x in ds_config["compression_training"]["layer_reduction"]['teacher_layer']]
                att_list = [x for x in teacher_layers]
                rep_list = [teacher_layers[0] - 1,] + [x + 1 for x in teacher_layers]
            else:
                ## ATTENTION: The knowledge distillation is designed for skip-layer KD
                layers_per_block = int(teacher_layer_num / student_layer_num)  ###[1, 3, 5, 7, 9, 11]
                att_list = [i * layers_per_block + layers_per_block - 1 for i in range(student_layer_num)]
                rep_list = [i * layers_per_block for i in range(student_layer_num + 1)]  ###[0, 2, 4, 6, 8, 10, 12]
            
            new_teacher_reps = [teacher_reps[i] for i in rep_list]
            new_teacher_atts = [teacher_atts[i] for i in att_list]

            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                tmp_loss = loss_mse(student_att, teacher_att)
                att_loss += tmp_loss

            for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss
            loss += att_loss + rep_loss
            return [loss, rep_loss.item(), cls_loss.item(), att_loss.item(), ]

        return _kd_function    


def record_stat(stat_history, all_loss):
    past_loss = stat_history['tmp_loss']
    tr_loss, tr_rep_loss, tr_cls_loss, tr_att_loss = past_loss[0]+all_loss[0].item(), past_loss[1]+all_loss[1], past_loss[2]+all_loss[2], past_loss[3]+all_loss[3] 
    stat_history['tmp_loss'] = [tr_loss, tr_rep_loss, tr_cls_loss, tr_att_loss]
    return stat_history


def update_stat_and_print(args, print_rank_0, forward_step, stat_history, optimizer, arrange_out, ds_config):
    eval_result, previous_best, best_dev_acc, save_model = arrange_out[0], arrange_out[1], arrange_out[2], arrange_out[3]
    print_rank_0( f"***** Running evaluation Stage {args.distill_method}*****")
    print_rank_0("  {} step of {}".format(forward_step, args.max_train_steps))
    
    past_loss = stat_history['tmp_loss']
    tr_loss, tr_rep_loss, tr_cls_loss, tr_att_loss = past_loss[0], past_loss[1], past_loss[2], past_loss[3], 
    loss = tr_loss / (args.eval_step  + 1)
    cls_loss = tr_cls_loss / (args.eval_step + 1)
    att_loss = tr_att_loss / (args.eval_step  + 1)
    rep_loss = tr_rep_loss / (args.eval_step  + 1)
        
    stat_history['lr1'].append(optimizer.param_groups[0]["lr"])
    stat_history['lr2'].append(optimizer.param_groups[1]["lr"])
    stat_history['train_ffn_loss'].append(rep_loss)
    stat_history['train_att_loss'].append(att_loss)
    stat_history['train_loss'].append(loss)
    stat_history['eval'].append(eval_result)
    stat_history['forward_step'].append(forward_step)
    teacher_result =  stat_history['teacher_result']
    try:
        print_rank_0(
            '{' +
            f"eval_result: {eval_result}, step: {forward_step/args.max_train_steps}, train_loss: {stat_history['train_loss'][-1]}, train_ffn_loss: {stat_history['train_ffn_loss'][-1]},  train_att_loss:{stat_history['train_att_loss'][-1]}, lr1: { stat_history['lr1'][-1]}, lr2: { stat_history['lr2'][-1]}, "
            + '}')
    except:
        print_rank_0(eval_result)
    if previous_best is not None:
        print_rank_0(f"task {args.task_name}, teacher_result: {teacher_result}\nPrevious best: {previous_best}")

    tr_loss, tr_rep_loss, tr_cls_loss, tr_att_loss = 0., 0., 0., 0.,
    stat_history['tmp_loss'] = [ 0, 0., 0., 0.,]

    ##############for pruning
    sparse_prune, row_prune, head_prune = False, False, False
    sparse_iter, row_iter, head_iter = 0, 0, 0
    if ds_config["compression_training"]["sparse_pruning"]["shared_parameters"]["enabled"]:
        sparse_prune = True
        sparse_iter = ds_config["compression_training"]["sparse_pruning"]["shared_parameters"]["schedule_offset"]
    if ds_config["compression_training"]["row_pruning"]["shared_parameters"]["enabled"]:
        row_prune = True
        row_iter = ds_config["compression_training"]["row_pruning"]["shared_parameters"]["schedule_offset"]
    if ds_config["compression_training"]["head_pruning"]["shared_parameters"]["enabled"]:
        head_prune = True
        head_iter = ds_config["compression_training"]["head_pruning"]["shared_parameters"]["schedule_offset"]
    if sparse_prune or row_prune or head_prune:
        save_iter = np.max([sparse_iter, row_iter, head_iter]) 
        if forward_step<save_iter:
            save_model = False
            best_dev_acc = 0
    return stat_history, best_dev_acc, save_model

def save_checkpoint_and_config(args, model, config, tokenizer, ds_config=None):
    WEIGHTS_NAME = "pytorch_model.bin"
    CONFIG_NAME = 'config.json'
    output_dir = os.path.join(args.output_dir, 'best')    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    ### get the model to be saved
    model_to_save = model.module if hasattr(model, 'module') else model
    model_will_save = copy.deepcopy(model_to_save)
    if ds_config is None:
        if args.local_rank in [-1, 0]:
            torch.save(model_will_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
    else:
        if args.local_rank in [-1, 0]:
            torch.save(model_will_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)
            if args.deepspeed:
                new_json_path = os.path.join(output_dir, "ds_config.json")
                with open(new_json_path, 'w') as f:
                    json.dump(ds_config, f)


def save_clean_best_model(args, print_rank_0,  model, tokenizer, config, redundancy_clean, eval_dataloader, mm_eval_dataloader, device, is_regression, previous_best, best_dev_acc, ds_config=None):
    if ds_config is not None:
        WEIGHTS_NAME = "pytorch_model.bin"
        CONFIG_NAME = 'config.json'        
        layer_reduction_enabled, prune_enabled, quantization_enabled = check_and_identify_compresssion(args, ds_config)
        output_dir_best = os.path.join(args.output_dir, 'best')   
        best_model_path = os.path.join(output_dir_best, WEIGHTS_NAME) 
        if os.path.exists(output_dir_best):
            best_model = torch.load(best_model_path)
            new_sd = {}
            for k, v in best_model.items():
                new_sd["module."+k] = v
            model.load_state_dict(new_sd, strict=False)  
        else:
            print_rank_0 ("WARNING: no best model yet")

        result = do_eval(args, model, eval_dataloader, mm_eval_dataloader, device, is_regression=is_regression)
        current_result, previous_best, best_dev_acc, _ = arrange_output(args.task_name, result, previous_best, best_dev_acc)
        print_rank_0( f"Before clean, double check the perforamnce of best model is {current_result}")           
        try:
             model = redundancy_clean(model, args.deepspeed_config)           
        except:
            print_rank_0 ("WARNING: redundany_clean is not applicable")
            pass  

        if  ds_config["compression_training"]["head_pruning"]["shared_parameters"]["enabled"]:
            for module in model.modules():
                if hasattr(module, 'num_attention_heads'):
                    ratio = ds_config["compression_training"]['head_pruning']["different_groups"]["rp1"]["params"]["dense_ratio"]
                    config.num_attention_heads = math.ceil(config.num_attention_heads * ratio)
                    module.num_attention_heads = math.ceil(module.num_attention_heads * ratio)
                    module.all_head_size = int(module.num_attention_heads * 64)

        result = do_eval(args, model, eval_dataloader, mm_eval_dataloader, device, is_regression=is_regression)
        current_result, previous_best, best_dev_acc, _ = arrange_output(args.task_name, result, previous_best, best_dev_acc)
        print_rank_0( f"Clean the best model, and the accuracy of the clean model is {current_result}")

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save = copy.deepcopy(model_to_save)
        WEIGHTS_NAME = "pytorch_model.bin"
        CONFIG_NAME = 'config.json'

        if args.local_rank in [-1, 0]:
            output_dir_best_clean = os.path.join(args.output_dir, 'clean') 
            if not os.path.exists(output_dir_best_clean):
                os.makedirs(output_dir_best_clean)
            output_model_file = os.path.join(output_dir_best_clean, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir_best_clean, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file) 
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir_best_clean)
            if args.deepspeed:
                new_json_path = os.path.join(args.output_dir, "ds_config.json")
                with open(new_json_path, 'w') as f:
                    json.dump(ds_config, f)        
        