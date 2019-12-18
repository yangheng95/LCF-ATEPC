# -*- coding: utf-8 -*-
# file: batch_train.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os, sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.tokenization_bert import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from utils.data_utils import ATEPCProcessor, convert_examples_to_features

from model.lcf_atepc import LCF_ATEPC, BertModel
from model.lcf_atepc_chinese import LCF_ATEPC_Chinese, BertModel

from sklearn.metrics import f1_score
from time import strftime, localtime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))
os.mkdir('logs/{}'.format(time))
os.system('cp -r *.py logs/{}/'.format(time))
os.system('cp -r model/*.py logs/{}/'.format(time))
log_file = 'logs/{}/{}.log'.format(time, time)
logger.addHandler(logging.FileHandler(log_file))


def main(config):
    args = config

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    datasets = {
        'camera': "atepc_datasets/camera",
        'car': "atepc_datasets/car",
        'phone': "atepc_datasets/phone",
        'notebook': "atepc_datasets/notebook",
        'laptop': "atepc_datasets/laptop",
        'restaurant': "atepc_datasets/restaurant",
        'twitter': "atepc_datasets/twitter",
        'mixed': "atepc_datasets/mixed",
    }
    if args.dataset in {'camera','car','phone','notebook'}:
        logger.warning("\n\nThis is the training script for Chinese review dataset,"
                       " DO NOT use this script to train model on {} dataset!!!\n\n".format(args.dataset))
    pretrained_bert_models = {
        'camera': "bert-base-chinese",
        'car': "bert-base-chinese",
        'phone': "bert-base-chinese",
        'notebook': "bert-base-chinese",
        # 'laptop': "bert-base-uncased",
        'laptop': "bert_pretrained_laptop",
        'restaurant': "bert-base-uncased",
        'twitter': "bert-base-uncased",
        'mixed': "bert-base-multilingual-uncased",
    }

    args.bert_model = pretrained_bert_models[args.dataset]
    args.data_dir = datasets[args.dataset]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    bert_base_model = BertModel.from_pretrained(args.bert_model)

    bert_base_model.config.num_labels = num_labels
    model = LCF_ATEPC(bert_base_model, args=args)

    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00001}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.00001)

    label_map = {i: label for i, label in enumerate(label_list, 1)}

    eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                 tokenizer)
    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                              all_polarities, all_valid_ids, all_lmask_ids)
    # Run prediction for full data
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    def evaluate(eval_ATE=True, eval_APC=True):
        # evaluate
        apc_result = 0
        ate_result = 0
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        model.eval()
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in eval_dataloader:
            input_ids_spc = input_ids_spc.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            polarities = polarities.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                ate_logits, apc_logits = model(input_ids_spc, segment_ids, input_mask,
                                               valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask)

            if eval_APC:
                polarities = LCF_ATEPC.get_batch_polarities(model, polarities)
                n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
                n_test_total += len(polarities)

                if test_polarities_all is None:
                    test_polarities_all = polarities
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)

            if eval_ATE:
                ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = ate_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_list):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map[label_ids[i][j]])
                            if not (0 < ate_logits[i][j] < 5): ate_logits[i][j] = 1
                            temp_2.append(label_map[ate_logits[i][j]])

        test_acc = n_test_correct / n_test_total
        test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                           labels=[0, 1, 2], average='macro')
        test_acc = round(test_acc * 100, 2)
        test_f1 = round(test_f1 * 100, 2)
        apc_result = {'max_apc_test_acc': test_acc, 'max_apc_test_f1': test_f1}

        if eval_ATE:
            report = classification_report(y_true, y_pred, digits=4)
            tmps = report.split()
            ate_result = round(float(tmps[7]) * 100, 2)
        return apc_result, ate_result

    def train():
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        max_apc_test_acc = 0
        max_apc_test_f1 = 0
        max_ate_test_f1 = 0

        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            logger.info('#' * 80)
            logger.info('Train {} Epoch{}'.format(args.seed, epoch + 1, args.data_dir))
            logger.info('#' * 80)
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch
                loss_ate, loss_apc = model(input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids,
                                           l_mask)
                if args.only_ate_or_apc is None:
                    loss = loss_ate + loss_apc
                    loss.backward()
                elif 'ate' in args.only_ate_or_apc:
                    loss_ate.backward()
                elif 'apc' in args.only_ate_or_apc:
                    loss_apc.backward()
                # logger.info(f'loss={round(loss.item(), 4)} (loss_ate{round(loss_ate.item(), 4)}+loss_apc{round(loss_apc.item(), 4)})')
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.eval_steps == 0:
                    apc_result, ate_result = evaluate()
                    path = '{0}/{1}_{2}_apcacc_{3}_apcf1_{4}_atef1_{5}'.format(
                        args.output_dir,
                        args.data_dir.split('/')[1],
                        args.local_context_focus,
                        round(apc_result['max_apc_test_acc'], 2),
                        round(apc_result['max_apc_test_f1'], 2),
                        round(ate_result, 2)
                    )
                    max_apc_test_acc = apc_result['max_apc_test_acc'] if apc_result[
                                                                             'max_apc_test_acc'] > max_apc_test_acc else max_apc_test_acc
                    max_apc_test_f1 = apc_result['max_apc_test_f1'] if apc_result[
                                                                           'max_apc_test_f1'] > max_apc_test_f1 else max_apc_test_f1
                    max_ate_test_f1 = ate_result if ate_result > max_ate_test_f1 else max_ate_test_f1

                    current_apc_test_acc = apc_result['max_apc_test_acc']
                    current_apc_test_f1 = apc_result['max_apc_test_f1']
                    current_ate_test_f1 = round(ate_result, 2)

                    logger.info('*' * 80)
                    logger.info('Train {} Epoch{}, Evaluate for {}'.format(args.seed, epoch + 1, args.data_dir))
                    logger.info(f'APC_test_acc:{current_apc_test_acc}(max:{max_apc_test_acc})  '
                                f'APC_test_f1:{current_apc_test_f1}(max:{max_apc_test_f1})')
                    logger.info(f'ATE_test_f1:{current_ate_test_f1}(max:{max_ate_test_f1})')
                    logger.info('*' * 80)
        return [max_apc_test_acc, max_apc_test_f1, max_ate_test_f1]

    return train()


def parse_experiments(path):
    configs = []
    opt = argparse.ArgumentParser()
    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        ## Required parameters

        parser.add_argument("--dataset", default=config['dataset'], type=str)
        parser.add_argument("--output_dir", default=config['output_dir'], type=str)
        parser.add_argument("--SRD", default=int(config['SRD']), type=int)
        parser.add_argument("--learning_rate", default=float(config['learning_rate']), type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--bert_base", default=config['bert_base'], type=bool,
                            help="Use the BERT-Base model instead of LCF-ATEPC model")
        # parser.add_argument("--bert_model", default='bert-base-uncased' if 'bert_model' not in config.keys() else config['bert_model'], type=str)
        parser.add_argument("--local_context_focus",
                            default=None if 'None' in config['local_context_focus'] else config['local_context_focus'],
                            type=str)
        parser.add_argument("--num_train_epochs", default=float(config['num_train_epochs']), type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--train_batch_size", default=int(config['train_batch_size']), type=int,
                            help="Total batch size for training.")
        parser.add_argument("--dropout", default=float(config['dropout']), type=int)
        parser.add_argument("--max_seq_length", default=int(config['max_seq_length']), type=int)
        parser.add_argument("--only_ate_or_apc",
                            default=None if 'None' in config['only_ate_or_apc'] else config['only_ate_or_apc'],
                            type=str)

        ## Other parameters
        parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
        parser.add_argument("--eval_steps", default=5, help="evaluate per steps")
        parser.add_argument("--warmup_proportion", default=0.4, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")

        configs.append(parser.parse_args())
    return configs


if __name__ == "__main__":

    experiments = argparse.ArgumentParser()
    experiments.add_argument('--config_path',default='exp-batch.json',type=str,help='Path of experiments config file')
    experiments = experiments.parse_args()

    from utils.Pytorch_GPUManager import GPUManager

    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")
    results = []
    exp_configs = parse_experiments(experiments.config_path)
    n = 3
    for config in exp_configs:

        for i in range(n):
            config.device = device
            config.seed = i + 1
            logger.info('No.{} training process of {}'.format(i + 1, n))
            results.append(main(config))
            logger.info(results for result in results)
