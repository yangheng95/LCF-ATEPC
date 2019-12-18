# from __future__ import absolute_import, division, print_function
#
# import argparse
# import json
# import logging
# import os, sys
# import random
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from pytorch_transformers.optimization import AdamW
# from pytorch_transformers.tokenization_bert import BertTokenizer
# from seqeval.metrics import classification_report
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
# from tqdm import tqdm, trange
#
# from utils.data_utils import ATEPCProcessor, convert_examples_to_features
#
# from model.lcf_atepc import BERT_ATE, BertModel
#
# from sklearn.metrics import f1_score
# from time import strftime, localtime
#
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
#
# time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))
# os.mkdir('logs/{}'.format(time))
# os.system('cp -r *.py logs/{}/'.format(time))
# os.system('cp -r model/*.py logs/{}/'.format(time))
# log_file = 'logs/{}/{}.log'.format(time, time)
# logger.addHandler(logging.FileHandler(log_file))
#
# def main(seed = 1,device='cuda'):
#     parser = argparse.ArgumentParser()
#
#
#     ## Required parameters
#     parser.add_argument("--data_dir", default='atepc_datasets/laptop', type=str)
#     parser.add_argument("--output_dir", default='output', type=str)
#     parser.add_argument("--SRD", default=5, type=int)
#     parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
#     parser.add_argument("--local_context_focus", default='cdm', type=str)
#     parser.add_argument("--num_train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
#     parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
#     parser.add_argument("--dropout", default=0, type=float)
#     parser.add_argument("--max_seq_length", default=80, type=int)
#     parser.add_argument("--only_ate_or_apc", default=None, type=str)
#     parser.add_argument('--seed', type=int, default=seed, help="random seed for initialization")
#
#     ## Other parameters
#     parser.add_argument("--eval_batch_size",  default=32, type=int, help="Total batch size for eval.")
#     parser.add_argument("--eval_steps", default=5, help="evaluate per steps")
#     parser.add_argument("--device", default=device, type = str, help="evaluate per steps")
#     parser.add_argument("--warmup_proportion", default=0.4, type=float,
#                         help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument('--loss_scale', type=float, default=0,
#                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
#                              "0 (default value): dynamic loss scaling.\n"
#                              "Positive power of 2: static loss scaling value.\n")
#     args = parser.parse_args()
#
#     if args.gradient_accumulation_steps < 1:
#         raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
#             args.gradient_accumulation_steps))
#
#     args.bert_model='bert-base-uncased'
#     # if 'laptop' in args.data_dir:
#     #     args.bert_model = 'bert_pretrained_laptop'
#     # elif 'rest' in args.data_dir:
#     #     args.bert_model = 'bert_pretrained_restaurant'
#     # else:
#     #     args.bert_model = 'bert_pretrained_joint'
#
#     args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
#
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     processor = ATEPCProcessor()
#     label_list = processor.get_labels()
#     num_labels = len(label_list) + 1
#
#     tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
#
#     train_examples = processor.get_train_examples(args.data_dir)
#     num_train_optimization_steps = int(
#             len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
#
#     bert_base_model = BertModel.from_pretrained(args.bert_model)
#     bert_base_model.config.num_labels = num_labels
#     model = BERT_ATE(bert_base_model, args=args)
#
#     for arg in vars(args):
#         logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))
#
#     model.to(device)
#
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
#
#     label_map = {i: label for i, label in enumerate(label_list, 1)}
#
#     eval_examples = processor.get_test_examples(args.data_dir)
#     eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
#                                                  tokenizer)
#     all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
#     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
#     all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
#     all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
#     all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
#     eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
#                               all_polarities, all_valid_ids, all_lmask_ids)
#     # Run prediction for full data
#     eval_sampler = RandomSampler(eval_data)
#     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
#
#     def evaluate(eval_ATE=True, eval_APC=True):
#         # evaluate
#         model.evaluate=True
#         apc_result=None
#         ate_result=None
#         y_true = []
#         y_pred = []
#         n_test_correct, n_test_total = 0, 0
#         test_apc_logits_all, test_polarities_all = None, None
#         model.eval()
#         label_map = {i: label for i, label in enumerate(label_list, 1)}
#         for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in eval_dataloader:
#             input_ids_spc = input_ids_spc.to(device)
#             input_mask = input_mask.to(device)
#             segment_ids = segment_ids.to(device)
#             valid_ids = valid_ids.to(device)
#             label_ids = label_ids.to(device)
#             polarities = polarities.to(device)
#             l_mask = l_mask.to(device)
#
#             with torch.no_grad():
#                 ate_logits, apc_logits = model(input_ids_spc, segment_ids, input_mask,
#                                                valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask)
#
#             # code block for eval_APC task
#             if eval_APC:
#                 polarities = BERT_ATE.get_batch_polarities(model, polarities)
#                 n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
#                 n_test_total += len(polarities)
#
#                 if test_polarities_all is None:
#                     test_polarities_all = polarities
#                     test_apc_logits_all = apc_logits
#                 else:
#                     test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
#                     test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)
#             # code block for eval_APC task
#
#             # code block for eval_ATE task
#             try:
#                 if eval_ATE:
#                     ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
#                     ate_logits = ate_logits.detach().cpu().numpy()
#                     label_ids = label_ids.to('cpu').numpy()
#                     input_mask = input_mask.to('cpu').numpy()
#                     for i, label in enumerate(label_ids):
#                         temp_1 = []
#                         temp_2 = []
#                         for j, m in enumerate(label):
#                             if j == 0:
#                                 continue
#                             elif label_ids[i][j] == len(label_list):
#                                 y_true.append(temp_1)
#                                 y_pred.append(temp_2)
#                                 break
#                             else:
#                                 temp_1.append(label_map[label_ids[i][j]])
#                                 if not (0<ate_logits[i][j]<5):ate_logits[i][j]=1
#                                 temp_2.append(label_map[ate_logits[i][j]])
#             except Exception as e:
#                 e.with_traceback()
#                 ate_result = 'eval failed!'
#             # code block for eval_ATE task
#
#         # code block for eval_APC task
#         try:
#             test_acc = n_test_correct / n_test_total
#             test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
#                                labels=[0, 1, 2], average='macro')
#             test_acc=round(test_acc * 100, 2)
#             test_f1=round(test_f1 * 100, 2)
#             apc_result={'max_apc_test_acc':test_acc,'max_apc_test_f1': test_f1}
#         except:
#             apc_result='eval failed!'
#         # code block for eval_APC task
#
#         # code block for eval_ATE task
#         if eval_ATE and ate_result != 'eval failed!':
#             report = classification_report(y_true, y_pred, digits=4)
#             # logger.info("\n%s", report)
#             tmps = report.split()
#             try:
#                 ate_result=round(float(tmps[7])*100, 2)
#             except:
#                 ate_result=0
#         else:
#             ate_result = 0
#
#         return apc_result, ate_result
#
#     def train():
#         train_features = convert_examples_to_features(
#             train_examples, label_list, args.max_seq_length, tokenizer)
#         logger.info("***** Running training *****")
#         logger.info("  Num examples = %d", len(train_examples))
#         logger.info("  Batch size = %d", args.train_batch_size)
#         logger.info("  Num steps = %d", num_train_optimization_steps)
#         all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
#         all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
#         all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
#         all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
#         all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
#         all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
#         all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)
#         train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
#                                    all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)
#
#         train_sampler = SequentialSampler(train_data)
#         train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
#         max_apc_test_acc = 0
#         max_apc_test_f1 = 0
#         max_ate_test_f1 = 0
#
#         global_step = 0
#         for epoch in range(int(args.num_train_epochs)):
#             logger.info('#' * 80)
#             logger.info('Train {} Epoch{}'.format(args.seed, epoch + 1, args.data_dir))
#             logger.info('#' * 80)
#             nb_tr_examples, nb_tr_steps = 0, 0
#             for step, batch in enumerate(train_dataloader):
#                 model.train()
#                 model.evaluate = False
#                 batch = tuple(t.to(device) for t in batch)
#                 input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch
#                 loss_ate, loss_apc = model(input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids, l_mask)
#                 if args.only_ate_or_apc is None:
#                     loss = loss_ate + loss_apc
#                     loss.backward()
#                 elif 'ate' in args.only_ate_or_apc:
#                     loss_ate.backward()
#                 elif 'apc' in args.only_ate_or_apc:
#                     loss_apc.backward()
#                 # logger.info(f'loss={round(loss.item(), 4)} (loss_ate{round(loss_ate.item(), 4)}+loss_apc{round(loss_apc.item(), 4)})')
#                 nb_tr_examples += input_ids_spc.size(0)
#                 nb_tr_steps += 1
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 global_step += 1
#                 if global_step % args.eval_steps == 0:
#                     apc_result, ate_result = evaluate()
#                     path = '{0}/{1}_{2}_apcacc_{3}_apcf1_{4}_atef1_{5}'.format(
#                         args.output_dir,
#                         args.data_dir.split('/')[1],
#                         args.local_context_focus,
#                         round(apc_result['max_apc_test_acc'], 2),
#                         round(apc_result['max_apc_test_f1'], 2),
#                         round(ate_result, 2)
#                     )
#                     max_apc_test_acc = apc_result['max_apc_test_acc'] if apc_result['max_apc_test_acc'] > max_apc_test_acc else max_apc_test_acc
#                     max_apc_test_f1 = apc_result['max_apc_test_f1'] if apc_result['max_apc_test_f1'] > max_apc_test_f1 else max_apc_test_f1
#                     max_ate_test_f1 = ate_result if ate_result > max_ate_test_f1 else max_ate_test_f1
#
#                     current_apc_test_acc = apc_result['max_apc_test_acc']
#                     current_apc_test_f1 = apc_result['max_apc_test_f1']
#                     current_ate_test_f1 = round(ate_result,2)
#
#
#                     logger.info('*' * 80)
#                     logger.info('Train {} Epoch{}, Evaluate for {}'.format(args.seed, epoch+1, args.data_dir))
#                     logger.info(f'APC_test_acc:{current_apc_test_acc}(max:{max_apc_test_acc})  '
#                                 f'APC_test_f1:{current_apc_test_f1}(max:{max_apc_test_f1})')
#                     logger.info(f'ATE_test_f1:{current_ate_test_f1}(max:{max_ate_test_f1})')
#                     logger.info('*'*80)
#         # for arg in vars(args):
#         #     logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))
#         return [max_apc_test_acc, max_apc_test_f1, max_ate_test_f1]
#     return train()
#
#
# if __name__ == "__main__":
#     from utils.Pytorch_GPUManager import GPUManager
#     index = GPUManager().auto_choice()
#     device = torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")
#     results = []
#     for i in range(3):
#         logger.info('No.{} training process of {}'.format(i+1, 3))
#         results.append(main(seed=i+1, device=device))
#         np.array(results)
#         logger.info(results for result in results)


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

from model.lcf_atepc import BERT_ATE, BertModel

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

def main(seed = 1,device='cuda'):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='atepc_datasets/laptop', type=str)
    #parser.add_argument("--data_dir", default='atepc_datasets/restaurant', type=str)
    #parser.add_argument("--data_dir", default='atepc_datasets/twitter', type=str)

    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--SRD", default=5, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")

    # parser.add_argument("--local_context_focus", default=None, type=str)
    parser.add_argument("--local_context_focus", default='cdw', type=str)
    # parser.add_argument("--local_context_focus", default='cdm', type=str)

    parser.add_argument("--num_train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--max_seq_length", default=80, type=int)
    parser.add_argument("--only_ate_or_apc", default=None, type=str)
    parser.add_argument('--seed', type=int, default=seed, help="random seed for initialization")
    # 下面的参数不用改
    ## Other parameters
    parser.add_argument("--eval_batch_size",  default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_steps", default=5, help="evaluate per steps")
    parser.add_argument("--device", default=device, type = str, help="evaluate per steps")
    parser.add_argument("--warmup_proportion", default=0.4, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.bert_model='bert-base-uncased'
    # if 'laptop' in args.data_dir:
    #     args.bert_model = 'bert_pretrained_laptop'
    # elif 'rest' in args.data_dir:
    #     args.bert_model = 'bert_pretrained_restaurant'
    # else:
    #     args.bert_model = 'bert_pretrained_joint'

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True,)

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    bert_base_model = BertModel.from_pretrained(args.bert_model)
    bert_base_model.config.num_labels = num_labels
    model = BERT_ATE(bert_base_model, args=args)

    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

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
        model.evaluate=True
        apc_result=None
        ate_result=None
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

            # code block for eval_APC task
            if eval_APC:
                polarities = BERT_ATE.get_batch_polarities(model, polarities)
                n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
                n_test_total += len(polarities)

                if test_polarities_all is None:
                    test_polarities_all = polarities
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)
            # code block for eval_APC task

            # code block for eval_ATE task
            try:
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
                                temp_2.append(label_map[ate_logits[i][j]])
            except Exception as e:
                ate_result = 'eval failed!'
            # code block for eval_ATE task

        # code block for eval_APC task
        try:
            test_acc = n_test_correct / n_test_total
            test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                               labels=[0, 1, 2], average='macro')
            test_acc=round(test_acc * 100, 2)
            test_f1=round(test_f1 * 100, 2)
            apc_result={'max_apc_test_acc':test_acc,'max_apc_test_f1': test_f1}
        except:
            apc_result='eval failed!'
        # code block for eval_APC task

        # code block for eval_ATE task
        if eval_ATE and ate_result != 'eval failed!':
            report = classification_report(y_true, y_pred, digits=4)
            # logger.info("\n%s", report)
            tmps = report.split()
            try:
                ate_result=round(float(tmps[7])*100, 2)
            except:
                ate_result=0
        else:
            ate_result = 0

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
                model.evaluate = False
                batch = tuple(t.to(device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch
                loss_ate, loss_apc = model(input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids, l_mask)
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
                    max_apc_test_acc = apc_result['max_apc_test_acc'] if apc_result['max_apc_test_acc'] > max_apc_test_acc else max_apc_test_acc
                    max_apc_test_f1 = apc_result['max_apc_test_f1'] if apc_result['max_apc_test_f1'] > max_apc_test_f1 else max_apc_test_f1
                    max_ate_test_f1 = ate_result if ate_result > max_ate_test_f1 else max_ate_test_f1

                    current_apc_test_acc = apc_result['max_apc_test_acc']
                    current_apc_test_f1 = apc_result['max_apc_test_f1']
                    current_ate_test_f1 = round(ate_result,2)


                    logger.info('*' * 80)
                    logger.info('Train {} Epoch{}, Evaluate for {}'.format(args.seed, epoch+1, args.data_dir))
                    logger.info(f'APC_test_acc:{current_apc_test_acc}(max:{max_apc_test_acc})  '
                                f'APC_test_f1:{current_apc_test_f1}(max:{max_apc_test_f1})')
                    logger.info(f'ATE_test_f1:{current_ate_test_f1}(max:{max_ate_test_f1})')
                    logger.info('*'*80)
        # for arg in vars(args):
        #     logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))
        return [max_apc_test_acc, max_apc_test_f1, max_ate_test_f1]
    return train()


if __name__ == "__main__":
    from utils.Pytorch_GPUManager import GPUManager
    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")
    results = []
    for i in range(3):
        logger.info('No.{} training process of {}'.format(i+1, 3))
        results.append(main(seed=i+1, device=device))
        np.array(results)
        logger.info(results for result in results)
