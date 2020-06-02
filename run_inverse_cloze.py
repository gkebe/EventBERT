# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForNextSentencePrediction, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from apex import amp
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score, classification_report, \
    confusion_matrix
from utils import is_main_process

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(task_name, preds, labels, label_names=None):
    if label_names is None:
        label_names = []
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": acc_and_f1(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "frames":
        return metrics_frame(preds, labels, label_names)
    else:
        raise KeyError(task_name)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def metrics_frame(preds, labels, label_names):
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    cr = classification_report(labels, preds, labels=list(range(len(label_names))), target_names=label_names)
    model_metrics = {"Precision, Micro": precision_micro, "Precision, Macro": precision_macro,
                     "Recall, Micro": recall_micro, "Recall, Macro": recall_macro,
                     "F1 score, Micro": f1_micro, "F1 score, Macro": f1_macro, "Classification report": cr}
    return model_metrics


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, T, F1, F2, F3, F4, F5):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the instance.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the instance. This should be
            specified for train and dev instances, but not for test instances.
        """
        self.guid = guid
        self.T = T
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.F4 = F4
        self.F5 = F5


class InputInstance(object):
    """A single training/test instance for simple sequence classification."""

    def __init__(self, guid, T, F1, F2, F3, F4, F5):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the instance.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the instance. This should be
            specified for train and dev instances, but not for test instances.
        """
        self.guid = guid
        self.T = T
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.F4 = F4
        self.F5 = F5


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_instances(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir)))
    def get_num_events(self, data_dir):
        dataset = self._read_json(os.path.join(data_dir))
        return len(dataset["0"]["T"])
    @classmethod
    def _read_json(cls, input_file):
        """Reads a tab separated value file."""
        dataset_json = ""
        with open(input_file, "rb") as f:
            dataset_json = f.read()
        return json.loads(dataset_json)

    def _create_instances(self, dataset):
        """Creates instances for the training and dev sets."""
        instances = []
        for i, value in dataset.items():
            guid = int(i)
            T = value["T"]
            F1 = value["F1"]
            F2 = value["F2"]
            F3 = value["F3"]
            F4 = value["F4"]
            F5 = value["F5"]
            instances.append(
                InputInstance(guid=guid, T=T, F1=F1, F2=F2, F3=F3, F4=F4, F5=F5))
        return instances


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def tokenize_sequence(sequence, max_seq_length, tokenizer):
    sequence_ = []
    for first, second in zip(sequence, sequence[1:]):
        tokens_a = tokenizer.tokenize(first)
        tokens_b = tokenizer.tokenize(second)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        seq = dict()
        seq["input_ids"] = input_ids
        seq["input_mask"] = input_mask
        seq["segment_ids"] = segment_ids
        sequence_.append(seq)
    return sequence_


def convert_instances_to_features(instances, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, instance) in enumerate(instances):
        features.append(
            InputFeatures(guid=instance.guid,
                          T=tokenize_sequence(instance.T, max_seq_length, tokenizer),
                          F1=tokenize_sequence(instance.F1, max_seq_length, tokenizer),
                          F2=tokenize_sequence(instance.F2, max_seq_length, tokenizer),
                          F3=tokenize_sequence(instance.F3, max_seq_length, tokenizer),
                          F4=tokenize_sequence(instance.F4, max_seq_length, tokenizer),
                          F5=tokenize_sequence(instance.F5, max_seq_length, tokenizer)))
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


from apex.multi_tensor_apply import multi_tensor_applier


class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """

    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    args = parser.parse_args()
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()



    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=128)  # for bert large


    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = BertForNextSentencePrediction(config)
    print("USING CHECKPOINT from", args.init_checkpoint)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"], strict=False)
    print("USED CHECKPOINT from", args.init_checkpoint)

    model.to(device)
    # Prepare optimizer
    processor = DataProcessor()
    instances = processor.get_instances(args.data_dir)
    num_events = processor.get_num_events(args.data_dir)
    eval_features = convert_instances_to_features(
        instances, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(instances))
    logger.info("  Batch size = %d", 24)
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    for f in eval_features:
        sequences = [f.T, f.F1, f.F2, f.F3, f.F4, f.F5]
        input_ids = []
        input_mask = []
        segment_ids = []
        for seq in sequences:
            input_ids += [s["input_ids"] for s in seq]
            input_mask += [s["input_mask"] for s in seq]
            segment_ids += [s["segment_ids"] for s in seq]

        all_input_ids += input_ids
        all_input_mask += input_mask
        all_segment_ids += segment_ids

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)

    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=6*(num_events-1))

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    preds = []
    probs_ = []
    probs_prod_ = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            tmp_eval_loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, next_sentence_label=None)
            logits = model(input_ids, segment_ids, input_mask)

            probabilities = torch.softmax(logits, 1)
            probs = probabilities.tolist()
            probs = [i[0] for i in probs]
            probs_seq = [probs[x:x + (num_events-1)] for x in range(0, len(probs), (num_events-1))]
            probs_prod = [np.prod(i) for i in probs_seq]
            pred = np.argmax(probs_prod)
            preds.append(pred)
            probs_.append(probs_seq)
            probs_prod_.append(probs_prod)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    accuracy = simple_accuracy(np.array(preds), np.array([0]*len(preds)))

    eval_loss = eval_loss / nb_eval_steps

    instance_template = ["T", "F1", "F2", "F3", "F4", "F5"]

    for i in range(len(preds)):
        seqs = [instances[i].T, instances[i].F1, instances[i].F2, instances[i].F3, instances[i].F4, instances[i].F5]
        for j in range(len(probs_prod_[i])):
            for k in range(len(probs_[i][j])):
                print(seqs[j][k] + ": " + str(probs_[i][j][k]))
            print("Product = " + str(probs_prod_[i][j]))
        print("Predicted " + instance_template[int(preds[i])])
        print()

    results = {'eval_loss': eval_loss,
               'accuracy': accuracy}

    print(results)
    output_eval_file = os.path.join(args.output_dir,
                                    "eval_results_" + args.init_checkpoint.split("/")[-1].split(".")[0] + "_"
                                    + args.data_dir.split("/")[-1].split(".")[0] + ".txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))


if __name__ == "__main__":
    main()
