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
import math

import numpy as np
from scipy.special import softmax
import torch

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
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


def tokenize_sequence(sequence):
    seq = dict()
    seq["seed"] = sequence[0]
    seq["next"] = " ".join(sequence[1:])
    seq["full"] = " ".join(sequence)
    return seq


def convert_instances_to_features(instances, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, instance) in enumerate(instances):
        features.append(
            InputFeatures(guid=instance.guid,
                          T=tokenize_sequence(instance.T),
                          F1=tokenize_sequence(instance.F1),
                          F2=tokenize_sequence(instance.F2),
                          F3=tokenize_sequence(instance.F3),
                          F4=tokenize_sequence(instance.F4),
                          F5=tokenize_sequence(instance.F5)))
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def perplexity(tokenizer, model, full, device):
    input_ids = torch.tensor(tokenizer.encode(full)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--gpt2_model", default=None, type=str, required=True,
                        help="GPT-2 pre-trained model selected in the list: gpt2, "
                             "gpt2-medium, gpt2-large, gpt2-xl.")
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
                        help="Vocabulary mapping/file GPT-2 was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The GPT-2 model config")

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
    #tokenizer = XLNetTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=128)  # for xlnet base
    tokenizer = GPT2Tokenizer.from_pretrained(args.init_checkpoint)

    # Prepare model
    #config = XLNetConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    #if config.vocab_size % 8 != 0:
     #   config.vocab_size += 8 - (config.vocab_size % 8)

    #model = XLNetForSequenceClassification(config)
    #print("USING CHECKPOINT from", args.init_checkpoint)
    #model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"], strict=False)
    #print("USED CHECKPOINT from", args.init_checkpoint)

    model = GPT2LMHeadModel.from_pretrained(args.init_checkpoint)
    print("USING CHECKPOINT from", args.init_checkpoint)
    #model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
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
    preds=[]
    probs=[]
    i=0
    instance_template = ["T", "F1", "F2", "F3", "F4", "F5"]

    for f in eval_features:
        sequences = [f.T, f.F1, f.F2, f.F3, f.F4, f.F5]
        input_ids = []
        input_mask = []
        log_ls = []
        for seq in sequences:
            print(seq)
            log_l = perplexity(tokenizer,model,seq["full"],device)
            log_ls.append(log_l)
        pred = np.argmin(log_ls)
        print("Predicted: " + instance_template[int(pred)])
        preds.append(pred)
        probs.append(log_ls)
        i+=1
        print(str(i)+"/"+str(len(eval_features)))

    accuracy = simple_accuracy(np.array(preds), np.array([0]*len(preds)))

    instance_template = ["T", "F1", "F2", "F3", "F4", "F5"]


    results = {'accuracy': accuracy}

    output_eval_file = os.path.join(args.output_dir,
                                    "eval_results_gpt2_" + args.init_checkpoint.split("/")[-1].split(".")[0] + "_"
                                    + args.data_dir.split("/")[-1].split(".")[0] + ".txt")
    with open(output_eval_file, "w") as writer:
        for i in range(len(preds)):
            seqs = [eval_features[i].T["full"], eval_features[i].F1["full"], eval_features[i].F2["full"], eval_features[i].F3["full"], eval_features[i].F4["full"], eval_features[i].F5["full"]]
            for j in range(len(probs[i])):
                print(instance_template[j])
                writer.write(instance_template[j] + "\n")
                print("Log likelihood = " + str(probs[i][j]))
                writer.write("Log likelihood = " + str(probs[i][j])+"\n")
                print()
                writer.write("\n")
            print("Predicted " + instance_template[int(preds[i])])
            writer.write("Predicted " + instance_template[int(preds[i])]+"\n")
            print()
            print()
            writer.write("\n\n")
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    print(results)

if __name__ == "__main__":
    main()
