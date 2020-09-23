import re
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
import os
import glob
from inverse_cloze import inverse_cloze
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    args = parser.parse_args()
    results =dict()
    for file in list(glob.glob('results/'+args.dataset+'/eval_results_alt_ckpt_*_cloze_dataset_weber.txt')):
        epoch = int(re.findall("\d+", file)[0])
        with open(file, "r") as f:
            text = f.read()
        accuracy = float(re.findall("\d+\.\d+",re.findall("accuracy = \d+\.\d+", text)[0])[0])
        mrr = float(re.findall("\d+\.\d+", re.findall("MRR = \d+\.\d+", text)[0])[0])
        results[epoch] = (accuracy, mrr)
    for i in sorted(results.items()):
        print(str(i[0])+","+str(i[1][0])+","+str(i[1][1]))
if __name__ == "__main__":
    main()