# -*- coding: utf-8 -*-
"""bert-babble.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MxKZGtQ9SSBjTK5ArsZ5LKhkztzg52RV
"""
from transformers import BertTokenizer, BertModel
from modeling import BertForMaskedLM, BertConfig
import numpy as np
import torch
import argparse
import math
import time
from collections import Counter
from nltk.util import ngrams
from nltk.translate import bleu_score as bleu
import os
from itertools import chain

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
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrained on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument("--mode",
                        default="sequential",
                        type=str,
                        required=False,
                        help="The generation method")
    parser.add_argument("--max_len",
                        default=4,
                        type=int,
                        help="The number of tokens per sentence")
    parser.add_argument("--seq_len",
                        default=5,
                        type=int,
                        help="The number of sentences in sequence")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model")
    parser.add_argument('--seed_sentence',
                        type=str,
                        default="[CLS]",
                        help="The seed used for generation")

    args = parser.parse_args()
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = BertForMaskedLM(config)
    print("USING CHECKPOINT from", args.init_checkpoint)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"], strict=False)
    print("USED CHECKPOINT from", args.init_checkpoint)
    # Load pre-trained model (weights)
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=128)
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

    def tokens_len(text):
        tokens = [tokenizer.tokenize(i) for i in text]
        ids = list(chain.from_iterable([tokenizer.convert_tokens_to_ids(i) for i in tokens]))
        return len(ids)

    def tokenize_batch(batch):

        tokens = [[tokenizer.tokenize(i) for i in j] for j in batch]
        ids =[list(chain.from_iterable([tokenizer.convert_tokens_to_ids(i) for i in sent])) for sent in tokens]
        return ids

    def untokenize_batch(batch):
        return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

    def detokenize(sent):
        """ Roughly detokenizes (mainly undoes wordpiece) """
        new_sent = []
        for i, tok in enumerate(sent):
            if tok.startswith("##"):
                new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
            else:
                new_sent.append(tok)
        return new_sent

    def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]

        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k
        """
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx

    def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """
        batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
        # if rand_init:
        #    for ii in range(max_len):
        #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

        return tokenize_batch(batch)
    def get_init_sequence(seed_text, max_len, batch_size=1, seq_len=4, rand_init=False):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """
        batch = [seed_text + ([MASK] * max_len + ["."] + [SEP]) * seq_len for _ in range(batch_size)]
        return tokenize_batch(batch)
    def printer(sent, should_detokenize=True):
        if should_detokenize:
            sent = detokenize(sent)
        print(" ".join(sent))
    def printer_sequence(sequence):
        dot_=""
        sent_ = ""
        seq_=""
        for sent in sequence:
            sent_ = [i for i in sent if i != "[SEP]" and i != "[CLS]"]
            if "." in sent_:
                dot_=". "
            sent_ = " ".join([i for i in sent_ if i!="."])+dot_

        seq_+=sent_
        return seq_
    def follow_up(sent, seed_len, should_detokenize=True):
        if should_detokenize:
            sent = detokenize(sent)
        return ["[CLS]"] +sent[seed_len:]

    """Let's call the actual generation function! We'll use the following settings
    - max_len (40): length of sequence to generate
    - top_k (100): at each step, sample from the top_k most likely words
    - temperature (1.0): smoothing parameter for the next word distribution. Higher means more like uniform; lower means more peaky
    - burnin (250): for non-sequential generation, for the first burnin steps, sample from the entire next word distribution, instead of top_k
    - max_iter (500): number of iterations to run for
    - seed_text (["CLS"]): prefix to generate for. We found it crucial to start with the CLS token; you can try adding to it
    """

    n_samples = 10
    batch_size = 10
    max_len = 20
    top_k = 100
    temperature = 1.0
    burnin = 250
    sample = True
    max_iter = 500
    seed_sentence =args.seed_sentence
    """This is the meat of the algorithm. The general idea is
    1. start from all masks
    2. repeatedly pick a location, mask the token at that location, and generate from the probability distribution given by BERT
    3. stop when converged or tired of waiting

    We consider three "modes" of generating:
    - generate a single token for a position chosen uniformly at random for a chosen number of time steps
    - generate in sequential order (L->R), one token at a time
    - generate for all positions at once for a chosen number of time steps

    The `generate` function wraps and batches these three generation modes. In practice, we find that the first leads to the most fluent samples.
    """

    # Generation modes as functions

    def parallel_sequential_generation(seed_text, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                       cuda=False, print_every=10, verbose=True):
        """ Generate for one random position at a timestep

        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        seed_len = tokens_len(seed_text)
        batch = get_init_text(seed_text, max_len, batch_size)

        for ii in range(max_iter):
            kk = np.random.randint(0, max_len)
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = mask_id
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = model(inp)
            topk = top_k if (ii >= burnin) else 0
            idxs = generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = idxs[jj]

            if verbose and np.mod(ii + 1, print_every) == 0:
                for_print = tokenizer.convert_ids_to_tokens(batch[0])
                for_print = for_print[:seed_len + kk + 1] + ['(*)'] + for_print[seed_len + kk + 1:]
                print("iter", ii + 1, " ".join(for_print))

        return untokenize_batch(batch)

    def parallel_generation(seed_text, batch_size=10, seq_len=4, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True,
                            cuda=False, print_every=10, verbose=True):
        """ Generate for all positions at each time step """
        seed_len = len(seed_text)
        batch = get_init_sequence(seed_text, max_len, batch_size, seq_len)

        for ii in range(max_iter):
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = model(inp)
            for kk in range(max_len):
                idxs = generate_step(out, gen_idx=seed_len + kk, top_k=top_k, temperature=temperature, sample=sample)
                for jj in range(batch_size):
                    batch[jj][seed_len + kk] = idxs[jj]

            if verbose and np.mod(ii, print_every) == 0:
                print("iter", ii + 1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))

        return untokenize_batch(batch)

    def sequential_generation(seed_text, batch_size=10, seq_len=4, max_len=15, leed_out_len=15,
                              top_k=0, temperature=None, sample=True, cuda=False):
        """ Generate one word at a time, in L->R order """
        seed_len = len(seed_text)
        batch = get_init_sequence(seed_text, max_len, batch_size, seq_len)

        for ii in range(max_len):
            inp = [sent[:seed_len + ii + leed_out_len] + [sep_id] for sent in batch]
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = model(inp)
            idxs = generate_step(out, gen_idx=seed_len + ii, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len + ii] = idxs[jj]

        return untokenize_batch(batch)

    def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25,
                 sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
                 cuda=False, print_every=1):
        # main generation function to call
        sentences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()
        for batch_n in range(n_batches):
            batch = parallel_sequential_generation(seed_text, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                   cuda=cuda, verbose=False)

            # batch = sequential_generation(seed_text, batch_size=20, max_len=max_len, top_k=top_k, temperature=temperature, leed_out_len=leed_out_len, sample=sample)
            # batch = parallel_generation(seed_text, max_len=max_len, top_k=top_k, temperature=temperature, sample=sample, max_iter=max_iter)

            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()

            sentences += batch
        return sentences
    # Choose the prefix context
    seed_text = seed_sentence.split(",")
    bert_sents = generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                          sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                          cuda=True)

    for i in range(len(bert_sents)):
        printer(bert_sents[i], should_detokenize=True)

if __name__ == "__main__":
    main()