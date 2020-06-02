# -*- coding: utf-8 -*-
"""bert-babble.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MxKZGtQ9SSBjTK5ArsZ5LKhkztzg52RV
"""
from modeling import BertModel, BertForMaskedLM, BertConfig
from tokenization import BertTokenizer
import numpy as np
import torch
import argparse
import math
import time
from collections import Counter
from nltk.util import ngrams
from nltk.translate import bleu_score as bleu

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

    def tokenize_batch(batch):
        return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

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
        batch = [seed_text + [MASK] * max_len + ["."] + [SEP] for _ in range(batch_size)]
        # if rand_init:
        #    for ii in range(max_len):
        #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

        return tokenize_batch(batch)
    def get_init_sequence(seed_texts, max_len, batch_size=1, seq_len=4, rand_init=False):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """
        batch = [seed_text + ([MASK] * max_len + ["."] + [SEP]) * seq_len for i in range(batch_size)]
        return tokenize_batch(batch)
    def printer(sent, should_detokenize=True):
        if should_detokenize:
            sent = detokenize(sent)
        print(" ".join(sent))

    """Let's call the actual generation function! We'll use the following settings
    - max_len (40): length of sequence to generate
    - top_k (100): at each step, sample from the top_k most likely words
    - temperature (1.0): smoothing parameter for the next word distribution. Higher means more like uniform; lower means more peaky
    - burnin (250): for non-sequential generation, for the first burnin steps, sample from the entire next word distribution, instead of top_k
    - max_iter (500): number of iterations to run for
    - seed_text (["CLS"]): prefix to generate for. We found it crucial to start with the CLS token; you can try adding to it
    """

    n_samples = 5
    batch_size = 5
    max_len = args.max_len
    top_k = 100
    temperature = 1.0
    generation_mode = args.mode
    leed_out_len = 5  # max_len
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

    def parallel_sequential_generation(seed_texts, batch_size=10, seq_len=4, max_len=15, top_k=0, temperature=None, max_iter=300,
                                       burnin=200,
                                       cuda=False, print_every=10, verbose=True):
        """ Generate for one random position at a timestep

        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        seed_len = len(seed_texts[0])
        batch = get_init_sequence(seed_texts, max_len, batch_size, seq_len)
        print(batch)
        print(seed_len)
        mask_indices = [i for i in range(len(batch[0])) if batch[0][i] == mask_id]
        for ii in range(max_iter):
            kk = mask_indices[np.random.randint(seed_len, len(mask_indices))] - seed_len
            print(mask_indices)
            print(kk)
            print(batch[0])
            print()
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = mask_id
            # mas = []
            # inp = batch
            # for seq in inp:
            #     while len(seq) < args.max_seq_length:
            #         seq.append(0)
            #     seq_mask = [float(i > 0) for i in seq]
            #     mas.append(seq_mask)
            # print(len(inp[0]))
            # print(len(mas[0]))
            # inp = torch.tensor(inp).cuda() if cuda else torch.tensor(inp)
            # mas = torch.tensor(mas).cuda() if cuda else torch.tensor(mas)
            # out = model(input_ids=inp, attention_mask=mas)
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

    def generate(n_samples, seed_text, batch_size=10, seq_len=4, max_len=25,
                 generation_mode="parallel-sequential",
                 sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
                 cuda=False, print_every=1):
        # main generation function to call
        sentences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()
        for batch_n in range(n_batches):
            if generation_mode == "parallel-sequential":
                batch = parallel_sequential_generation(seed_text, batch_size=batch_size, seq_len=seq_len, max_len=max_len, top_k=top_k,
                                                       temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                       cuda=cuda, verbose=False)
            elif generation_mode == "sequential":
                batch = sequential_generation(seed_text, batch_size=batch_size, seq_len=seq_len, max_len=max_len, top_k=top_k,
                                              temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                              cuda=cuda)
            elif generation_mode == "parallel":
                batch = parallel_generation(seed_text, batch_size=batch_size, seq_len=seq_len,
                                            max_len=max_len, top_k=top_k, temperature=temperature,
                                            sample=sample, max_iter=max_iter,
                                            cuda=cuda, verbose=False)

            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()

            sentences += batch
        return sentences
    # Choose the prefix context
    seed_text = seed_sentence.split(",")
    seq_len = args.seq_len - 1
    bert_sents = []
    seed_texts = [seed_text for _ in range(batch_size)]
    for i in range(seq_len):
        bert_sent = generate(n_samples, seed_text=seed_text, batch_size=batch_size, seq_len=2, max_len=max_len,
                          generation_mode=generation_mode,
                          sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                          cuda=cuda)
#        seed_text = seed_from_output(bert_sent)
#        bert_sents = append_output(bert_sents, bert_sent)


    for sent in bert_sents:
        printer(sent, should_detokenize=True)

    """
    Evaluation methods for unconditional generation aren't perfect. We'll measure the diversity of our generated samples via self-BLEU: we compute corpus BLEU where for each generated sentence, we compute BLEU treating the other sentences as references. We also compute the percentage of $n$-grams that are unique among the generations. We try some other strategies, including comparing to outside models, in our report, and you can see some of the code for that [here](https://github.com/kyunghyuncho/bert-gen/blob/master/bert-babble.ipynb).
    """
    def self_bleu(sents):
        return bleu.corpus_bleu([[s for (j, s) in enumerate(sents) if j != i] for i in range(len(sents))], sents)


    def get_ngram_counts(sents, max_n=4):
        size2count = {}
        for i in range(1, max_n + 1):
            size2count[i] = Counter([n for sent in sents for n in ngrams(sent, i)])
        return size2count


    def self_unique_ngrams(preds, max_n=4):
        # get # of pred ngrams with count 1
        pct_unique = {}
        pred_ngrams = get_ngram_counts(preds, max_n)
        for i in range(1, max_n + 1):
            n_unique = len([k for k, v in pred_ngrams[i].items() if v == 1])
            total = sum(pred_ngrams[i].values())
            pct_unique[i] = n_unique / total
        return pct_unique
    """# **Evaluation**"""

    max_n = 4
    print("BERT self-BLEU: %.2f" % (100 * self_bleu(bert_sents)))

    pct_uniques = self_unique_ngrams(bert_sents, max_n)
    for i in range(1, max_n + 1):
        print("BERT unique %d-grams relative to self: %.2f" % (i, 100 * pct_uniques[i]))
if __name__ == "__main__":
    main()