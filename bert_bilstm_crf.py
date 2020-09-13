import torch
import logging
from transformers import BertModel, BertTokenizer
from transformers import *
from typing import List
from itertools import chain
import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from tqdm import tqdm
import sklearn

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

logger = logging.getLogger(__name__)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


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


def select_field(features, field):
    """As the output is dic, return relevant field"""
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def create_examples(_list, set_type="train"):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(_list):
        guid = "%s-%s" % (set_type, i)
        text_a = line
        # text_b = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a))
    return examples

class BertBiLSTM_CRF(nn.Module):
    def __init__(self,
                 tag_to_idx,
                 hidden_dim,
                 device,
                 pretrained_weights='bert-base-uncased',
                 tokenizer_class=BertTokenizer,
                 model_class=BertModel,
                 max_seq_len=128,
                 init_checkpoint="model/ckpt_794.pt"):
        super().__init__()

        self.tag_to_ix = tag_to_idx
        self.pretrained_weights = pretrained_weights
        self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.model = self.model_class.from_pretrained(pretrained_weights)
        self.model.load_state_dict(torch.load(init_checkpoint, map_location='cpu')["model"], strict=False)
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_idx)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.lstm = nn.LSTM(768, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.device = device
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size, device=self.device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    #         for param in self.model.parameters():
    #             param.requires_grad = False
    # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    # model = BertModel.from_pretrained(pretrained_weights)

    # Clause by clause when getting BERT embeddings
    # Performance as training data varies (10%$25% of the paragraphs)
    #
    def bert_embeddings(self, all_input_ids, all_input_mask, all_segment_ids) -> torch.tensor:
        ##### Check if this is actually doing what you think it's doing

        #         examples = create_examples(raw_text)

        #         features = convert_examples_to_features(
        #             examples, self.tokenizer, self.max_seq_len, True)

        #         all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=self.device)
        #         all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
        #         all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)

        #         cls_embeddings_ = []
        #         i = 0
        #####     Check if this
        #         while all_input_ids.size()[0] > 100:
        #             i+=1
        #             all_input_ids_ = all_input_ids[:100]
        #             all_input_mask_ = all_input_mask[:100]
        #             all_segment_ids_ = all_segment_ids[:100]
        #             print(i)
        #             cls_embeddings_.append(self.model(all_input_ids_,all_input_mask_,all_segment_ids_)[1])

        #             all_input_ids = all_input_ids[100:]
        #             all_input_mask = all_input_mask[100:]
        #             all_segment_ids = all_segment_ids[100:]
        #         cls_embeddings_.append(self.model(all_input_ids,all_input_mask,all_segment_ids)[1])
        #         cls_embeddings = torch.cat(cls_embeddings_)
        cls_embeddings = self.model(all_input_ids, all_input_mask, all_segment_ids)[1]
        return cls_embeddings

    #     def bert_embeddings(self, all_input_ids, all_input_mask, all_segment_ids):
    #         cls_embeddings = self.model(all_input_ids,all_input_mask,all_segment_ids)[1]
    #         return cls_embeddings

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, device=self.device),
                torch.randn(2, 1, self.hidden_dim // 2, device=self.device))

    def _get_lstm_features(self, all_input_ids, all_input_mask, all_segment_ids):
        self.hidden = self.init_hidden()
        embeds = self.bert_embeddings(all_input_ids, all_input_mask, all_segment_ids)
        steps = embeds.size()[0]
        embeds = embeds.view(steps, 1, -1)
        embeds = self.dropout(embeds)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(steps, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat([i.to(self.device) for i in viterbivars_t]) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    #     def neg_log_likelihood(self, all_input_ids, all_input_mask, all_segment_ids, tags):
    #         feats = self._get_lstm_features(all_input_ids, all_input_mask, all_segment_ids)

    # Use cls tokens to predict the layer
    # Compare to 80% accuracy
    def forward(self, all_input_ids, all_input_mask, all_segment_ids,
                tags=None):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(all_input_ids, all_input_mask, all_segment_ids)
        if tags is None:
            score, tag_seq = self._viterbi_decode(lstm_feats)
            return score, tag_seq
        else:
            forward_score = self._forward_alg(lstm_feats)
            gold_score = self._score_sentence(lstm_feats, tags)
            return forward_score - gold_score

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The train_path.jsonl file.")

    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The eval_path.jsonl file.")

    # parser.add_argument("--output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    # parser.add_argument("--train_batch_size",
    #                     default=32,
    #                     type=int,
    #                     help="Total batch size for training.")
    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="GPU to be used.")
    # parser.add_argument("--eval_batch_size",
    #                     default=32,
    #                     type=int,
    #                     help="Total batch size for eval.")
    parser.add_argument("--lr",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs",
                        default=4,
                        type=int,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--prob_threshold",
    #                     default=0.5,
    #                     type=float,
    #                     help="Probabilty threshold for multiabel classification.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # parser.add_argument("--do_lower_case",
    #                     action='store_true',
    #                     help="Set this flag if you are using an uncased model.")
    #
    # parser.add_argument('--vocab_file',
    #                     type=str, default=None, required=True,
    #                     help="Vocabulary mapping/file BERT was pretrainined on")
    # parser.add_argument("--config_file",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The BERT model config")

    args = parser.parse_args()
    gpu = str(args.gpu)

    tr_data = []
    te_data = []
    with open(args.train_file, 'rb') as infile:
        for line in infile.readlines():
            tr_data.append((json.loads(line)["sentences"], json.loads(line)["labels"]))

    with open(args.test_file, 'rb') as infile:
        for line in infile.readlines():
            te_data.append((json.loads(line)["sentences"], json.loads(line)["labels"]))
    # Make up some training data
    train_data = [(["[CLS] " + i + " [SEP]" for i in sentences], [j for j in labels]) for sentences, labels in tr_data]
    test_data = [(["[CLS] " + i + " [SEP]" for i in sentences], [j for j in labels]) for sentences, labels in te_data]

    unique_tags = list(set(list(chain.from_iterable([i[1] for i in train_data + test_data]))))
    device_name = f'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # tr_data = list(chain.from_iterable([masc_data[i][2] for i in masc_data.keys() if train_test_split[i] == "train"]))
    # te_data = list(chain.from_iterable([masc_data[i][2] for i in masc_data.keys() if train_test_split[i] == "test"]))
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    HIDDEN_DIM = 522

    tag_to_ix = {}
    for sentences, tags in train_data:
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    tag_to_ix.update({START_TAG: len(tag_to_ix), STOP_TAG: len(tag_to_ix) + 1})

    all_features = []
    all_tags = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for sentences, tags in train_data:
        examples = create_examples(sentences)
        all_features.append(convert_examples_to_features(examples, tokenizer, 64, True))
        all_tags.append([tag_to_ix[t] for t in tags])

    all_labels = [torch.tensor([label for label in labels], dtype=torch.long, device=device) for labels in all_tags]
    all_input_ids = [torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device) for features in
                     all_features]
    all_input_mask = [torch.tensor([f.input_mask for f in features], dtype=torch.long, device=device) for features in
                      all_features]
    all_segment_ids = [torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=device) for features in
                       all_features]

    all_features_t = []
    all_tags_t = []
    for sentences, tags in test_data:
        examples = create_examples(sentences)
        all_features_t.append(convert_examples_to_features(examples, tokenizer, 128, True))
        all_tags_t.append([tag_to_ix[t] for t in tags])

    all_labels_t = [torch.tensor([label for label in labels], dtype=torch.long, device=device) for labels in all_tags_t]
    all_input_ids_t = [torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device) for features in
                       all_features_t]
    all_input_mask_t = [torch.tensor([f.input_mask for f in features], dtype=torch.long, device=device) for features in
                        all_features_t]
    all_segment_ids_t = [torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=device) for features
                         in all_features_t]

    model = BertBiLSTM_CRF(tag_to_ix, HIDDEN_DIM, device, max_seq_len=args.max_seq_len, init_checkpoint=args.init_checkpoint)
    model.to(device)

    # bert_param_optimizer = list(model.model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.0}
    # ]

    optimizer = AdamW(model.parameters(), lr=args.lr)

    epochs = args.num_epochs

    for epoch in tqdm(range(epochs), desc="Epoch"):  # again, normally you would NOT do 300 epochs, it is toy data
        for i in tqdm(range(len(all_input_ids)), desc="Instance"):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            model.train()
            #         examples = create_examples(sentences)

            #         features = convert_examples_to_features(
            #             examples, tokenizer, max_seq_len, True)

            #         all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device)
            #         all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=device)
            #         all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=device)
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            #         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long, device=device)
            # Step 3. Run our forward pass.
            loss = model(all_input_ids[i], all_input_mask[i], all_segment_ids[i], all_labels[i])

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()+
            loss.backward()
            optimizer.step()
        print(loss.item())
        torch.save(model.state_dict(), "model/bert_bilstm_crf_frames_" + str(epoch) + "_epochs.pt")
        torch.save(model.model.state_dict(), "model/bert_frames_" + str(epoch) + "_epochs.pt")
        with torch.no_grad():
            model.eval()
            predictions = []
            labels = []
            for i in tqdm(range(len(all_input_ids_t)), desc="Instance"):
                # turn them into Tensors of word indices.
                # Step 3. Run our forward pass.
                preds = model(all_input_ids_t[i], all_input_mask_t[i], all_segment_ids_t[i])[1]
                predictions.append(preds)
                labels.append(all_labels_t[i].cpu().detach().numpy().tolist())

            y_true = list(chain.from_iterable(predictions))
            y_pred = list(chain.from_iterable(labels))
            print(sklearn.metrics.accuracy_score(y_true, y_pred))
            print(sklearn.metrics.f1_score(y_true, y_pred, average="macro"))

if __name__ == "__main__":
    main()