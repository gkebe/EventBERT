import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import os
import csv
import sys
import numpy as np
import string
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler,TensorDataset)
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def metrics_frame(preds, labels, label_names):
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)
    cr = classification_report(labels, preds, labels=list(range(len(label_names))), target_names=label_names)
    model_metrics = {"Precision, Micro": precision_micro, "Precision, Macro": precision_macro,
                     "Recall, Micro": recall_micro, "Recall, Macro": recall_macro,
                     "F1 score, Micro": f1_micro, "F1 score, Macro": f1_macro, "Confusion matrix": cm, "Classification report": cr}
    return model_metrics
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

    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell.decode('utf-8') for cell in line)
                lines.append(line)
            return lines

class FramesProcessor(DataProcessor):
    """Processor for the Frames data set (Wiki_70k version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        train_examples = self.get_train_examples(data_dir)
        dev_examples = self.get_dev_examples(data_dir)
        test_examples = self.get_test_examples(data_dir)
        return list(set([i.label for i in train_examples+dev_examples+test_examples]))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            sentence = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=sentence, label=label))
        return examples
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 300

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
class MeanEmbeddingVectorizer(object):
    def __init__(self, glove):
        self.glove = glove
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        #self.dim = len(word2vec.itervalues().next())
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.glove[w] for w in words if w in self.glove]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
            
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def glove_features(examples, label_list, data_dir, glove):

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    glovevec = TfidfEmbeddingVectorizer(glove)
    glovevec.fit([i.text_a for i in examples])
    for (ex_index, example) in enumerate(examples):
        tokens = example.text_a.split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table).lower() for w in tokens]
        label_id = label_map[example.label]
        input_ids = glovevec.transform([list(tokens)]).flatten()
        features.append(
                InputFeatures(input_ids=input_ids,
                              label_id=label_id))
    return features

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

   
#train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
#test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

data_dir = "./model/wiki_70k_frames"
processor = FramesProcessor()
train_examples = processor.get_train_examples(data_dir)
test_examples = processor.get_test_examples(data_dir)

batch_size = 100
n_iters = 11325

label_list = processor.get_labels(data_dir=data_dir)
num_labels = len(label_list)
input_dim = 300
output_dim = num_labels
lr_rate = 0.001

glove = loadGloveModel("./data/glove/glove.6B.300d.txt")

train_features = glove_features(train_examples, label_list, data_dir, glove)
test_features = glove_features(test_examples, label_list, data_dir, glove)

train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.float32)
train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.float32)
test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

train_data = TensorDataset(train_input_ids, train_label_ids)
test_data = TensorDataset(test_input_ids, test_label_ids)

epochs = n_iters / (len(train_data) / batch_size)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model = LogisticRegression(input_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

iter = 0
for epoch in range(int(epochs)):
    for i, (inputs, labels) in enumerate(train_loader):
        input_ids = Variable(inputs)
        label_id = Variable(labels)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter+=1
        if iter%500==0:
            preds = None
            out_label_ids = None 
            # calculate Accuracy
            correct = 0
            total = 0
            for input_ids, labels in test_loader:
                input_ids = Variable(input_ids)
                outputs = model(input_ids)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                if preds is None:
                    preds = predicted
                    out_label_ids = labels
                else:
                    preds = np.append(preds, predicted, axis=0)
                    out_label_ids = np.append(out_label_ids, labels, axis=0)
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
        results = {}
        np.set_printoptions(threshold=sys.maxsize)
        result = metrics_frame(preds, out_label_ids, label_list)
        results.update(result)
        print(results)
        output_eval_file = os.path.join("./data/", "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))
