import nlp
import logging
import torch
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")
checkpoint = torch.load("model/ckpt_294.pt", map_location="cpu")
checkpoint_gpt = torch.load("model/gpt.bin", map_location="cpu")
model.encoder.load_state_dict(checkpoint['model'], strict=False)
model.decoder.load_state_dict(checkpoint_gpt, strict=False)

# cache is currently not supported by EncoderDecoder framework
model.decoder.config.use_cache = False
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token

# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# set decoding params
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

# load train and validation data
# train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
# val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")
with open("model/gen_data_wiki.pkl",'rb') as f:
    dataset = pickle.load(f)
X = [i[0] for i in dataset]
y = [i[1] for i in dataset]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# load rouge for validation
rouge = nlp.load_metric("rouge", experiment_id=1)

encoder_length = 512
decoder_length = 128
batch_size = 32


class GenDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        input_ids = self.data["input_ids"][index]
        attention_mask = self.data["attention_mask"][index]
        decoder_input_ids = self.data["decoder_input_ids"][index]
        labels = self.data["labels"][index]
        decoder_attention_mask = self.data["decoder_attention_mask"][index]


        return {"input_ids" : input_ids, "attention_mask" : attention_mask, "decoder_input_ids" : decoder_input_ids, "labels" : labels, "decoder_attention_mask" : decoder_attention_mask}

# map data correctly
def map_to_encoder_decoder_inputs(articles, highlights):    # Tokenizer will automatically set [BOS] <text> [EOS]
    # use bert tokenizer here for encoder
    inputs = bert_tokenizer(articles, padding="max_length", truncation=True, max_length=encoder_length)
    # force summarization <= 128
    outputs = gpt2_tokenizer(highlights, padding="max_length", truncation=True, max_length=decoder_length)
    batch = dict()
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask

    # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
    batch["labels"] = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
    ]

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return GenDataset(batch)

train_dataset = map_to_encoder_decoder_inputs(X_train, y_train)
val_dataset = map_to_encoder_decoder_inputs(X_test, y_test)

# make train dataset ready

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
#    predict_from_generate=True,
    do_train=True,
    logging_steps=1000,
    save_steps=1000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=10,
    fp16=True,
)

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
    label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

print(train_dataset[100])
# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# start training
trainer.train()

torch.save(model, 'model/bert2gpt2.pt')