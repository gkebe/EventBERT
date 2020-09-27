from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel
bert2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained("results/NYT_midtuning/ckpt_294.pt", "gpt2")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
article = "thinks Schuh None None intrude None strangeness on 's None None Non meets None eye None did He None None did making sorts None."

input_ids = bert_tokenizer(article, return_tensors="pt").input_ids
output_ids = bert2gpt2.generate(input_ids)
print(gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True))