import argparse

def wiki70k_preprocess(filename, tuple_to_sen = True):
  f = open(filename, "r", encoding="utf8")
  paragraphs = f.read()
  paragraphs_articles = paragraphs.split("\n")
  paragraphs_lst = [list(i.split("<PARAGRAPH>")[1:]) for i in paragraphs_articles]
  paragraphs_lst_words = [[list(j.split(" ")) for j in i] for i in paragraphs_lst][:-1]
  paragraph_lst_sentences = '\n'.join([''.join([' '.join([' '.join(v[i:i+5][:4]) for i in range(0, len(v), 5)][:-1]) + ".\n" for v in j]) for j in paragraphs_lst_words])
  if tuple_to_sen:
      paragraph_lst_sentences = '\n'.join([' '.join([' '.join([' '.join(v[i:i+5][:4]) + ".\n" for i in range(0, len(v), 5)][:-1]) for v in j]) for j in paragraphs_lst_words]) 
  f.close()
  print("Articles: "+ str(sum([len(i) for i in paragraphs_lst_words])))
  print("Sentences: "+ str(sum([sum([len(j) for j in i]) for i in paragraphs_lst_words])))
  print("Words: "+ str(sum([sum([sum([len(k) for k in j]) for j in i]) for i in paragraphs_lst_words])))
  
  return paragraph_lst_sentences

def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file",
                        default="valid_pre.txt",
                        type=str,
                        required=False,
                        help="Specify a input filename!")
    parser.add_argument("--output_file",
                        default="valid.txt",
                        type=str,
                        required=False,
                        help="Specify a output filename!")
    parser.add_argument("--tuple_to_sen",
                        default=False,
                        type=bool,
                        required=False,
                        help="Specify a output filename!")
    return parser.parse_args()
args = parse_arguments()
val_sen = wiki70k_preprocess(args.input_file, args.tuple_to_sen)
text_file = open(args.output_file, "w", encoding="utf8")
n = text_file.write(val_sen)
text_file.close()