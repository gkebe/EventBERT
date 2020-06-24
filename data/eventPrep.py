import argparse

def tup(i, limit):
    if i < limit-1:
        return " <TUP>"
    return ""
def wiki70k_preprocess(filename, tuple_to_sen = False, keep_label = False, add_tup=False):
  f = open(filename, "r", encoding="utf8")
  paragraphs = f.read()
  paragraphs_articles = paragraphs.split("\n")
  paragraphs_lst = [list(i.split("<PARAGRAPH>")[1:]) for i in paragraphs_articles]
  paragraphs_lst_words = [[list(j.split(" ")) for j in i] for i in paragraphs_lst][:-1]
  if add_tup:
    paragraph_lst_sentences = '\n'.join(
          [''.join([' '.join([' '.join(v[i:i + 5][:4]) + tup(i,len(v)-5) for i in range(0, len(v), 5)][:-1]) + ".\n" for v in j]) for j in
           paragraphs_lst_words])
  else:
    paragraph_lst_sentences = '\n'.join([''.join([' '.join([' '.join(v[i:i+5][:4]) for i in range(0, len(v), 5)][:-1]) + ".\n" for v in j]) for j in paragraphs_lst_words])
  paragraph_lst_labels = []
  if keep_label:
    paragraph_lst_labels = '\n'.join([''.join([', '.join([' '.join(v[i:i+5][4:5]) for i in range(0, len(v), 5)][:-1]) + "\n" for v in j]) for j in paragraphs_lst_words])
  if tuple_to_sen:
    paragraph_lst_sentences = '\n'.join([''.join([''.join([' '.join(v[i:i+5][:4]) + ".\n" for i in range(0, len(v), 5)][:-1]) for v in j]) for j in paragraphs_lst_words]) 
    if keep_label:
        paragraph_lst_labels = '\n'.join([''.join([''.join([' '.join(v[i:i+5][4:5]) + "\n" for i in range(0, len(v), 5)][:-1]) for v in j]) for j in paragraphs_lst_words])
  f.close()
  print("Articles: "+ str(sum([len(i) for i in paragraphs_lst_words])))
  print("Sentences: "+ str(sum([sum([len(j) for j in i]) for i in paragraphs_lst_words])))
  print("Words: "+ str(sum([sum([sum([len(k) for k in j]) for j in i]) for i in paragraphs_lst_words])))
  
  return paragraph_lst_sentences, paragraph_lst_labels

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
    parser.add_argument("--keep_label",
                        default=False,
                        type=bool,
                        required=False,
                        help="Specify a output filename!")
    parser.add_argument("--add_tup",
                        default=False,
                        type=bool,
                        required=False,
                        help="Specify a output filename!")
    return parser.parse_args()
args = parse_arguments()
sentences, labels = wiki70k_preprocess(args.input_file, args.tuple_to_sen, args.keep_label, args.add_tup)
text_file = open(args.output_file, "w", encoding="utf8")
n = text_file.write(sentences)
text_file.close()
if len(labels)>0:
    idx = args.output_file.index(".txt")
    labels_title = args.output_file[:idx] + "_labels" + args.output_file[idx:]
    text_file = open(labels_title, "w", encoding="utf8")
    n = text_file.write(labels)
    text_file.close()