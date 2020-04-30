#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

# python3 /workspace/bert/data/eventPrep.py --input_file wiki_70k/step1/train.txt --output_file wiki_70k/step2/train.txt

# python3 /workspace/bert/data/eventPrep.py --input_file wiki_70k/step1/valid.txt --output_file wiki_70k/step2/valid.txt

python3 ${PWD}/eventPrep.py --input_file wiki_70k/step1/test.txt --output_file wiki_70k/step2/test_with_labels.txt --tuple_to_sen True --keep_label True

# awk '{filename = "wiki_70k/step3/wiki_70k_training_" int((NR-1)/80000) ".txt"; print >> filename}' wiki_70k/step2/train.txt

# awk '{filename = "wiki_70k/step3/wiki_70k_valid_" int((NR-1)/80000) ".txt"; print >> filename}' wiki_70k/step2/valid.txt

awk '{filename = "wiki_70k/step3/wiki_70k_test_with_labels_" int((NR-1)/80000) ".txt"; print >> filename}' wiki_70k/step2/test_with_labels_labels.txt
awk '{filename = "wiki_70k/step3/wiki_70k_test_with_labels_" int((NR-1)/80000) ".txt"; print >> filename}' wiki_70k/step2/test_with_labels.txt

#python3 /workspace/bert/data/eventHdf5.py --max_seq_length 128  \
 --max_predictions_per_seq 20 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --do_lower_case 1 --n_training_shards 210 --n_test_shards 1

#python3 /workspace/bert/data/eventHdf5.py --max_seq_length 512  \
 --max_predictions_per_seq 80 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --do_lower_case 1 --n_training_shards 210 --n_test_shards 1


# Download
#python3 /workspace/bert/data/bertPrep.py --action download --dataset bookscorpus
#python3 /workspace/bert/data/bertPrep.py --action download --dataset wikicorpus_en

#python3 /workspace/bert/data/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab

#python3 /workspace/bert/data/bertPrep.py --action download --dataset squad
#python3 /workspace/bert/data/bertPrep.py --action download --dataset mrpc


# Properly format the text files
#python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset bookscorpus
#python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset wiki_70k
#python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset wikicorpus_en


# Shard the text files (group wiki+books then shard)
#python3 /workspace/bert/data/bertPrep.py --action sharding --dataset wiki_70k --n_training_shards 4 --n_test_shards 1


# Create HDF5 files Phase 1
#python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset bookscorpus --max_seq_length 128 \
# --max_predictions_per_seq 20 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1  --n_training_shards 70 --n_test_shards 25


# Create HDF5 files Phase 2
#python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset bookscorpus --max_seq_length 512 \
# --max_predictions_per_seq 80 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1  --n_training_shards 70 --n_test_shards 25
 
# python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset wiki_70k --max_seq_length 128  \
# --max_predictions_per_seq 20 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --do_lower_case 1 --n_training_shards 70 --n_test_shards 25
 
# python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset wiki_70k --max_seq_length 512  \
# --max_predictions_per_seq 80 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --do_lower_case 1 --n_training_shards 70 --n_test_shards 25