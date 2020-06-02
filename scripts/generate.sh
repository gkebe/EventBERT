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


echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint="${PWD}/checkpoints/bert-base.pt"
masking="none"
gpu="0"
master_port="8599"
seed_sentence="[CLS]"
seq_len=5
vocab_file="${PWD}/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt"
CONFIG_FILE="${PWD}/bert_config.json"
DATA_DIR=${PWD}/data/generation
OUT_DIR=${PWD}/results/generation
interact=true
n_append_mask=0
decoding_strategy="sequential"
token_strategy="argmax"

while getopts g:p:c:d:o:m:s:l:i:n:s:t: option
do
 case "${option}"
 in
 g) gpu=${OPTARG};;
 p) master_port=${OPTARG};;
 c) init_checkpoint=${OPTARG};;
 d) DATA_DIR=${OPTARG};;
 o) OUT_DIR=${OPTARG};;
 m) masking=${OPTARG};;
 s) seed_sentence=${OPTARG};;
 l) seq_len=${OPTARG};;
 i) interact=${OPTARG};;
 n) n_append_mask=${OPTARG};;
 s) decoding_strategy=${OPTARG};;
 t) token_strategy=${OPTARG};;
 esac
done
mkdir -p $OUT_DIR


export CUDA_VISIBLE_DEVICES=$gpu

if [ "$interact" == "true" ] ; then
   interact="--interact"
fi

CMD="python generate.py "
CMD+="--masking $masking "
CMD+="--do_lower_case "
CMD+="--bert_version bert-base-uncased "
CMD+="--seed_sentence=$seed_sentence "
CMD+="--n_append_mask=$n_append_mask "
CMD+="--decoding_strategy=$decoding_strategy "
CMD+="--token_strategy=$token_strategy "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--max_seq_length 128 "
#CMD+="--max_len 4 "
#CMD+="--seq_len $seq_len "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$CONFIG_FILE "
CMD+="--output_dir $OUT_DIR "
CMD+=" $interact"
suffix=$(basename "$init_checkpoint")
suffix="${suffix%.*}"
LOGFILE=$OUT_DIR/logfile_$suffix

echo $CMD
$CMD |& tee $LOGFILE
