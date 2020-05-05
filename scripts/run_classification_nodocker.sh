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

DATA_DIR=${PWD}/model/wiki_70k_frames
OUT_DIR=${PWD}/results/wiki_70k_frames

mkdir -p $OUT_DIR

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint="${PWD}/model/ckpt_0.pt"
mode="train test"
max_steps="-1.0" # if < 0, has no effect
batch_size="32"
learning_rate="2e-5"
precision="fp16"
num_gpu=2
gpu="2,3"
master_port="8599"
epochs="4"
warmup_proportion="0.1"
seed=1
vocab_file="${PWD}/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt"
CONFIG_FILE="${PWD}/bert_config.json"

while getopts g:p:c:n:d:b:e:l:o:m: option
do
 case "${option}"
 in
 g) gpu=${OPTARG};;
 p) master_port=${OPTARG};;
 c) init_checkpoint=${OPTARG};;
 n) num_gpu=${OPTARG};;
 d) DATA_DIR=${OPTARG};;
 b) batch_size=${OPTARG};;
 e) epochs=${OPTARG};;
 l) learning_rate=${OPTARG};;
 o) OUT_DIR=${OPTARG};;
 m) mode=${OPTARG};;
 esac
done



if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=$gpu
  mpi_command=""
else
  export CUDA_VISIBLE_DEVICES=$gpu
  mpi_command=" -m torch.distributed.launch --master_port $master_port --nproc_per_node=$num_gpu"
fi


CMD="python $mpi_command run_classification_huggingface.py "
CMD+="--task_name frames "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [ "$mode" == "eval" ] ; then
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi
if [ "$mode" == "train eval" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi
if [ "$mode" == "train test" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
  CMD+="--do_test "
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--do_lower_case "
CMD+="--data_dir $DATA_DIR "
CMD+="--bert_model bert-base-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$CONFIG_FILE "
CMD+="--output_dir $OUT_DIR "
CMD+="$use_fp16"

LOGFILE=$OUT_DIR/logfile

echo $CMD
$CMD |& tee $LOGFILE
