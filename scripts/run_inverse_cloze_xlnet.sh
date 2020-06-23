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

init_checkpoint="${PWD}/model/xlnet/model.ckpt"
master_port="8599"
seed=2
vocab_file="${PWD}/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt"
CONFIG_FILE="${PWD}/bert_config.json"
input_file="inverse_cloze.json"
while getopts g:p:c:i: option
do
 case "${option}"
 in
 g) gpu=${OPTARG};;
 p) master_port=${OPTARG};;
 c) init_checkpoint=${OPTARG};;
 i) input_file=${OPTARG};;
 esac
done

DATA_DIR=${PWD}/data/inverse_cloze/$input_file
OUT_DIR=${PWD}/results/inverse_cloze

mkdir -p $OUT_DIR

export CUDA_VISIBLE_DEVICES=$gpu


CMD="python run_inverse_cloze_xlnet.py "


CMD+="--do_lower_case "
CMD+="--data_dir $DATA_DIR "
CMD+="--xlnet_model xlnet-base-cased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--max_seq_length 128 "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$CONFIG_FILE "
CMD+="--output_dir $OUT_DIR "
suffix=$(basename "$init_checkpoint")
suffix="${suffix%.*}"
LOGFILE=$OUT_DIR/logfile_$suffix

echo $CMD
$CMD |& tee $LOGFILE
