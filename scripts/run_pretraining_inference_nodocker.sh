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
CODEDIR="${PWD}"

DATASET=model/wiki_70k # change this for other datasets
gpu="2,3"
num_gpus=2
eval_batch_size=14
master_port="8595"

while getopts g:p:n:d:b option 
do 
 case "${option}" 
 in 
 g) gpu=${OPTARG};; 
 p) master_port=${OPTARG};;
 n) num_gpus=${OPTARG};;
 d) DATASET=${OPTARG};;
 b) train_batch_size=${OPTARG};;
 esac 
done


DATA_DIR=$CODEDIR/${DATASET}/
BERT_CONFIG=bert_config.json
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$CODEDIR/model

if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Inference cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be loaded from $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

precision="fp16"
inference_mode="prediction"
model_checkpoint="-1"
inference_steps="-1"
create_logfile="true"
seed=42

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi


MODE=""
if [ "$inference_mode" = "eval" ] ; then
   MODE="--eval"
elif [ "$inference_mode" = "prediction" ] ; then
   MODE="--prediction"
else
   echo "Unknown <inference_mode> argument"
   exit -2
fi
export CUDA_VISIBLE_DEVICES=$gpu

echo $DATA_DIR
CMD="${PWD}"
CMD+="/run_pretraining_inference_nodocker.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --ckpt_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-base-uncased"
CMD+=" --eval_batch_size=$eval_batch_size"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$inference_steps"
CMD+=" --ckpt_step=$model_checkpoint"
CMD+=" --seed=$seed"
CMD+=" --vocab_file $CODEDIR/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --do_lower_case 1"
CMD+=" $PREC"
CMD+=" $MODE"

if [ "$num_gpus" -gt 1 ] ; then
   CMD="python3 -m torch.distributed.launch --master_port $master_port --nproc_per_node=$num_gpus $CMD"
else
   CMD="python3  $CMD"
fi

if [ "$create_logfile" = "true" ] ; then
  export GBS=$((eval_batch_size * num_gpus))
  printf -v TAG "pyt_bert_pretraining_inference_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x