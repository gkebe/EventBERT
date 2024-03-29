# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.10-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/bert
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py html2text nltk progressbar onnxruntime\
 git+https://github.com/NVIDIA/dllogger

RUN apt-get install -y iputils-ping

COPY . .
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user