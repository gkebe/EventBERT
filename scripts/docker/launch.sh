#!/bin/bash

docker run -it --rm \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user $(id -u):$(id -g) \
  -v ${PWD}:/workspace/bert \
  bert bash
