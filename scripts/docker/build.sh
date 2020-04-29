#!/bin/bash
docker build . --rm -t bert \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=$(id -g)
