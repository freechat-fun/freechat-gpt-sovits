#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

docker run -it --rm --name local-gpt-sovits \
  -p 9880:9880 \
  -v ${PROJECT_PATH}/local-data/server:/workspace/data \
  ${HELM_image_repository}:latest-cpu