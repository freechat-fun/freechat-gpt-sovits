#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

docker run -it --rm --name local-gpt-sovits \
  -p 9871:9871 \
  -p 9872:9872 \
  -p 9873:9873 \
  -p 9874:9874 \
  -p 9880:9880 \
  --entrypoint "python webui.py" \
  ${HELM_image_repository}:latest-cpu