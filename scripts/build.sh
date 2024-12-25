#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

COMPOSE_CONFIG=$(mktemp -d)/build.yml

# compose config
cat > ${COMPOSE_CONFIG} <<EOF
services:
  tts-cuba:
    build:
      context: ${DOCKER_CONFIG_HOME}
      dockerfile: Dockerfile_tts_cuba
      args:
        - model_version=main
      tags:
        - ${HELM_image_repository}:cuba-${HELM_version}
      platforms:
        - linux/amd64
    image: ${HELM_image_repository}:cuba-latest
  tts-cpu:
    build:
      context: ${DOCKER_CONFIG_HOME}
      dockerfile: Dockerfile_tts_cpu
      args:
        - model_version=main
      tags:
        - ${HELM_image_repository}:cpu-${HELM_version}
      platforms:
        - linux/amd64
    image: ${HELM_image_repository}:cpu-latest
EOF

if [[ "${VERBOSE}" == "1" ]];then
  echo "[COMPOSE CONFIG]"
  cat ${COMPOSE_CONFIG}
fi

docker compose -f ${COMPOSE_CONFIG} -p ${PROJECT_NAME} build --push tts-cuba tts-cpu
