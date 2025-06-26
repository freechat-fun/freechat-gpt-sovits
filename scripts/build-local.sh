#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

mkdir "${DOCKER_CONFIG_HOME}/data"
cp -r "${PROJECT_PATH}/src/"* "${DOCKER_CONFIG_HOME}/data"

COMPOSE_CONFIG=$(mktemp -d)/build.yml

# compose config
cat > ${COMPOSE_CONFIG} <<EOF
services:
  cpu:
    build:
      context: ${DOCKER_CONFIG_HOME}
      dockerfile: Dockerfile_cpu
      platforms:
        - linux/arm64
    image: ${HELM_image_repository}:latest-cpu
EOF

if [[ "${VERBOSE}" == "1" ]];then
  echo "[COMPOSE CONFIG]"
  cat ${COMPOSE_CONFIG}
fi

export COMPOSE_BAKE=true
docker-compose -f ${COMPOSE_CONFIG} -p ${PROJECT_NAME} build ${ARGS[*]} cpu

rm -f ${COMPOSE_CONFIG}
rm -rf "${DOCKER_CONFIG_HOME}/data"
