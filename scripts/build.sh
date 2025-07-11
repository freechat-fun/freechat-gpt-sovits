#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

mkdir -p "${DOCKER_CONFIG_HOME}/data/nls"
cp -r "${PROJECT_PATH}/src/"* "${DOCKER_CONFIG_HOME}/data/"

COMPOSE_CONFIG=$(mktemp -d)/build.yml

# compose config
cat > ${COMPOSE_CONFIG} <<EOF
services:
  cu128:
    build:
      context: ${DOCKER_CONFIG_HOME}
      dockerfile: Dockerfile_cu128
      tags:
        - ${HELM_image_repository}:${HELM_version}-cu128
        - ${HELM_image_repository}:latest-cu128
      platforms:
        - linux/amd64
        - linux/arm64
    image: ${HELM_image_repository}:latest
  cpu:
    build:
      context: ${DOCKER_CONFIG_HOME}
      dockerfile: Dockerfile_cpu
      tags:
        - ${HELM_image_repository}:${HELM_version}-cpu
        - ${HELM_image_repository}:latest-cpu
      platforms:
        - linux/amd64
        - linux/arm64
    image: ${HELM_image_repository}:latest-cpu
EOF

if [[ "${VERBOSE}" == "1" ]];then
  echo "[COMPOSE CONFIG]"
  cat ${COMPOSE_CONFIG}
fi

builder=$(docker buildx ls | grep "^multiple-platforms-builder" | awk '{print $1}')
if [[ -z "${builder}" ]]; then
  docker buildx create --name multiple-platforms-builder --driver docker-container --bootstrap
fi

export DOCKER_BUILDKIT=1
export COMPOSE_BAKE=true
docker compose -f ${COMPOSE_CONFIG} -p ${PROJECT_NAME} build \
  --builder multiple-platforms-builder \
  --push cu128 cpu

rm -f ${COMPOSE_CONFIG}
rm -rf "${DOCKER_CONFIG_HOME}/data"
