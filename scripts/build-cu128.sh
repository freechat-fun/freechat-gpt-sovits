#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

mkdir -p "${DOCKER_CONFIG_HOME}/data/nls"
cp "${PROJECT_PATH}/src/"* "${DOCKER_CONFIG_HOME}/data/"
cp -r "${PROJECT_PATH}/src/nls/"* "${DOCKER_CONFIG_HOME}/data/nls/"

# Display configuration if VERBOSE is set
if [[ "${VERBOSE}" == "1" ]]; then
  echo "[BUILD CONFIG for CU128]"
  echo "Context: ${DOCKER_CONFIG_HOME}"
  echo "Dockerfile: Dockerfile_cu128"
  echo "Image: ${HELM_image_repository}:${HELM_version}-cu128/latest-cu128/latest"
fi

docker build "${DOCKER_CONFIG_HOME}" \
  --platform linux/amd64 \
  -f "${DOCKER_CONFIG_HOME}/Dockerfile_cu128" \
  -t "${HELM_image_repository}:${HELM_version}-cu128" \
  -t "${HELM_image_repository}:latest-cu128" \
  -t "${HELM_image_repository}:latest"

rm -rf "${DOCKER_CONFIG_HOME}/data"
