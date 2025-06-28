#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_docker

mkdir -p "${DOCKER_CONFIG_HOME}/data/nls"
cp "${PROJECT_PATH}/src/"*.* "${DOCKER_CONFIG_HOME}/data/"
cp -r "${PROJECT_PATH}/src/nls/"* "${DOCKER_CONFIG_HOME}/data/nls/"

# Display configuration if VERBOSE is set
if [[ "${VERBOSE}" == "1" ]]; then
  echo "[BUILD CONFIG for CPU]"
  echo "Context: ${DOCKER_CONFIG_HOME}"
  echo "Dockerfile: Dockerfile_cpu"
  echo "Image: ${HELM_image_repository}:latest-cpu"
fi

# Build the Docker image using docker build
docker build "${DOCKER_CONFIG_HOME}" \
  -f "${DOCKER_CONFIG_HOME}/Dockerfile_cpu" \
  -t "${HELM_image_repository}:${HELM_version}-cpu" \
  -t "${HELM_image_repository}:latest-cpu"

rm -rf "${DOCKER_CONFIG_HOME}/data"
