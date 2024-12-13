#!/usr/bin/env bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

check_helm

helm install --kubeconfig ${KUBE_CONFIG} --namespace ${NAMESPACE} --create-namespace -f ${values_yaml} \
  --set deployment.enabled=false \
  --set persistence.enabled=true \
  ${ARGS[*]} \
  ${PROJECT_NAME}-pvc ${HELM_CONFIG_HOME}
