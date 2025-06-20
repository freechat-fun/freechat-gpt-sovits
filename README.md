# Scripts for GPT-SoVITS TTS

1. Put the Kubernetes configuration file in the `configs/helm/` directory, named `kube-private.conf`.
2. Place the Helm configuration file in the same directory, named `values-private.yaml`. Make sure to reference the default `values.yaml` and customize the variables as needed.
3. If needed, run `install-in.sh` to deploy `ingress-nginx` on the Kubernetes cluster.
4. If needed, run `install-cm.sh` to deploy `cert-manager` on the Kubernetes cluster, which automatically issues certificates for domains specified in `ingress.hosts`.
5. Run `install.sh` script to install GPT-SoVITS and its dependencies.
6. Run `restart.sh` to restart the GPT-SoVITS service.
7. If you modified any Helm configuration files, use `upgrade.sh` to update the corresponding Kubernetes resources.
8. To remove specific resources, run the `uninstall*.sh` script corresponding to the resource you want to uninstall.
