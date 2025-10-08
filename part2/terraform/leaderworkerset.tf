# LeaderWorkerSet Installation
# This installs the LeaderWorkerSet CRDs and controller needed for multi-node inference workloads

data "http" "lws_manifests" {
  url = "https://github.com/kubernetes-sigs/lws/releases/download/v0.7.0/manifests.yaml"
}

resource "kubectl_manifest" "lws" {
  yaml_body = data.http.lws_manifests.response_body
  # Note: This assumes the GKE cluster already exists (created in part1)
  # You'll need to configure the kubernetes/kubectl providers to connect to your cluster
}