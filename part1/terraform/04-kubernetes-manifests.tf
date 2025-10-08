data "google_client_config" "default" {}

provider "kubernetes" {
  alias                  = "gke"
  host                   = "https://${google_container_cluster.primary.endpoint}"
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "gke-gcloud-auth-plugin"
    args        = ["--token-source=gcloud"]
  }
}

provider "kubectl" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  token                  = data.google_client_config.default.access_token
}

locals {
  is_gke_version_ge_1_34 = var.gke_version != null ? version(var.gke_version) >= version("1.34.0-gke.1626000") : true
  manifests_as_list      = local.is_gke_version_ge_1_34 ? [yamldecode(data.http.inference_objective_crd.response_body)] : yamldecode(data.http.inference_pool_crd.response_body)
  cleaned_manifests      = [for m in local.manifests_as_list : { for k, v in m : k => v if k != "status" }]
}

data "http" "inference_objective_crd" {
  url = "https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml"
}

data "http" "inference_pool_crd" {
  url = "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.0/manifests.yaml"
}

resource "kubectl_manifest" "inference_crds" {
  # We re-encode the cleaned manifest objects back into a single multi-document YAML string
  yaml_body = join("\n---\n", [
    for m in local.cleaned_manifests : yamlencode(m)
  ])
  depends_on = [google_container_cluster.primary]
}





resource "kubectl_manifest" "gvnic_param_set" {
  depends_on = [google_container_cluster.primary]
  # Use yamlencode to convert the map to a YAML string
  yaml_body = yamlencode({
    apiVersion = "networking.gke.io/v1"
    kind       = "GKENetworkParamSet"
    metadata = {
      name = "gvnic-1"
    }
    spec = {
      vpc        = google_compute_network.gvnic_vpc.name
      vpcSubnet  = google_compute_subnetwork.gvnic_sub.name
      deviceMode = "NetDevice"
    }
  })
}

resource "kubectl_manifest" "gvnic_network" {
  # The dependency now points to the new kubectl_manifest resource
  depends_on = [kubectl_manifest.gvnic_param_set]
  yaml_body = yamlencode({
    apiVersion = "networking.gke.io/v1"
    kind       = "Network"
    metadata = {
      name = "gvnic-1"
    }
    spec = {
      type = "Device"
      parametersRef = {
        group = "networking.gke.io"
        kind  = "GKENetworkParamSet"
        name  = "gvnic-1"
      }
    }
  })
}

resource "kubectl_manifest" "rdma_param_set" {
  depends_on = [google_container_cluster.primary]
  for_each   = { for i, sub in google_compute_subnetwork.rdma_sub : i => sub }
  yaml_body = yamlencode({
    apiVersion = "networking.gke.io/v1"
    kind       = "GKENetworkParamSet"
    metadata = {
      name = "rdma-${each.key}"
    }
    spec = {
      vpc        = google_compute_network.rdma_vpc.name
      vpcSubnet  = each.value.name
      deviceMode = "RDMA"
    }
  })
}

resource "kubectl_manifest" "rdma_network" {
  for_each = { for i, sub in google_compute_subnetwork.rdma_sub : i => sub }
  # Update the depends_on to correctly reference the looped kubectl_manifest
  depends_on = [kubectl_manifest.rdma_param_set]
  yaml_body = yamlencode({
    apiVersion = "networking.gke.io/v1"
    kind       = "Network"
    metadata = {
      name = "rdma-${each.key}"
    }
    spec = {
      type = "Device"
      parametersRef = {
        group = "networking.gke.io"
        kind  = "GKENetworkParamSet"
        name  = "rdma-${each.key}"
      }
    }
  })
}



data "http" "nccl_rdma_installer" {
  url = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/gpudirect-rdma/nccl-rdma-installer.yaml"
}

resource "kubectl_manifest" "nccl_rdma_installer" {
  # Pass the raw YAML string from the data source directly
  yaml_body  = data.http.nccl_rdma_installer.response_body
  depends_on = [google_container_cluster.primary]
}