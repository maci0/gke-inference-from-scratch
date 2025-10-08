# Kubernetes and Kubectl providers configuration
# These connect to the GKE cluster created in part1

data "google_client_config" "default" {}

# Reference the existing GKE cluster from part1
data "google_container_cluster" "primary" {
  name     = "${var.name_prefix}-cluster"
  location = var.region
}

provider "kubernetes" {
  host                   = "https://${data.google_container_cluster.primary.endpoint}"
  cluster_ca_certificate = base64decode(data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "gke-gcloud-auth-plugin"
    args        = ["--token-source=gcloud"]
  }
}

provider "kubectl" {
  host                   = "https://${data.google_container_cluster.primary.endpoint}"
  cluster_ca_certificate = base64decode(data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  token                  = data.google_client_config.default.access_token
}
