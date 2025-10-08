output "gke_cluster_name" {
  description = "The name of the GKE cluster."
  value       = google_container_cluster.primary.name
}

output "gke_cluster_region" {
  description = "The region of the GKE cluster."
  value       = google_container_cluster.primary.location
}

output "gke_cluster_endpoint" {
  description = "The endpoint of the GKE cluster."
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "gpu_nodepool_name" {
  description = "The name of the GPU node pool."
  value       = google_container_node_pool.gpu_nodes.name
}

output "gpu_nodepool_max_nodes" {
  description = "The maximum number of nodes in the GPU node pool."
  value       = var.autoscaling_max_nodes
}

output "gpus_per_node" {
  description = "The number of GPUs per node in the GPU node pool."
  value       = var.gpu_accelerator_count
}
