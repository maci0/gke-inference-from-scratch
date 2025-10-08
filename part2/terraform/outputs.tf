# Outputs for Part 2: Inference Frameworks

output "leaderworkerset_installed" {
  description = "Confirmation that LeaderWorkerSet has been installed"
  value       = "LeaderWorkerSet v0.7.0 installed successfully"
  depends_on  = [kubectl_manifest.lws]
}

output "cluster_name" {
  description = "Name of the GKE cluster (from Part 1)"
  value       = data.google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "Endpoint of the GKE cluster (from Part 1)"
  value       = data.google_container_cluster.primary.endpoint
  sensitive   = true
}

output "verification_commands" {
  description = "Commands to verify the installation"
  value = {
    check_crd        = "kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io"
    check_controller = "kubectl get pods -n lws-system"
    wait_for_ready   = "kubectl wait --for=condition=Available deployment/lws-controller-manager -n lws-system --timeout=300s"
  }
}
