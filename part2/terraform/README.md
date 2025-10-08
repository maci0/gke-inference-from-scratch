# Part 2: Inference Frameworks - Terraform Configuration

This Terraform configuration installs the inference framework dependencies on the GKE cluster created in Part 1.

## What This Deploys

- **LeaderWorkerSet (LWS)**: Kubernetes controller for managing multi-node inference workloads

## Prerequisites

1. Complete Part 1 infrastructure setup (GKE cluster must exist)
2. Terraform >= 1.0
3. `gke-gcloud-auth-plugin` installed
4. Authenticated to GCP with appropriate permissions

## File Structure

```
part2/terraform/
├── main.tf              -> symlink to ../../part1/terraform/main.tf
├── variables.tf         -> symlink to ../../part1/terraform/variables.tf
├── providers.tf         # Kubernetes/kubectl provider configuration
├── leaderworkerset.tf   # LeaderWorkerSet installation
└── README.md           # This file
```

## Usage

### Initialize Terraform

```bash
cd part2/terraform
terraform init
```

### Review the Plan

```bash
terraform plan
```

### Apply the Configuration

```bash
terraform apply
```

### Verify Installation

```bash
# Check that LeaderWorkerSet CRD is installed
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io

# Check LeaderWorkerSet controller is running
kubectl get pods -n lws-system

# Wait for controller to be ready
kubectl wait --for=condition=Available deployment/lws-controller-manager -n lws-system --timeout=300s
```

## Configuration Variables

This module uses the same variables as Part 1 via symlink:

- `project_id`: GCP project ID
- `region`: GCP region (must match Part 1)
- `name_prefix`: Resource name prefix (must match Part 1)

## Dependencies

This configuration assumes:
1. The GKE cluster from Part 1 exists
2. You have `kubectl` access to the cluster
3. The cluster name follows the pattern: `${name_prefix}-cluster`

## Cleanup

To remove the LeaderWorkerSet installation:

```bash
terraform destroy
```

**Note**: This will only remove the LeaderWorkerSet components. The GKE cluster and other Part 1 infrastructure will remain.

## Next Steps

After applying this configuration, you can:
1. Deploy single-GPU vLLM workloads (Section 1 of Part 2)
2. Deploy multi-GPU single-node workloads (Section 2 of Part 2)
3. Deploy multi-node workloads with LeaderWorkerSet (Section 3 of Part 2)

## Troubleshooting

### Provider Authentication Issues

If you see authentication errors:

```bash
# Re-authenticate with GCP
gcloud auth application-default login

# Ensure gke-gcloud-auth-plugin is installed
gcloud components install gke-gcloud-auth-plugin
```

### Cluster Not Found

Ensure the cluster from Part 1 exists:

```bash
gcloud container clusters list --region=<your-region>
```

### LeaderWorkerSet Installation Failed

Check the logs:

```bash
kubectl logs -n lws-system deployment/lws-controller-manager
```
