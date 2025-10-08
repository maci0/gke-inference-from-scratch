# GKE Infrastructure for Distributed LLM Inferencing

This repository contains Terraform code to provision a Google Kubernetes Engine (GKE) cluster optimized for distributed inferencing workloads. It sets up a GKE cluster with high-performance networking (gVNIC and RDMA) and a dedicated GPU node pool.

## Features

-   **High-Performance Networking:** Creates two separate VPC networks:
    -   One for general purpose GPU traffic using [Google Virtual NIC (gVNIC)](https://cloud.google.com/compute/docs/instances/create-start-instance/about-gvnic).
    -   One for high-throughput, low-latency [RDMA over Converged Ethernet (RoCE)](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet) for demanding workloads.
-   **Optimized GKE Cluster:**
    -   VPC-native networking with Dataplane V2 for enhanced security and performance.
    -   Multi-networking enabled to connect to both gVNIC and RDMA networks.
    -   Workload Identity for secure access to Google Cloud services.
    -   Essential addons pre-configured: HTTP Load Balancing, Horizontal Pod Autoscaling, GCE Persistent Disk CSI Driver, and GCS Fuse CSI Driver.
    -   Managed Prometheus for monitoring and Gateway API for advanced traffic management.
-   **Powerful GPU Node Pool:**
    -   A dedicated node pool with autoscaling for GPU-intensive workloads.
    -   Configurable machine type, GPU accelerator type, and count.
    -   gVNIC enabled for faster networking.

## Kubernetes Manifests

This project also applies several Kubernetes manifests to the GKE cluster.

-   **gVNIC:** A `GKENetworkParamSet` and a `Network` resource are created for the gVNIC network. This allows pods to use gVNIC for high-performance networking.
-   **RDMA:** For each RDMA subnetwork, a `GKENetworkParamSet` and a `Network` resource are created. This enables pods to leverage RDMA for ultra-low-latency communication, which is crucial for distributed training workloads.
-   **LWS:** The [LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws) An API for deploying a group of pods as a unit of replication.

## Prerequisites

1.  **Terraform:** Install Terraform CLI (version 1.0.0 or later).
2.  **Google Cloud SDK:** Install and configure the `gcloud` CLI with credentials to access your Google Cloud project.
3.  **Permissions:** Ensure that you have the required permissions for creating and managing the GKE cluster and associated service accounts:
    - Kubernetes Engine Admin (roles/container.admin)
    - Compute Admin (roles/compute.admin)
    - Storage Admin (roles/storage.admin)
    - Project IAM Admin (roles/resourcemanager.projectIamAdmin)
    - Service Account Admin (roles/iam.serviceAccountAdmin)
    - Service Account User (roles/iam.serviceAccountUser)
    - Service Usage Consumer (roles/serviceusage.serviceUsageConsumer)
4.  **Application Default Credentials:** Authenticate with Google Cloud to set up Application Default Credentials:
    ```bash
    gcloud auth application-default login
    ```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/maci0/gke-inference-from-scratch
    cd gke-inference-from-scratch/part1/terraform
    ```

2.  **Configure variables:**
    Create a `terraform.tfvars` file and provide values for the required variables.

    ```hcl
    project_id              = "your-gcp-project-id"
    region                  = "your-gcp-region"
    zone                    = "your-gcp-zone"
    name_prefix             = "llm-infra"
    ```

3.  **Initialize Terraform:**
    ```bash
    terraform init
    ```

4.  **Review the plan:**
    ```bash
    terraform plan
    ```

5.  **Apply the configuration:**
    ```bash
    terraform apply
    ```

## NCCL Test

This repository includes a script to run an NCCL test to verify the high-performance networking setup. The test uses a `Job` to run an `all_gather` performance test across multiple nodes.

### Prerequisites

-   `kubectl` configured to communicate with your GKE cluster.
  
### Running the Test

1.  Navigate to the `tests` directory:
    ```bash
    cd tests
    ```

2.  Run the `run-nccl-test.sh` script:
    ```bash
    ./run-nccl-test.sh
    ```

    You can also override the default values for the nodepool name, number of nodes, and GPUs per node:

    ```bash
    ./run-nccl-test.sh --nodepool-name <your-nodepool-name> --num-nodes <number-of-nodes> --gpus-per-node <gpus-per-node>
    ```

3.  The script will substitute the variables in the `nccl-test.yaml.tmpl` manifest, apply it to the cluster, and provide instructions on how to check the status and logs of the job.

## Using a Specific Reservation

To use a specific reservation for the GPU node pool, set the `reservation_name` variable in your `terraform.tfvars` file to the name of your desired reservation. If this variable is not set or is left empty, a new reservation will be created with the name specified in the `reservation_name` variable.

You can also create a new reservation by setting the `create_reservation` variable to `true`. The following variables are used to configure the reservation:
- `reservation_name`: The name of the reservation to create.
- `zone`: The zone for the reservation.
- `reservation_instance_count`: The number of instances to reserve.
- `gpu_machine_type`: The machine type to reserve.
- `gpu_accelerator_count`: The number of GPUs per node.
- `gpu_accelerator_type`: The type of GPU accelerator.

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|:----:|:-----:|:---:|
| `project_id` | The Google Cloud project ID. | `string` | - | yes |
| `region` | The Google Cloud region for the resources. | `string` | - | yes |
| `zone` | The zone for the GPU node pool. | `string` | - | yes |
| `name_prefix` | A common prefix for all created resources. | `string` | - | yes |
| `gpu_machine_type` | The machine type for the GPU node pool. | `string` | `a3-ultragpu-8g` | no |
| `gpu_accelerator_type` | The type of GPU accelerator. | `string` | `nvidia-h200-141gb` | no |
| `gpu_accelerator_count` | The number of GPUs per node. | `number` | `8` | no |
| `autoscaling_min_nodes` | The minimum number of nodes in the GPU node pool. | `number` | `0` | no |
| `autoscaling_max_nodes` | The maximum number of nodes in the GPU node pool. | `number` | `4` | no |
| `reservation_name` | The name of the specific reservation to use for the GPU node pool. If left empty, no reservation is used. | `string` | `""` | no |
| `gke_release_channel` | The GKE release channel to use for the cluster. | `string` | `STABLE` | no |
| `gke_version` | The GKE version to use for the cluster. If not specified, the default version for the release channel will be used. | `string` | `null` | no |
| `gpu_driver_version` | The GPU driver version to use for the GPU node pool. | `string` | `LATEST` | no |
| `create_reservation` | Flag to determine if a reservation should be created. | `bool` | `false` | no |
| `reservation_instance_count` | The number of instances to reserve. | `number` | `1` | no |

## Outputs

| Name | Description | Sensitive |
|------|-------------|:---:|
| `gke_cluster_name` | The name of the GKE cluster. | no |
| `gke_cluster_endpoint` | The endpoint of the GKE cluster. | yes |
| `gpu_nodepool_name` | The name of the GPU node pool. | no |
