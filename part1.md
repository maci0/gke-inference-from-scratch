# Building a Production Ready LLM Inferencing Platform on GKE from Scratch – Part 1: The Foundation

## Introduction

In this multi-part series of blog posts, we will explore how to build a production-ready LLM inference platform on Google Kubernetes Engine (GKE) from the ground up. Whether you're scaling from prototype to production or building new infrastructure, this guide will walk you through every component needed for a robust, performant inference system.

While there are plenty of tools available ( cluster toolkit, xpk, etc. ) to abstract this away. It is beneficial to have an understanding of every layer of the stack.

Throughout this series we will build those layers bottom to top. We will look at how to build them using regular `gcloud` command and how to build the automation using `terraform`.

### Series Overview

This series covers:

1. **Base Infrastructure** - GKE setup with GPU support and RDMA networking
2. Inference Frameworks - vLLM, TGI, TensorRT-LLM comparison and deployment
3. Inference Gateway - Intelligent routing
4. Distributed Inferencing - Advanced patterns with llm-d
   - KV Cache sharing across instances
   - Disaggregated Prefill/Decode architecture
5. Storage Options - Model storage, caching, and versioning strategies
6. Consumption Models - DWS, Flex-Start, Ondemand, Spot
7. Accelerating Pod Startup - Techniques to minimize cold start time
   - Secondary boot disk
   - Pod Snapshot / Restore
8. Testing and Benchmarking - Performance validation and optimization

---

## Part 1: The Foundation

Every platform needs a solid foundation. For our LLM inference platform, that foundation is GKE with specialized networking and accelerator support.

### What We'll Build

In this first part, we'll set up:

- A GKE cluster optimized for ML workloads
- Additional networks configured for RDMA (Remote Direct Memory Access)
- Node pools with GPU accelerators
- RDMA drivers for high-performance inter-node communication

### Prerequisites

Before you begin, ensure you have:

- A Google Cloud Project with billing enabled
- `gcloud` CLI installed and authenticated
- `kubectl` installed
- Terraform installed
- Appropriate IAM permissions to create GKE clusters and networks

### Architecture Overview

<!-- TODO: Add architecture diagram -->

Our infrastructure consists of:

- **Control Plane**: Managed by GKE
- **Compute Node Pools**: GPU-enabled nodes (H200)
- **gVNIC Network**: Increase network traffic speed for GPU nodes
- **RDMA Network**: Secondary network for low-latency GPU-to-GPU communication
- **Storage**: GCS for model weights, local NVMe for caching

---

## Step 1: Setting Up the GKE Cluster

### 1.1 Create the Base Cluster

First, let's create a base GKE cluster with the appropriate configuration for ML workloads:

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export CLUSTER_NAME="llm-inference-cluster"

gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --machine-type=n2-standard-8 \
  --num-nodes=3 \
  --enable-ip-alias \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --network="default" \
  --subnetwork="default"
```

### 1.2 Configure kubectl Access

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID}
```

Verify the cluster is accessible:

```bash
kubectl get nodes
```

### Terraform Automation

Below an example for how this would look like with `terraform`.

```hcl
resource "google_container_cluster" "primary" {
  name     = var.name_prefix
  location = var.region

  deletion_protection = false

  # Use the default compute network for the control plane. Enables ip aliasing
  networking_mode = "VPC_NATIVE"

  # Use dataplane v2
  datapath_provider = "ADVANCED_DATAPATH"

  # Enable Multi-Networking
  enable_multi_networking = true

  # Use default node pool for cluster addons
  remove_default_node_pool = false
  initial_node_count       = 1

  # --workload-pool "${PROJECT_ID}.svc.id.goog" \
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  node_config {
    machine_type = "e2-standard-8"
  }

  # Addons from your gcloud command
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  # Enable Managed Prometheus
  monitoring_config {
    managed_prometheus {
      enabled = true
    }
  }

  # Enable Gateway API
  gateway_api_config {
    channel = "CHANNEL_STANDARD"
  }

  # Enable Shielded Nodes
  enable_shielded_nodes = true

  release_channel {
    channel = var.gke_release_channel
  }

  min_master_version = var.gke_version
}
```

---

## Step 2: Additional Networks for RDMA

Before we dive into setup, let’s briefly review what RDMA is.
Remote Direct Memory Access (RDMA) allows one computer (or GPU) to read and write directly into another’s memory without involving the CPU. This drastically reduces latency and CPU overhead, which is critical for large-scale AI inference and training workloads.

The A3 Ultra VMs with NVIDIA H200 GPUs support GPUDirect RDMA, an advanced form of RDMA that connects GPUs across nodes through RDMA over Converged Ethernet (RoCE). These VMs use Google’s Titanium ML network adapter based on ConnectX-7 NICs, delivering up to 3.2 Tbps of non-blocking GPU-to-GPU bandwidth. This setup makes them ideal for high-performance distributed inference or training.

![GPUDirect RDMA Overview](https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/GPUDirect/gpudirect-rdma.png "GPUDirect RDMA Overview")
https://developer.nvidia.com/gpudirect

To make the most of this network fabric, Google Cloud provides several GPUDirect variants optimized for different VM types:

| Feature                   | Machine Type      | GPU Model          | Description / Benefit                                                                     | Max Transfer Speed                   |
| ------------------------- | ----------------- | ------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| **GPUDirect-TCPX**        | A3 High           | NVIDIA H100        | Reduces packet-transfer overhead and improves throughput at scale.                        | ~1,000 Gbps                          |
| **GPUDirect-TCPXO**       | A3 Mega           | NVIDIA H100 Mega   | Optimized for GPU-to-VM communication with higher throughput than TCPX.                   | ~1,800 Gbps                          |
| **GPUDirect-RDMA (RoCE)** | A3 Ultra / A4     | NVIDIA H200 / B200 | Direct GPU memory access across nodes for maximum throughput.                             | ~3,200 Gbps                          |
| **gVNIC**                 | All A3 / A4 types | All GPUs           | Required for TCPX/TCPXO; provides header splitting, flow steering, and buffer management. | Up to ~400 Gbps (general networking) |

GPUDirect RDMA / Interfaces – The GPU and NIC communicate over PCIe using peer-to-peer transfers, so data moves directly from GPU memory to the network adapter without touching system RAM.

Network Stack (RoCE) – The underlying network fabric uses RDMA over Converged Ethernet, providing low-latency memory operations between GPUs across nodes.

gVNIC – Google’s virtual NIC that enables GPUDirect capabilities like header splitting and flow steering. It’s required for TCPX and TCPXO stacks but not for RDMA.

To enable full RDMA functionality, you’ll typically configure two networks:

An RDMA VPC dedicated to RoCE traffic.

A gVNIC VPC for TCPX/TCPXO or general networking.


### 2.1 Create RDMA-capable Network

```bash
# Create a secondary network for RDMA traffic
gcloud compute networks create rdma-network \
  --project=${PROJECT_ID} \
  --subnet-mode=custom \
  --mtu=8896

# Create a subnet with appropriate IP range
gcloud compute networks subnets create rdma-subnet \
  --project=${PROJECT_ID} \
  --network=rdma-network \
  --region=${REGION} \
  --range=10.128.0.0/20
```


### 2.2 Create Subnets

Why So Many Subnets?

https://cloud.google.com/vpc/docs/rdma-network-profiles

If you’ve looked at the setup script and wondered why it creates eight subnets, here’s the reason.

Each A3 Ultra VM has eight dedicated RDMA NICs (ConnectX-7 adapters), and Google Cloud requires that each RDMA NIC lives in its own subnet. This design isolates traffic, prevents IP conflicts, and allows each interface to reach its full bandwidth without interference.

In practice, that means your RDMA VPC needs eight separate /24 subnets—one for each NIC. The snippet below loops through and creates them automatically, using IP ranges 192.168.1.0/24 through 192.168.8.0/24. The offset ensures there’s no overlap with your gVNIC network, which handles general or TCPX/TCPXO traffic.

This might look excessive at first, but it’s what enables the full 3.2 Tbps bandwidth that A3 Ultra nodes can deliver through RoCE. Each subnet gives an RDMA NIC its own isolated lane on the high-performance network fabric, ensuring predictable and ultra-low-latency GPU-to-GPU communication.

### 2.2 Configure Network Policies

<!-- TODO: Add network policy manifests -->

```bash
# Apply network policies for RDMA traffic
kubectl apply -f network-policies.yaml
```

### 2.2 create gVNIC Network

gVNIC: Enable GPUDirect capabilities such as packet header splitting, flow steering, and buffer management. gVNIC is required to use GPUDirect-TCPX or GPUDirect-TCPXO. For details about gVNIC, see Increase network traffic speed for GPU nodes.



---

## Step 3: Create Node Pools with Accelerators

Now we'll create specialized node pools with GPUs attached.

### 3.1 GPU Node Pool (H200)

```bash
gcloud container node-pools create gpu-pool-a100 \
  --cluster=${CLUSTER_NAME} \
  --region=${REGION} \
  --machine-type=a3-ultragpu-8g \
  --accelerator=type=nvidia-h200-141gb,count=8 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=8 \
  --disk-type=pd-ssd \
  --disk-size=500 \
  --no-enable-autorepair --location-policy=ANY \
  --node-labels=workload=llm-inference,gpu=h200 \
    --additional-node-network network=${GVNIC_NETWORK_PREFIX}-net,subnetwork=${GVNIC_NETWORK_PREFIX}-sub \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-0 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-1 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-2 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-3 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-4 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-5 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-6 \
  --additional-node-network network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-7
  
```

### 3.2 Configure GPU Driver Installation

GKE automatically installs GPU drivers, but we can verify and customize:

```bash
# Verify GPU nodes are ready
kubectl get nodes -l gpu=a100

# Check GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable'
```

---

## Step 4: Install RDMA Drivers

### 4.1 Deploy RDMA Device Plugin

```bash
# Apply RDMA device plugin DaemonSet
```

### 4.2 Verify RDMA Installation

```bash
# Check that RDMA devices are detected
```

### 4.3 Test RDMA Connectivity

<!-- TODO: Add RDMA test manifests -->

```bash
# Deploy test pods to verify RDMA connectivity
kubectl apply -f rdma-test-pods.yaml

# Run perftest between pods
kubectl exec -it rdma-test-1 -- ib_write_bw
```

---

## Step 5: Install Supporting Infrastructure

---

## Validation and Testing

### Verify Cluster Setup

Run these checks to ensure everything is configured correctly:

```bash
# 1. Check node pools
kubectl get nodes --show-labels

# 2. Verify GPU resources
kubectl describe nodes -l gpu=a100 | grep -A 5 "Allocatable"

# 3. Check RDMA devices
kubectl get nodes -o json | jq '.items[].status.allocatable'

# 4. Test GPU with a simple workload
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.0.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

### Performance Baseline

<!-- TODO: Add baseline benchmark scripts -->

```bash
# Run initial GPU benchmarks
kubectl apply -f benchmark-gpu.yaml

# Check results
kubectl logs -f gpu-benchmark-pod
```

---

## Cost Optimization Tips

1. **Use Preemptible/Spot Instances** for non-critical workloads
2. **Enable cluster autoscaling** to scale down during low usage
3. **Set appropriate resource requests/limits** to maximize node utilization
4. **Use committed use discounts** for predictable workloads

---

## Troubleshooting

### Common Issues

**Node pool creation failures:**

```bash
# Check quota limits
gcloud compute project-info describe --project=${PROJECT_ID}

# View detailed error
gcloud container operations describe <operation-id> --region=${REGION}
```

---

## Next Steps

With our foundation in place, we're ready to deploy inference frameworks. In **Part 2: Inference Frameworks**, we'll:

- Deploy and compare vLLM, TGI, and TensorRT-LLM
- Run initial performance benchmarks

---

## Resources

- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GKE RDMA Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)
- [GKE gVNIC Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/using-gvnic)
  
---

## Feedback and Contributions

Questions or suggestions? [Open an issue](#) or reach out on [Twitter/LinkedIn](#).

---

**Coming Soon: Part 2 - Inference Frameworks**
