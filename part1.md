# Building a Production Ready LLM Inferencing Platform on GKE from Scratch – Part 1: The Foundation

## Introduction

You've built an LLM application that works great in development. Now you need to serve it at scale. Thousands of concurrent requests with sub-second latency. Your prototype ran on a single GPU, but your production model won't fit on one machine. Time for some real infrastructure work.

This series walks through building a production LLM inference platform on GKE from scratch. We'll cover everything you need: networking, GPU clusters, inference frameworks, routing, and optimization.

Tools like `cluster toolkit` and `xpk` can automate much of this, but understanding what's happening under the hood matters. When your cluster starts acting weird at 3 AM, you'll actually know how to debug it.

We'll start with `gcloud` commands to see what's happening at each step, then automate it with Terraform for production use. With each step we will also explore and explain some of the underlying technologies that make everything work.

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

Every platform needs a solid foundation. For our LLM inference platform, that foundation is **Google Kubernetes Engine (GKE)** with specialized networking and accelerator support.
The first part of this series will focus on exactly that. The foundation of our platform.

### What We'll Build

In this first part, we'll set up:

- A GKE cluster optimized for inference workloads
- Additional networks configured for RDMA (Remote Direct Memory Access)
- Node pools with H200 GPU accelerators
- RDMA binaries for high-performance inter-node communication
- `LeaderWorkerSet` for managing distributed inference workloads

### Prerequisites

Before you begin, ensure you have:

- A Google Cloud Project with billing enabled
- `gcloud` CLI installed and authenticated
- `kubectl` installed
- `Terraform` installed
- Appropriate IAM permissions to create GKE clusters and networks
- Appropriate quota to use accelerators

**Important Limitations:**
- GPUs have [limited availability](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones) and require quota approval in specific zones
- Network profiles (like RoCE) are [zone-specific and not available in all regions](https://cloud.google.com/vpc/docs/rdma-network-profiles#availability)


### Architecture Overview

![High Level Architecture Diagram](part1/GKE%20AI%20Infrastructure.drawio.png)

**Key Components:**

- **Control Plane**: Managed by GKE
- **Compute Node Pools**: A3 Ultra nodes with H200 GPUs (8 per node)
- **gVNIC Network**: High-throughput general networking (400 Gbps)
- **RDMA Network**: 8 dedicated RDMA NICs per node for GPU-to-GPU communication (3.2 Tbps total)

### Approach

In this guide, we'll take a hands-on approach by first using `gcloud` commands and shell scripts to build and understand each component. This approach helps you see exactly what's happening at each step and understand the underlying infrastructure.

Once we've worked through the manual setup, we'll show how to automate the entire process using Terraform, making it repeatable and production-ready.

---

## Step 1: Set Environment Variables

First, let's set up our environment variables that we'll use throughout the setup:

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export ZONE="us-central1-b"
export NAME_PREFIX="llm-inference"
export CLUSTER_NAME="${NAME_PREFIX}-cluster"

# GPU configuration
export GPU_MACHINE_TYPE="a3-ultragpu-8g"
export GPU_ACCELERATOR_TYPE="nvidia-h200-141gb"
export GPU_ACCELERATOR_COUNT=8
export AUTOSCALING_MIN_NODES=2
export AUTOSCALING_MAX_NODES=4

# GKE configuration
export GKE_RELEASE_CHANNEL="stable"
export GKE_VERSION="1.33.4-gke.1134000"
```

---

## Step 2: Understanding RDMA and Network Requirements

**Why RDMA matters for distributed inference:**

Large language models (70B+ parameters) often don't fit on a single GPU. When tensor-parallel inference splits a model across multiple GPUs and nodes, inter-GPU communication becomes a bottleneck. Traditional TCP/IP networking routes data through the CPU and kernel: `GPU → System RAM → Kernel → Network → Remote Kernel → Remote RAM → GPU`. Each hop adds latency (50-100 microseconds) and CPU overhead.

Remote Direct Memory Access (RDMA) enables direct GPU-to-GPU communication without CPU involvement: `GPU → RDMA NIC → Network → Remote RDMA NIC → Remote GPU`. Latency drops to single-digit microseconds. For distributed inference serving concurrent users, this difference is significant: higher throughput, lower latency percentiles, and better GPU utilization.

The A3 Ultra VMs with NVIDIA H200 GPUs support **GPUDirect RDMA**, it connects GPUs across nodes through RDMA over Converged Ethernet (RoCE). **RoCE** is a network protocol that allows RDMA to work over standard Ethernet networks, combining the low latency and high throughput of RDMA with the flexibility and scalability of Ethernet. These VMs use `Google's Titanium ML` network adapter based on NVIDIA's `ConnectX-7 NICs`, delivering up to 3.2 Tbps of non-blocking GPU-to-GPU bandwidth. This setup makes them ideal for high-performance distributed inference or training.

![GPUDirect RDMA Overview](https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/GPUDirect/gpudirect-rdma.png "GPUDirect RDMA Overview")
Diagram Courtesy of [NVIDIA](https://developer.nvidia.com/gpudirect)

To make the most of this network fabric, Google Cloud provides several GPUDirect variants optimized for different VM types:

| Feature                   | Machine Type      | GPU Model          | Description / Benefit                                                                     | Max Transfer Speed                   |
| ------------------------- | ----------------- | ------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| **[GPUDirect-TCPX](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)**        | A3 High  | `nvidia-h100-80gb`        | Reduces packet-transfer overhead and improves throughput at scale.                        | ~1,000 Gbps                          |
| **[GPUDirect-TCPXO](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)**       | A3 Mega           | `nvidia-h100-mega-80gb`   | Optimized for GPU-to-VM communication with higher throughput than TCPX.                   | ~1,800 Gbps                          |
| **[GPUDirect-RDMA (RoCE)](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom)** | A3 Ultra / A4     | `nvidia-h200-141gb` / `nvidia-b200` | Direct GPU memory access across nodes for maximum throughput.                             | ~3,200 Gbps                          |
| **[gVNIC](https://cloud.google.com/kubernetes-engine/docs/how-to/using-gvnic)**                 | All A3 / A4 types | All GPUs           | Required for TCPX/TCPXO; provides header splitting, flow steering, and buffer management. | Up to ~400 Gbps (general networking) |


That was a lot of information, but to recap:

**GPUDirect RDMA Interfaces** – The GPU and NIC communicate over PCIe using peer-to-peer transfers, so data moves directly from GPU memory to the network adapter without touching system RAM.

**Network Stack (RoCE)** – The underlying network fabric uses RDMA over Converged Ethernet, providing low-latency memory operations between GPUs across nodes.

**gVNIC** – Google's virtual NIC that provides header splitting, flow steering, and buffer management. It's required for TCPX and TCPXO stacks. A3 Ultra VMs include both 8 RDMA NICs (for GPU-to-GPU communication) and gVNIC interfaces (for general networking, up to 400 Gbps).

**When do we need RDMA for inferencing?**
* Smaller models running on a single node actually don't require RDMA
* Large models requiring distributed inference across multiple nodes need RDMA for efficient inter-node communication
* Even for smaller models, RDMA can significantly boost inference performance by enabling KV cache sharing across nodes (we'll explore this in a later post)
* While RDMA is critical for distributed training workloads (gradient synchronization, all-reduce operations), our focus in this series is on inference use cases

---

With a solid understanding of RDMA and why it matters, let's start building our network infrastructure.

## Step 3: Create gVNIC Network

To enable full RDMA functionality, we need to configure two separate networks and appropriate subnets:

1. **gVNIC VPC** - For general networking and GPU traffic
2. **RDMA VPC** - Dedicated to RoCE traffic with 8 subnets 

### 3.1 Create gVNIC VPC and Subnet

```bash
# Create gVNIC network for general GPU traffic
gcloud compute networks create ${NAME_PREFIX}-gvnic-net \
  --project=${PROJECT_ID} \
  --subnet-mode=custom

# Create gVNIC subnet
gcloud compute networks subnets create ${NAME_PREFIX}-gvnic-sub-0 \
  --project=${PROJECT_ID} \
  --network=${NAME_PREFIX}-gvnic-net \
  --region=${REGION} \
  --range=192.168.0.0/24
```

This creates a custom VPC network with a single subnet for gVNIC traffic. The `--subnet-mode=custom` allows us to manually define subnets rather than having one automatically created per region.

### 3.2 Create gVNIC Firewall Rules

```bash
# Allow all internal traffic within gVNIC network
gcloud compute firewall-rules create ${NAME_PREFIX}-gvnic-internal \
  --project=${PROJECT_ID} \
  --network=${NAME_PREFIX}-gvnic-net \
  --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=192.168.0.0/16 \
  --description="Allow all internal traffic within gVNIC network"
```

This firewall rule allows all TCP, UDP, and ICMP traffic within the `192.168.0.0/16` range, enabling pods on the gVNIC network to communicate freely with each other.

---

With our gVNIC network in place for general traffic, let's configure the specialized RDMA network for `GPU-to-GPU` communication.

## Step 4: Create RDMA Network with 8 Subnets

### 4.1 Why 8 Subnets?

Each A3 Ultra VM has **eight dedicated RDMA NICs** (Titanium ML adapters), and Google Cloud requires that each RDMA NIC lives in its own subnet. This design:

- Isolates traffic per NIC
- Prevents IP conflicts
- Allows each interface to reach its full bandwidth without interference

This enables the full **3.2 Tbps bandwidth** that A3 Ultra nodes can deliver through RoCE.

**Reference**: [RDMA Network Profiles Documentation](https://cloud.google.com/vpc/docs/rdma-network-profiles)

### 4.2 Create RDMA VPC with RoCE Profile

```bash
# Create RDMA network with RoCE profile
gcloud beta compute networks create ${NAME_PREFIX}-rdma-net \
  --project=${PROJECT_ID} \
  --network-profile=${ZONE}-vpc-roce \
  --subnet-mode=custom \
  --mtu=8896
```

**Note**: MTU 8896 is recommended for best RDMA performance on Google Cloud. **Maximum Transmission Unit (MTU)** defines the largest packet size that can be transmitted over a network. The larger MTU (8896 bytes vs standard 1500 bytes) reduces the number of packets needed for data transfer, decreasing overhead and improving throughput for GPU-to-GPU communication.

### 4.3 Create RDMA Subnets

```bash
# Create 8 subnets for the 8 RDMA NICs
for N in $(seq 0 7); do
  gcloud compute networks subnets create ${NAME_PREFIX}-rdma-sub-$N \
    --project=${PROJECT_ID} \
    --network=${NAME_PREFIX}-rdma-net \
    --region=${REGION} \
    --range=192.168.$((N+1)).0/24
done
```

This creates subnets with IP ranges:

- `192.168.1.0/24` through `192.168.8.0/24` (numbered as rdma-sub-0 through rdma-sub-7)

### 4.4 Create RDMA Network Firewall Rules

```bash
# Allow all internal traffic across RDMA subnets
gcloud compute firewall-rules create ${NAME_PREFIX}-rdma-internal \
  --project=${PROJECT_ID} \
  --network=${NAME_PREFIX}-rdma-net \
  --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=192.168.0.0/16 \
  --description="Allow all internal traffic within RDMA network for GPU-to-GPU communication"
```

---

Our networks are ready. Now let's create the GKE cluster that will orchestrate our inference workloads.

## Step 5: Create GKE Cluster

Now that our networks are in place, we can create the GKE cluster. For the most part we use default parameters for the control plane. Flags that deviate from the defaults will be explained below.

### 5.1 Create the Base Cluster

```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --release-channel=${GKE_RELEASE_CHANNEL} \
  --cluster-version=$GKE_VERSION \
  --machine-type=e2-standard-8 \
  --num-nodes=1 \
  --enable-ip-alias \
  --enable-dataplane-v2 \
  --enable-multi-networking \
  --network="default" \
  --subnetwork="default" \
  --gateway-api=standard
```

**Key flags explained**:

- `--enable-dataplane-v2`: [Enhanced networking and security features](https://cloud.google.com/kubernetes-engine/docs/concepts/dataplane-v2)
- `--enable-multi-networking`: [Required for attaching multiple networks to pods](https://cloud.google.com/kubernetes-engine/docs/how-to/setup-multinetwork-support-for-pods)
- `--enable-ip-alias`: [VPC-native networking](https://cloud.google.com/kubernetes-engine/docs/concepts/alias-ips) - Pod IP addresses are natively routable within the cluster's VPC network and other VPC networks connected to it by VPC Network Peering
- `--gateway-api=standard`: [Enables Gateway API](https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api) for advanced traffic management and routing capabilities. We'll explore this in detail in Part 3 when using the inference gateway for intelligent routing.

Cluster creation takes 5-10 minutes.

### 5.2 Configure kubectl Access

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID}
```

Verify the cluster is accessible:

```bash
kubectl get nodes
```

---

## Step 6: Create GPU Node Pool

### 6.1 Create A3 Ultra Node Pool with Multi-Network Attachment

```bash
gcloud container node-pools create ${NAME_PREFIX}-h200-pool \
  --node-version=${GKE_VERSION} \
  --cluster=${CLUSTER_NAME} \
  --region=${REGION} \
  --node-locations=${ZONE} \
  --machine-type=${GPU_MACHINE_TYPE} \
  --accelerator=type=${GPU_ACCELERATOR_TYPE},count=${GPU_ACCELERATOR_COUNT},gpu-driver-version=LATEST \
  --num-nodes=2 \
  --min-nodes=${AUTOSCALING_MIN_NODES} \
  --max-nodes=${AUTOSCALING_MAX_NODES} \
  --enable-autoscaling \
  --location-policy=ANY \
  --enable-gvnic \
  --no-enable-autorepair \
  --enable-autoupgrade \
  --ephemeral-storage-local-ssd count=32 \
  --node-labels=workload=llm-inference \
  --reservation-affinity=any \
  --additional-node-network network=${NAME_PREFIX}-gvnic-net,subnetwork=${NAME_PREFIX}-gvnic-sub-0 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-0 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-1 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-2 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-3 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-4 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-5 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-6 \
  --additional-node-network network=${NAME_PREFIX}-rdma-net,subnetwork=${NAME_PREFIX}-rdma-sub-7
```

**Key configuration points**:

- **10 total networks**: 1 default + 1 gVNIC + 8 RDMA subnets (this is the GKE maximum)
- **32 local SSDs**: For fast model weight caching (approximately 6TB total ephemeral storage)
- **gVNIC enabled**: Required for high-performance GPU networking
- **GPU drivers**: Automatically installed by GKE with `gpu-driver-version=LATEST`
- **No auto-repair**: Disabled to prevent disruption of distributed inference workloads. For multi-node inference, node repairs can break active GPU-to-GPU connections and interrupt long-running inference requests spanning multiple nodes. For single-node models, auto-repair can be safely enabled.
- **Location policy (`--location-policy=ANY`)**: Controls how GKE distributes nodes across zones when using a regional cluster. With `ANY`, GKE can place nodes in any zone within the region where capacity is available. For single-zone node pools (like ours with RDMA requirements), this still applies to node placement within that zone. Alternative values include `BALANCED` (evenly distribute across zones) and `ANY` (maximize availability). For GPU workloads with limited availability, `ANY` provides the most flexibility.
- **Reservation affinity (`--reservation-affinity=any`)**: Determines how nodes utilize capacity reservations. Options include:
  - `any`: Consume capacity from any matching reservation, or use on-demand if no reservation is available (default behavior)
  - `specific`: Only consume capacity from a specific reservation (specified by `--reservation=RESERVATION_NAME`)
  - `none`: Never consume reserved capacity, always use on-demand or spot instances

  For GPU workloads, using `any` allows you to benefit from reservations when available while falling back to on-demand capacity. If you have a specific GPU reservation, use `specific` to guarantee those nodes use your reserved capacity. We'll explore capacity reservations in more detail in Part 6 of this series.

**Limitations:**
- Node pool must be single-zone (specified via `--node-locations`) due to RDMA network profile requirements
- Minimum nodes set to 2, additional **scaling can take 10-15 minutes** due to GPU driver installation

### 6.2 Verify GPU Nodes

```bash
# Wait for nodes to be ready
kubectl get nodes -l cloud.google.com/gke-nodepool=${NAME_PREFIX}-h200-pool -w

# Verify GPUs are detected
kubectl get nodes -o json | jq '.items[].status.allocatable | select(."nvidia.com/gpu")'
```

---

GPU nodes are up, but we need one more step to make the networks available to pods.

## Step 7: Configure Network Attachment Definitions

### Why This Step Is Critical

Earlier, we attached additional networks to our **nodes** (the virtual machines running Kubernetes). However, attaching networks to nodes doesn't automatically make them available to **pods** (the containers running your workloads).

### 7.1 Apply Network CRDs

GKE's multi-networking feature uses Custom Resource Definitions (CRDs) to create a bridge between node-level networks and pod-level networking. 

These CRDs define:

- **Which VPC network and subnet** a pod should connect to
- **How the network interface should behave** (standard networking vs RDMA)
- **What name** pods use to reference this network in their configuration

**The Two Resource Types**:

- **Network**: A logical network definition that pods can reference by name (e.g., `gvnic-1`, `rdma-0`)
- **GKENetworkParamSet**: The actual configuration specifying:
  - Which VPC and subnet to use
  - The `deviceMode` (how the interface works):
    - `NetDevice`: Standard Linux network device - goes through kernel networking stack (used for gVNIC)
    - `RDMA`: Direct hardware access - bypasses kernel for ultra-low latency (used for RDMA networks)

**Network Topology**:

Each A3 Ultra node has **10 network interfaces**:
- 1 default interface (Kubernetes cluster networking)
- 1 gVNIC interface (high-throughput general networking, 400 Gbps)
- 8 RDMA interfaces (GPU-to-GPU communication via RoCE, 3200 Gbps total)

We need to create a CRD for each additional network interface so pods can request them.

```bash
# Create network attachment for gVNIC
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: Network
metadata:
  name: gvnic-1
spec:
  parametersRef:
    group: networking.gke.io
    kind: GKENetworkParamSet
    name: gvnic-params
  type: "Device"

---
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: gvnic-params
spec:
  vpc: ${NAME_PREFIX}-gvnic-net
  vpcSubnet: ${NAME_PREFIX}-gvnic-sub-0
  deviceMode: NetDevice
EOF
```

```bash
# Create network attachments for each RDMA subnet
for N in $(seq 0 7); do
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: Network
metadata:
  name: rdma-${N}
spec:
  parametersRef:
    group: networking.gke.io
    kind: GKENetworkParamSet
    name: rdma-params-${N}
  type: Device
---
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: rdma-params-${N}
spec:
  vpc: ${NAME_PREFIX}-rdma-net
  vpcSubnet: ${NAME_PREFIX}-rdma-sub-${N}
  deviceMode: RDMA
EOF
done
```

### 7.2 How Pods Use These Networks

Once these CRDs are applied, you can reference them in your pod specifications using annotations. Here's an example:

```yaml
metadata:
  annotations:
    # Which interface is the default for pod networking
    networking.gke.io/default-interface: 'eth0'

    # List of all network interfaces to attach to this pod
    networking.gke.io/interfaces: |
      [
        {"interfaceName":"eth0","network":"default"},
        {"interfaceName":"eth1","network":"gvnic-1"},
        {"interfaceName":"eth2","network":"rdma-0"},
        {"interfaceName":"eth3","network":"rdma-1"},
        {"interfaceName":"eth4","network":"rdma-2"},
        {"interfaceName":"eth5","network":"rdma-3"},
        {"interfaceName":"eth6","network":"rdma-4"},
        {"interfaceName":"eth7","network":"rdma-5"},
        {"interfaceName":"eth8","network":"rdma-6"},
        {"interfaceName":"eth9","network":"rdma-7"}
      ]
```

**Inside the pod, this creates**:
- `eth0`: Default Kubernetes network (pod-to-pod, pod-to-service communication)
- `eth1`: gVNIC interface (for high-throughput data transfer, model downloads)
- `eth2-eth9`: 8 RDMA interfaces (for GPU-to-GPU communication via RoCE)

**Key Constraints**:

1. **All or nothing**: Each pod using RDMA **must request all 8 RDMA network interfaces**. You cannot request only some of them.

2. **Exclusive access**: GKE requires pods to use all available RDMA NICs on a node. This means:
   - Only **one pod per node** can use RDMA
   - You cannot share RDMA between multiple pods on the same node ([GPUDirect-RDMA setup guide](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom))

3. **Full GPU allocation**: The pod must also request **all 8 GPUs** on the node. RDMA and GPU resources are tightly coupled. ([A3 Ultra workload configuration](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx))

**Why these constraints exist**:

RDMA provides direct hardware access for maximum performance. Sharing RDMA between pods would require virtualization overhead that defeats the purpose of using RDMA. For distributed inference workloads, this design ensures each node runs a single, high-performance inference server that can communicate with other nodes at full RDMA speed.

**Important Limitations:**
- These pod network configurations are immutable after pod creation - you cannot add/remove networks from a running pod
- Network attachments consume from your VPC quota (subnet IP addresses)
- Each network interface adds slight pod startup overhead (typically 2-3 seconds per interface)

---

Now that pods can request RDMA networks, let's install the software needed to actually use them.

## Step 8: Install RDMA Device Plugin

### 8.1 Understanding the Software Stack

Before installing the NCCL plugin, let's try to understand how the different software components work together:

**The Complete Stack:**

1. **GPU Drivers** (installed automatically by GKE)
   - Low-level kernel modules that enable the OS to communicate with NVIDIA GPUs
   - Provide basic GPU functionality: compute, memory management, device access
   - Installed via `gpu-driver-version=LATEST` in the node pool configuration

2. **RDMA Network Drivers** (pre-installed)
   - Enable RDMA/RoCE capabilities on the Titanium ML network adapters
   - Provide kernel-level support for RDMA operations
   - Already configured on A3 Ultra machines with the 8 RDMA NICs

3. **[NCCL (NVIDIA Collective Communications Library)](https://developer.nvidia.com/nccl)**
   - Application-level library for multi-GPU and multi-node communication
   - Implements collective operations: all-reduce, broadcast, all-gather, etc.
   - Used by inference frameworks (vLLM, TGI) and training frameworks (PyTorch, JAX)
   - Without RDMA support, NCCL cannot use direct RDMA paths and must fall back to socket-based transport, losing the benefits of RDMA.
   - Abstracts away the underlying communication mechanism.

4. **NCCL RDMA Plugin** (what we're installing now)
   - **The missing link** that connects NCCL to Google Cloud's RoCE network
   - A specialized NCCL network plugin optimized for Google's Titanium ML network adapters
   - Enables NCCL to bypass the kernel networking stack and use direct RDMA operations over RoCE


Here a diagram that hopefully helps to visualize the process.

![NCCL Dataflow](part1/NCCL%20Stack.drawio.png "NCCL Dataflow")


**How Data Actually Flows:**

Understanding the complete data path helps clarify what each component does:

**Setup Phase (happens once during initialization):**
```
RDMA Plugin → RDMA NICs
```
The plugin configures the RDMA network interfaces: sets up connections, establishes queue pairs, registers memory regions, and configures routing between nodes.

**Data Transfer Phase (happens during inference/training):**
```
Inference Workload (vLLM/PyTorch) → NCCL → GPU Driver → RDMA NIC → Network → Remote RDMA NIC → Remote GPU
```

Step by step:
1. **Inference Workload** calls NCCL functions (e.g., `ncclAllReduce()`, `ncclSend()`)
2. **NCCL** (with RDMA plugin loaded) knows to use RDMA transport
3. **GPU Driver (CUDA)** sets up GPUDirect transfer from GPU memory
4. **GPU Driver** initiates PCIe peer-to-peer DMA from GPU directly to RDMA NIC
5. **RDMA NIC** sends data over RoCE network to remote NIC
6. **Remote RDMA NIC** writes data directly into remote GPU memory via PCIe peer-to-peer DMA

**The key benefit:** Data never touches CPU, system RAM, or kernel networking stack on either side. It's a true zero-copy, hardware-accelerated path: `GPU → NIC → Network → NIC → GPU`.




### 8.2 Deploy NCCL RDMA Plugin Installer

Now that we have a good grasp on **why** and **what** we are installing, let's go ahead and set up the NCCL RDMA Plugin.

```bash
# Apply NCCL RDMA installer DaemonSet (for A3 Ultra / A4 with GPUDirect RDMA)
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer.yaml
```

**Note**: This installer is specifically for A3 Ultra/A4 machines with GPUDirect-RDMA (RoCE). For A3 High/Mega with GPUDirect-TCPX/TCPXO, use the `gpudirect-tcpxo` installer instead.
As we covered ealier, GPUDirect-TCPX/TCPXO is a different transport mechanism from GPUDirect RDMA.

### 8.3 Verify RDMA Installation

```bash
# Check DaemonSet status
kubectl rollout status daemonset/nccl-rdma-installer -n kube-system

# Verify NCCL libraries are installed
kubectl -n kube-system logs $(kubectl get pods -n kube-system -l name=nccl-rdma-installer -o jsonpath='{.items[0].metadata.name}') -c nccl-rdma-installer
```

---

Everything is configured—time to validate that our RDMA networking actually delivers the performance we expect.

## Step 9: Validation and Testing

### 9.1 Deploy NCCL Test Job

Let's verify that RDMA networking is working correctly with an NCCL bandwidth test. This test uses the NCCL RDMA plugin optimized for RoCE.

We provide a complete NCCL test configuration in the repository that you can use:

```bash
# Clone the repository if you haven't already
git clone https://github.com/maci0/gke-inference-from-scratch.git
cd gke-inference-from-scratch/part1/terraform/tests

# Run the NCCL test script (adjust parameters as needed)
./run-nccl-test.sh --num-nodes 2 --gpus-per-node 8 --nodepool-name ${NAME_PREFIX}-h200-pool
```

The test script will:
1. Generate a Kubernetes Job manifest from the template
2. Deploy the NCCL test job across the specified number of nodes
3. Run an all-gather performance test to validate RDMA throughput
4. Output bandwidth results

**Alternative**: You can also use Google's official cluster-toolkit example, note that this will require the [jobset CRD](https://github.com/kubernetes-sigs/jobset) to be installed also which is something we are not covering currently.

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/cluster-toolkit/main/examples/gke-a3-ultragpu/nccl-jobset-example.yaml
```

For more details on the test configuration, see:
- Test template: [`part1/terraform/tests/nccl-test.yaml.tmpl`](part1/terraform/tests/nccl-test.yaml.tmpl)
- Test script: [`part1/terraform/tests/run-nccl-test.sh`](part1/terraform/tests/run-nccl-test.sh)

### 9.2 Monitor Test Results

```bash
# Watch pods come up
kubectl get pods -l job-name=nccl-test-job -w

# Check logs from the primary test pod (rank 0)
kubectl logs -f -l job-name=nccl-test-job

# Look for bandwidth results
kubectl logs -f -l job-name=nccl-test-job | grep -E "(Avg bus bandwidth|Avg latency)"
```

Expected output should show something like this:
```bash
        4096            64     float    none      -1    28.59    0.14    0.13      0    28.47    0.14    0.13      0
        8192           128     float    none      -1    31.73    0.26    0.24      0    30.97    0.26    0.25      0
       16384           256     float    none      -1    31.34    0.52    0.49      0    30.95    0.53    0.50      0
       32768           512     float    none      -1    31.75    1.03    0.97      0    31.57    1.04    0.97      0
       65536          1024     float    none      -1    34.44    1.90    1.78      0    33.94    1.93    1.81      0
      131072          2048     float    none      -1    35.02    3.74    3.51      0    37.36    3.51    3.29      0
      262144          4096     float    none      -1    37.97    6.90    6.47      0    35.64    7.35    6.90      0
      524288          8192     float    none      -1    41.05   12.77   11.97      0    39.73   13.20   12.37      0
     1048576         16384     float    none      -1    51.79   20.25   18.98      0    51.83   20.23   18.97      0
     2097152         32768     float    none      -1    68.33   30.69   28.77      0    66.46   31.55   29.58      0
     4194304         65536     float    none      -1    71.76   58.45   54.80      0    71.37   58.77   55.09      0
     8388608        131072     float    none      -1    86.61   96.85   90.80      0    85.57   98.03   91.91      0
    16777216        262144     float    none      -1    116.1  144.53  135.50      0    121.5  138.10  129.47      0
    33554432        524288     float    none      -1    179.1  187.31  175.60      0    171.6  195.54  183.32      0
    67108864       1048576     float    none      -1    272.5  246.26  230.87      0    272.2  246.52  231.11      0
   134217728       2097152     float    none      -1    490.1  273.86  256.74      0    485.1  276.69  259.40      0
   268435456       4194304     float    none      -1    873.4  307.35  288.14      0    868.1  309.24  289.91      0
   536870912       8388608     float    none      -1   1575.8  340.70  319.41      0   1575.9  340.67  319.37      0
  1073741824      16777216     float    none      -1   3136.1  342.38  320.98      0   3094.6  346.97  325.28      0
  2147483648      33554432     float    none      -1   6233.3  344.52  322.98      0   6097.5  352.19  330.18      0

# Out of bounds values : 0 OK
# Avg bus bandwidth    : 122.314
```

**Cleanup after testing:**

```bash
# Delete the test job
kubectl delete job nccl-test-job

# Delete the service
kubectl delete svc nccl-svc
```

**What You've Accomplished So Far:**

At this point, you've built a production-grade foundation for LLM inference:
- ✅ Multi-network VPCs with RDMA and gVNIC configured
- ✅ GKE cluster with GPU nodes and 3.2 Tbps RDMA bandwidth
- ✅ Network CRDs enabling pods to use RDMA
- ✅ NCCL plugin installed for RDMA communication
- ✅ Validated RDMA performance with NCCL tests

This infrastructure can support single-node inference, distributed multi-node inference, and advanced patterns like `KV cache sharing` that we'll explore in later parts of this series.

### 9.3 Verify Cluster Setup

```bash
# 1. Check all node pools
kubectl get nodes --show-labels

# 2. Verify GPU resources
kubectl describe nodes -l cloud.google.com/gke-nodepool=${NAME_PREFIX}-h200-pool | grep -A 5 "Allocatable"

# 3. Check network attachments
kubectl get networks.networking.gke.io

# 4. Verify NCCL plugin is running
kubectl get pods -n kube-system -l name=nccl-rdma-installer

# 5. Test GPU with a simple workload
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.0.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"'${NAME_PREFIX}'-h200-pool"}}}' \
  -- nvidia-smi
```

---

## Step 10: Smoke Test - Deploy Simple vLLM Pod

Now for the moment of truth. Let's verify that all our infrastructure work actually... works. Before tackling complex distributed workloads with RDMA, we'll start with something simple: a single-node vLLM deployment. Think of this as your "hello world" for GPU inference on your new cluster.

This smoke test will use a small model (google/gemma-3-2b-it) that loads quickly and proves your GPUs, drivers, and basic inference pipeline are operational.

### 10.1 Create a Simple vLLM Test Pod

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: vllm-smoke-test
  labels:
    app: vllm-smoke-test
spec:
  restartPolicy: Never
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    - --model
    - google/gemma-3-2b-it
    - --port
    - "8000"
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      value: "your-hf-token-if-needed"
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
    ports:
    - containerPort: 8000
      name: http
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
  tolerations:
  - operator: "Exists"
    key: nvidia.com/gpu
EOF
```

### 10.2 Verify the Deployment

```bash
# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-smoke-test --timeout=300s

# Check pod logs
kubectl logs vllm-smoke-test

# Port forward to test the API
kubectl port-forward vllm-smoke-test 8000:8000 &

# Test the API
curl http://localhost:8000/v1/models

# Send a test completion request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-2b-it",
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 100,
    "temperature": 0
  }'

# Clean up
kubectl delete pod vllm-smoke-test
```

**Expected Results:**
- Pod should start successfully and load the model
- The `/v1/models` endpoint should return the loaded model
- Completion requests should return generated text

This smoke test validates that:
- GPU drivers are working correctly
- Container can access GPU resources
- Basic inference functionality is operational

---

One final piece of infrastructure will make managing distributed workloads much easier.

## Step 11: Install LeaderWorkerSet

### 11.1 What is LeaderWorkerSet?

**[LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws)** is a Kubernetes API that simplifies the deployment and management of AI/ML multi-node inference workloads. It addresses common patterns in distributed model serving by:

- **Grouping Pods**: Treats multiple Pods as a logical unit (one leader + N workers)
- **Coordinated Lifecycle**: Ensures leader and worker Pods are created, scaled, and deleted together
- **Simplified Networking**: Provides predictable DNS names for Pod-to-Pod communication
- **Rolling Updates**: Manages updates across the entire group atomically

For distributed inference workloads using RDMA, LWS is particularly valuable because it ensures that all Pods in a replica group (leader + workers) are scheduled and started together, which is critical for establishing RDMA connections.

### 11.2 Install LeaderWorkerSet CRDs

```bash
# Install the LeaderWorkerSet CRDs and controller
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.3.0/manifests.yaml
```

### 11.3 Verify Installation

```bash
# Check that the CRD is installed
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io

# Check the controller is running
kubectl get pods -n lws-system

# Wait for the controller to be ready
kubectl wait --for=condition=Available deployment/lws-controller-manager -n lws-system --timeout=300s
```

### 11.4 Understanding LeaderWorkerSet Structure

A typical LeaderWorkerSet manifest looks like this:

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: distributed-inference
spec:
  replicas: 1  # Number of replica groups (each has 1 leader + N workers)
  leaderWorkerTemplate:
    size: 2  # Total pods per group (1 leader + 1 worker)
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        containers:
        - name: inference
          image: your-inference-image
          # Leader-specific configuration
    workerTemplate:
      metadata:
        labels:
          role: worker
      spec:
        containers:
        - name: inference
          image: your-inference-image
          # Worker-specific configuration
```

**Key Concepts:**
- **Replicas**: Number of independent replica groups to create
- **Size**: Total number of Pods in each group (1 leader + size-1 workers)
- **Leader Template**: Configuration for the leader Pod in each group
- **Worker Template**: Configuration for worker Pods in each group

**DNS Naming Pattern:**
- Leader: `<lws-name>-<group-index>` (e.g., `distributed-inference-0`)
- Workers: `<lws-name>-<group-index>-<worker-index>` (e.g., `distributed-inference-0-1`)

We'll use LeaderWorkerSet extensively in Part 4 when deploying distributed inference workloads with RDMA.

---

Now that you understand each component, let's look at how to automate this entire setup with Terraform.

## Terraform Automation

All of the above steps can be automated using Terraform. A complete implementation is available at:

**Repository**: [maci0/gke-inference-from-scratch/part1/terraform](https://github.com/maci0/gke-inference-from-scratch/part1/terraform)

The repository includes:

- **01-networking.tf**: Complete gVNIC and RDMA network setup with 8 subnets, firewall rules
- **02-gke-cluster.tf**: GKE cluster with multi-networking and dataplane v2
- **03-gke-nodepool.tf**: A3 Ultra node pool with all network attachments
- **04-kubernetes-manifests.tf**: Network CRDs, LWS controller and NCCL RDMA installer
- **05-reservation.tf**: GPU capacity reservation management

### Quick Start with Terraform

```bash
# Clone the repository
git clone https://github.com/maci0/gke-inference-from-scratch.git
cd gke-inference-from-scratch/part1/terraform

# Create terraform.tfvars
cat <<EOF > terraform.tfvars
project_id                = "your-project-id"
region                    = "us-central1"
zone                      = "us-central1-b"
name_prefix               = "llm-inference"
gpu_machine_type          = "a3-ultragpu-8g"
gpu_accelerator_type      = "nvidia-h200-141gb"
gpu_accelerator_count     = 8
autoscaling_min_nodes     = 2
autoscaling_max_nodes     = 4
gke_release_channel       = "STABLE"
EOF

# Initialize and apply
terraform init
terraform plan
terraform apply
```

### Key Terraform Resources

The Terraform configuration automatically creates:

1. **Networks**:
   - gVNIC VPC with single subnet (192.168.0.0/24, named gvnic-sub-0)
   - RDMA VPC with RoCE profile and 8 subnets (192.168.1-8.0/24, named rdma-sub-0 through rdma-sub-7)
   - Firewall rules for both networks

2. **GKE Cluster**:
   - Multi-networking enabled
   - Dataplane v2 for enhanced security
   - Workload Identity configured
   - Managed Prometheus and Gateway API

3. **GPU Node Pool**:
   - A3 Ultra machines with H200 GPUs
   - 10 network attachments (default + gVNIC + 8 RDMA)
   - 32 local SSDs for caching
   - Autoscaling configured

4. **Kubernetes Resources**:
   - Network CRDs for gVNIC and RDMA
   - NCCL RDMA installer DaemonSet

For detailed configuration options and troubleshooting, refer to the [repository README](https://github.com/maci0/gke-ai-from-scratch/blob/main/README.md).

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

With our foundation in place, we're ready to deploy inference frameworks. 

- Deploy and compare vLLM, TGI, and TensorRT-LLM
- Run initial performance benchmarks

---

## Resources

Want to dive deeper? Here are some additional resources to expand your understanding:

- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GKE GPUDirect RDMA Guide](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom)
- [GKE gVNIC Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/using-gvnic)
- [GCP GPU Docs](https://cloud.google.com/compute/docs/gpus)
---

## Feedback and Contributions

Questions or suggestions? [Open an issue](#) or reach out on [Twitter/LinkedIn](#).

---

**Coming Soon: Part 2 - Inference Frameworks**

Look forward to Part 2, where we will cover different inference frameworks including vLLM, TGI, and TensorRT-LLM, and how to deploy them on our GPU infrastructure.
