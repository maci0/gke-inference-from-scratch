# Building a Production Ready LLM Inferencing Platform on GKE from Scratch – Part 2: Inference Pattern

## Introduction

You've built an LLM application that works great in development. Now you need to serve it at scale. Thousands of concurrent requests with sub-second latency. Your prototype ran on a single GPU, but your production model won't fit on one machine. Time for some real infrastructure work.

This series walks through building a production LLM inference platform on GKE from scratch. We'll cover everything you need: networking, GPU clusters, inference frameworks, routing, and optimization.

Tools like `cluster toolkit` and `xpk` can automate much of this, but understanding what's happening under the hood matters. When your cluster starts acting weird at 3 AM, you'll actually know how to debug it.

We'll start with `gcloud` commands to see what's happening at each step, then automate it with Terraform for production use. With each step we will also explore and explain some of the underlying technologies that make everything work.

### Series Overview

This series covers:

1. [Base Infrastructure](part1.md) - GKE setup with GPU support and RDMA networking
2. **[Inference Pattern](part2.md)** - vLLM, single/multi-GPU and multi-node deployments
   - Tensor & Pipeline Parallelism
   - LeaderWorkerSet & RayService
3. [Inference Gateway](part3.md) - Intelligent routing
4. [Distributed Inferencing](part4.md) - Advanced patterns with llm-d
   - KV Cache sharing across instances
   - Disaggregated Prefill/Decode architecture
5. Storage Options - Model storage, caching, and versioning strategies
6. Consumption Models - DWS, Flex-Start, Ondemand, Spot & Cost Optimization Techniques
7. Accelerating Pod Startup - Techniques to minimize cold start time
   - Secondary boot disk
   - Pod Snapshot / Restore
8. Testing and Benchmarking - Performance validation and optimization

---

## Part 2: Inference Frameworks

In [Part 1](part1.md), we built the foundation: a GKE cluster with GPU nodes, RDMA networking, and all the necessary infrastructure for high-performance inference. Now it's time to put that infrastructure to work by deploying actual inference frameworks.

### Prerequisites

Before starting this part, ensure you have:

1. **A GKE cluster with GPU support**: Either complete [Part 1](part1.md) to build the cluster from scratch, or use an existing cluster set up with [cluster toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
2. **HuggingFace token**: Required for downloading gated models. Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
3. **jq**: Command-line JSON processor for formatting API responses. Install with `sudo apt-get install jq` (Debian/Ubuntu) or `brew install jq` (macOS)

This part focuses on **vLLM**, one of the most popular and performant open-source inference frameworks for large language models. We'll cover both single-node and multi-node deployment patterns, showing you how to leverage your infrastructure for models of any size.

While Google Cloud offers [GKE Inference Quickstart](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/inference-quickstart) to simplify the deployment of  inference workloads, in this series we are more interested in the underpinnings.

### What We'll Cover

In this part, we'll explore:

- **Single-GPU Inference with vLLM**: Deploy small models on a single GPU
- **Multi-GPU Single-Node Inference**: Scale to larger models using tensor parallelism
- **Multi-Node Inference with vLLM**: Distributed inference across multiple nodes
- **LeaderWorkerSet**: Kubernetes orchestration pattern for multi-node workloads
- **Performance Considerations**: Choosing the right deployment pattern

---

## Deployment Pattern Comparison

Before diving into the details, here's a comparison of the three deployment patterns covered in this guide:

**Note:** Model sizes and VRAM requirements listed here are rough estimates and vary based on configuration parameters such as quantization, KV cache size, batch size, sequence length, and other runtime settings. Different node and accelerator types (A100, H100, H200, B200, etc.) may be better suited for different requirements.

| Aspect | Single-GPU | Multi-GPU (Single-Node) | Multi-Node |
|--------|-----------|------------------------|------------|
| **Model Size** | < 10B params | 10B-70B params | > 70B params (175B, 405B, 1T+) |
| **GPU Memory** | < 20GB | 20–140GB | 140GB, 350GB, 810GB, 2TB+ |
| **Advantages** | • Maximum Simplicity<br>• Lowest Latency<br>• Most Cost Effective<br>• Fast Iteration | • Higher Model Capacity <br>• Faster Inference<br>• Simple Setup (no RDMA needed)<br>• High GPU-to-GPU Bandwidth (via NVLink) | • Supports largest models<br>• Maximum throughput<br>• Highest possible performance |
| **Best For** | • Development and testing<br>• Low-traffic production<br>• Quick prototyping<br>• Small models | • Production workloads requiring low latency<br>• Medium-sized models<br>• Cost-effective scaling without multi-node complexity | • Production workloads with huge models<br>• Maximum throughput requirements<br>• Highest performance needs |
| **Model Examples** | Llama-3-8B, Mistral-7B, Gemma-7B | Llama-3-70B (2-8 GPUs) | Llama-3-405B (16+ GPUs) |
| **Complexity** | Minimal | Low | High |
| **Setup Time** | Seconds | Seconds | Minutes |
| **Cost** | Lowest | Medium | Highest |
| **Latency** | Lowest | Low | Higher (RDMA overhead) |
| **Throughput** | Low | Medium-High | Highest |
| **Interconnect** | N/A | NVLink | RDMA + NVLink |
| **RDMA Required** | No | No | Yes |
| **Requirements** | GPU drivers | GPU drivers, NVLink | GPU drivers, NVLink, RDMA networking, LeaderWorkerSet |

**NVLink Bandwidth by VM Type:**
- **N1 VMs (V100 GPUs)**: 300 GB/s bidirectional per GPU (NVLink 2nd gen)
- **A2 VMs (A100 GPUs)**: 600 GB/s bidirectional per GPU (NVLink 3rd gen)
- **A3 Standard VMs (H100 GPUs)**: 900 GB/s bidirectional per GPU (NVLink 4th gen)
- **A3 Ultra VMs (H200 GPUs)**: 900 GB/s bidirectional per GPU (NVLink 4th gen)
- **A4 VMs (B200 GPUs)**: 1,800 GB/s bidirectional per GPU (NVLink 5th gen)

**Important Note:** Not all multi-GPU VMs on GCP have NVLink. For example, `g2-standard` VMs with multiple NVIDIA L4 GPUs connect the GPUs via PCIe. While this allows for multi-GPU processing, the inter-GPU communication bandwidth is significantly lower than NVLink, making them better suited for inference workloads that can be parallelized without frequent, high-bandwidth data exchange between GPUs.

For a full list of available GPU types, interconnects, and performance characteristics, please consult the documentation at https://cloud.google.com/compute/docs/gpus.

---

## Setup: Create HuggingFace Token Secret

Before deploying any inference workloads, create a Kubernetes secret for your HuggingFace token. This allows secure access to gated models.

```bash
# Export your HuggingFace token
export HF_TOKEN="your-hf-token-here"

# Create a Kubernetes secret
kubectl create secret generic hf-token \
  --from-literal=token=${HF_TOKEN}

# Verify the secret was created
kubectl get secret hf-token
```

---

## Section 1: Single-GPU Inference with vLLM

The simplest deployment pattern uses a single GPU. This is ideal for small models, development, testing, and lower-traffic production workloads.

### 1.1 Deploy a Single-GPU vLLM Pod

Let's start with a basic vLLM deployment using a small model to verify everything works.

```yaml
cat <<EOF| kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: vllm-single-node
  labels:
    app: vllm-inference
spec:
  restartPolicy: Never
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    command:
      - "/bin/bash"
      - "-c"
    args:
      - |
        set -e
        python3 -m vllm.entrypoints.openai.api_server \
          --model google/gemma-3-1b-it \
          --port 8000 \
          --tensor-parallel-size 1
    ports:
    - containerPort: 8000
      name: http
    env:
    - name: LD_LIBRARY_PATH
      value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
    - name: library-dir-host
      mountPath: /usr/local/nvidia
      readOnly: true
  volumes:
  - name: library-dir-host
    hostPath:
      path: /home/kubernetes/bin/nvidia
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
EOF
```

**Key Configuration:**
Let's take a moment to understand the key configuration here.

```yaml
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
```
This sets the `HUGGING_FACE_HUB_TOKEN` environment variable inside the container. Instead of hardcoding a sensitive token directly in the manifest, it uses `valueFrom.secretKeyRef` to securely reference the `hf-token` Kubernetes Secret we created earlier. This is the standard, secure way to handle credentials in Kubernetes. The vLLM container will automatically use this environment variable to authenticate with the Hugging Face Hub when downloading the model.


```yaml
    volumeMounts:
    - name: library-dir-host
      mountPath: /usr/local/nvidia
      readOnly: true
  volumes:
  - name: library-dir-host
    hostPath: 
      path: /home/kubernetes/bin/nvidia
```
This configuration is crucial for GPU access. It mounts the NVIDIA driver directory from the host node (`/home/kubernetes/bin/nvidia`) into the container at `/usr/local/nvidia`. This ensures the container uses the same driver version as the host kernel module, preventing version mismatch issues.

```yaml
    env:
    - name: LD_LIBRARY_PATH
      value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
```
`LD_LIBRARY_PATH` is an environment variable that tells the dynamic linker where to find shared libraries. By setting this path, we instruct the vLLM application to load the necessary GPU libraries (like `libcuda.so`) from the `/usr/local/nvidia/lib64` directory, which we mounted from the host. This ensures the container uses the correct, host-compatible drivers to communicate with the GPU hardware.

```yaml
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
```
The `nodeSelector` is a fundamental Kubernetes feature that tells the scheduler which nodes a Pod is eligible to run on. It works by matching the Pod's `nodeSelector` labels with the labels on the nodes.

In this case, we are instructing Kubernetes to schedule this Pod *only* on nodes that have the label `cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool`. We configured this in Part 1 of this series when we set up the nodepool.
Using the label ensures our Pod lands on a machine in our specialized GPU node pool.

GKE also provdes a set of built in labels that can be used to further configure placement of our workloads. For example:
```yaml
cloud.google.com/gke-accelerator: "nvidia-h200-141gb" # Selects nodes with a specific GPU type
topology.kubernetes.io/zone: "us-central1-b" # Selects nodes in a specific zone
cloud.google.com/gke-machine-type: "a3-ultragpu-8g" # Selects a specific machine type
```

These types of topology specific labels become more important as we leverage high speed interconnects across different nodes later.

```yaml
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
```
This block specifies the compute resources required by the container.

- **`resources.requests`** tells the Kubernetes scheduler the minimum resources the container needs. The scheduler will only place this Pod on a node that has at least one `nvidia.com/gpu` available, guaranteeing it runs on a GPU-enabled machine.
- **`resources.limits`** defines the maximum resources the container can use. 
Setting `requests` equal to `limits` ensures the Pod gets a `Guaranteed` Quality of Service (QoS) class for the GPU, preventing other pods from trying to use the same device and guaranteeing predictable performance.
- **`nvidia.com/gpu: 1`** is the resource name for NVIDIA GPUs. The value `1` indicates that the container needs exclusive access to one full GPU.

**vLLM Parameters**
- `--model google/gemma-3-1b-it`: The model to use. It will either be downloaded from huggingface hub or, if present loaded from the default vLLM data directory.
- `--tensor-parallel-size 1`: Use a single GPUs for the model.

**Note on Model Loading:**
Each time a pod starts, it will download the model from HuggingFace Hub. For small models like Gemma-3-1B, this takes a few minutes. For larger models (70B+), initial startup can take 10-20 minutes or longer. In Part 5, we'll cover how to significantly speed this up by caching models in GCS, using local SSDs and other storage options.


### 1.2 Verify the Deployment

```bash
# Wait for pod to be ready (may take several minutes for model download)
kubectl wait --for=condition=Ready pod/vllm-single-node --timeout=300s

# Check pod logs
kubectl logs vllm-single-node

# Port forward to test the API
kubectl port-forward vllm-single-node 8000:8000 &

# Test the API
curl http://localhost:8000/v1/models | jq

# Send a test completion request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-1b-it",
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 100,
    "temperature": 0
  }' | jq

# Clean up
kubectl delete pod vllm-single-node
```

**Expected Results:**
- Pod should start successfully and load the model
- The `/v1/models` endpoint should return the loaded model
- Completion requests should return generated text

Congratulations, you just deployed an LLM!

---

## Section 2: Multi-GPU Single-Node Inference with vLLM

As models grow larger, a single GPU often isn't enough. Multi-GPU single-node inference uses **tensor parallelism** to split a model across multiple GPUs on the same node, connected via high-speed **NVLink**.

### 2.1 Understanding Tensor Parallelism

**How It Works:**

Tensor parallelism splits individual model layers across multiple GPUs. Each GPU:
1. Holds a portion of each layer's parameters
2. Processes the same batch of data
3. Communicates intermediate results via NVLink
4. Combines results to produce the output

**Example: 8-GPU Tensor Parallelism**
```
Input Batch → GPU 0 (Layer Shard 0) ──┐
            → GPU 1 (Layer Shard 1) ──┤
            → GPU 2 (Layer Shard 2) ──┤
            → GPU 3 (Layer Shard 3) ──├─→ All-Reduce → Output
            → GPU 4 (Layer Shard 4) ──┤
            → GPU 5 (Layer Shard 5) ──┤
            → GPU 6 (Layer Shard 6) ──┤
            → GPU 7 (Layer Shard 7) ──┘
```

**Communication Pattern:**
- All GPUs must synchronize at each layer
- NVLink enables this with minimal latency (<5 microseconds)
- Total communication per token: ~10-50 microseconds

### 2.1.1 The Role of vLLM and NCCL

So how does this communication actually happen? This is where vLLM and NCCL come into play.

1.  **vLLM's Role**: At a high level, vLLM is responsible for:
    *   **Model Sharding**: It partitions the model's weights (tensors) and distributes them across the number of GPUs.
    *   **Execution Orchestration**: It manages the inference process, ensuring that each GPU processes its part of the computation for each layer.
    *   **Calling NCCL**: When GPUs need to exchange data (e.g., after a matrix multiplication to combine partial results), vLLM calls NCCL functions like `all-reduce`.

2.  **NCCL's Role (NVIDIA Collective Communications Library)**: NCCL is a highly optimized library for inter-GPU communication. Its job is to:
    *   **Execute Collectives**: It performs collective communication operations (like `all-reduce`, `broadcast`, `all-gather`) efficiently.
    *   **Abstract the Hardware**: NCCL automatically detects the best available communication path between GPUs.
        *   **On a single node (like our A3 Ultra)**, it will use the high-speed **NVLink** interconnect, which provides direct, low-latency communication between GPUs. This is why we don't need RDMA for single-node tensor parallelism.
        *   **Across multiple nodes**, it will use the network. With the RDMA plugin we installed in Part 1, NCCL will bypass the kernel and use RDMA over RoCE for the fastest possible multi-node communication. On other machine types it may use different transport mechanisms.

In short, vLLM handles the "what" (splitting the model and orchestrating the work), while NCCL handles the "how" (moving data between GPUs as fast as possible). This combination allows for efficient scaling across multiple GPUs both within a single machine and across nodes.

### 2.2 Deploy Multi-GPU vLLM (2 GPUs)

Let's start with a 2-GPU example for a medium-sized model:

```yaml
cat <<EOF| kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: vllm-multi-gpu-2
  labels:
    app: vllm-inference
spec:
  restartPolicy: Never
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    command:
      - "/bin/bash"
      - "-c"
    args:
      - |
        set -e
        cat /usr/local/gib/scripts/set_nccl_env.sh
        python3 -m vllm.entrypoints.openai.api_server \
          --model google/gemma-3-27b-it \
          --port 8000 \
          --tensor-parallel-size 2
    ports:
    - containerPort: 8000
      name: http
    env:
    - name: NCCL_NET_PLUGIN
      value: "none"
    - name: NCCL_TUNER_CONFIG_PATH
      value: /usr/local/gib/configs/tuner_config_a3u.txtpb
    - name: VLLM_LOGGING_LEVEL
      value: DEBUG
    - name: NCCL_DEBUG
      value: TRACE
    - name: LD_LIBRARY_PATH
      value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
    resources:
      requests:
        nvidia.com/gpu: 2
      limits:
        nvidia.com/gpu: 2
    volumeMounts:
    - name: library-dir-host
      mountPath: /usr/local/nvidia
      readOnly: true
    - name: gib
      mountPath: /usr/local/gib
      readOnly: true
    - name: shm
      mountPath: /dev/shm
  volumes:
  # vLLM needs to access the host's shared memory for tensor parallel inference.
  - name: shm
    emptyDir:
      medium: Memory
  - name: library-dir-host
    hostPath:
      path: /home/kubernetes/bin/nvidia
  - name: gib
    hostPath:
      path: /home/kubernetes/bin/gib
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
EOF
```

**Key Configuration:**

Let's break down the new configuration items in this multi-GPU deployment.

```yaml
    volumeMounts:
    - name: gib
      mountPath: /usr/local/gib
      readOnly: true
...
  volumes:
  - name: gib
    hostPath:
      path: /home/kubernetes/bin/gib
```
Contains additional configuration and NCCL plugins for multi-gpu and multi-node workloads.
Specifically the RDMA network plugins for NCCL. As well as the machine specific tuner plugin for NCCL.

```yaml
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
...
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
```

When vLLM runs in tensor-parallel mode, it spawns multiple worker processes. These processes need a fast way to share data and coordinate with each other. The standard mechanism for this is shared memory. This configuration provides a large, dedicated, RAM-based shared memory space for the vLLM processes.


```yaml
    resources:
      requests:
        nvidia.com/gpu: 2
      limits:
        nvidia.com/gpu: 2
```
Similar to the single-GPU example, this requests GPU resources. By setting the value to `2`, we instruct Kubernetes to find a single node with at least two available GPUs and assign them exclusively to this Pod. This allows vLLM to perform tensor parallelism across multiple GPUs within the same machine.

```yaml
    - name: NCCL_NET_PLUGIN
      value: "none"
    - name: NCCL_TUNER_CONFIG_PATH
      value: /usr/local/gib/configs/tuner_config_a3u.txtpb
```
Since we are using a GPU that is on the same node, we can configure NCCL to skip the network plugin initialization and just use NVLink for GPU-to-GPU configuration. 
These environment variables are used to tune NCCL's behavior for optimal single-node performance:

`NCCL_NET_PLUGIN=none`: For communication within a single node, NCCL automatically uses the fastest available interconnect, which is NVLink on A3 VMs. This variable explicitly tells NCCL not to load any network plugins (like the RDMA plugin), which is a slight optimization that can speed up initialization.

`NCCL_TUNER_CONFIG_PATH:` This points to a configuration file (provided via the `gib` volume mount) containing pre-tuned parameters for NCCL communication specifically for A3 Ultra machines. This tuner helps NCCL choose the most efficient algorithm for collective operations based on the hardware topology, further boosting performance. 
This setting will be different for different machine types and interconnects.

**vLLM Parameters**
- `--tensor-parallel-size 2`: Split model across 2 GPUs
- No additional networking annotations needed (NVLink is automatic)

### 2.3 Deploy Multi-GPU vLLM (8 GPUs - Full Node)

For maximum single-node performance, use all 8 GPUs:

```yaml
cat <<EOF| kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: vllm-multi-gpu-8
  labels:
    app: vllm-inference
spec:
  restartPolicy: Never
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    command:
      - "/bin/bash"
      - "-c"
    args:
      - |
        set -e
        source /usr/local/gib/scripts/set_nccl_env.sh
        python3 -m vllm.entrypoints.openai.api_server \
          --model google/gemma-3-27b-it \
          --port 8000 \
          --tensor-parallel-size 8
    ports:
    - containerPort: 8000
      name: http
    env:
    - name: VLLM_LOGGING_LEVEL
      value: DEBUG
    - name: NCCL_DEBUG
      value: TRACE
    - name: LD_LIBRARY_PATH
      value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
    volumeMounts:
    - mountPath: /dev/shm
      name: shm
    resources:
      requests:
        nvidia.com/gpu: 8
      limits:
        nvidia.com/gpu: 8
    ports:
    - containerPort: 8000
      name: http
  volumes:
  # vLLM needs to access the host's shared memory for tensor parallel inference.
  - name: shm
    emptyDir:
      medium: Memory
  nodeSelector:
    cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
EOF
```



**Key Configuration:**



TODO: explain /usr/local/gib/scripts/set_nccl_env.sh

**vLLM Parameters**
- `--tensor-parallel-size 8`: Use all 8 GPUs on the node
- This configuration provides maximum throughput for models up to beyond 70B parameters

### 2.4 Verify Multi-GPU Deployment

```bash
# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-multi-gpu-8 --timeout=600s

# Check logs - should show tensor parallel initialization
kubectl logs vllm-multi-gpu-8 | grep "tensor_parallel"

# Verify GPU usage
kubectl exec vllm-multi-gpu-8 -- nvidia-smi

# Port forward and test
kubectl port-forward vllm-multi-gpu-8 8000:8000 &

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "prompt": "Explain how tensor parallelism works:",
    "max_tokens": 200
  }' | jq
```

### 2.5 Performance Tuning for Multi-GPU

**Memory Management:**
```bash
# Adjust KV cache size based on workload
--gpu-memory-utilization 0.9  # Use 90% of GPU memory (default: 0.9)
--max-num-seqs 256           # Max concurrent sequences
--max-model-len 4096         # Reduce if running out of memory
```

**Throughput Optimization:**
```bash
--enable-chunked-prefill     # Better batching for mixed workloads
--max-num-batched-tokens 8192  # Control batch size
```

**Monitoring GPU Utilization:**
```bash
# Watch GPU usage in real-time
kubectl exec vllm-multi-gpu-8 -- watch -n 1 nvidia-smi

# Check for optimal GPU utilization:
# - GPU Utilization: Should be 80-95% during inference
# - Memory Usage: Should be close to max without OOM errors
# - Temperature: Should be stable (typically 60-80°C)
```

### 2.6 Common Issues and Solutions

**Issue: Out of Memory (OOM) Errors**
```bash
# Solution 1: Reduce max sequence length
--max-model-len 2048

# Solution 2: Reduce concurrent sequences
--max-num-seqs 128

# Solution 3: Reduce KV cache
--gpu-memory-utilization 0.85
```

**Issue: Low GPU Utilization**
```bash
# Solution: Enable continuous batching and increase batch size
--enable-chunked-prefill
--max-num-batched-tokens 16384
```

**Issue: Slow Model Loading**
```bash
# Solution: Use local SSD for model caching
# Add volume mount in pod spec:
volumeMounts:
- name: model-cache
  mountPath: /root/.cache/huggingface
volumes:
- name: model-cache
  hostPath:
    path: /mnt/stateful_partition/huggingface
    type: DirectoryOrCreate
```

---

## Section 3: Multi-Node Inference with vLLM

For models larger than 70B parameters or when you need maximum throughput, multi-node distributed inference becomes necessary. This is where our RDMA infrastructure really shines.

### 3.1 Install LeaderWorkerSet

Before deploying multi-node workloads, we need to install LeaderWorkerSet, a Kubernetes API that simplifies the management of distributed inference workloads.

#### What is LeaderWorkerSet?

**[LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws)** is a Kubernetes API that simplifies the deployment and management of AI/ML multi-node inference workloads. It addresses common patterns in distributed model serving by:

- **Grouping Pods**: Treats multiple Pods as a logical unit (one leader + N workers)
- **Coordinated Lifecycle**: Ensures leader and worker Pods are created, scaled, and deleted together
- **Simplified Networking**: Provides predictable DNS names for Pod-to-Pod communication
- **Rolling Updates**: Manages updates across the entire group atomically

For distributed inference workloads using RDMA, LWS is particularly valuable because it ensures that all Pods in a replica group (leader + workers) are scheduled and started together, which is critical for establishing RDMA connections.

#### Install LeaderWorkerSet CRDs

```bash
# Install the LeaderWorkerSet CRDs and controller
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.7.0/manifests.yaml
```

#### Verify Installation

```bash
# Check that the CRD is installed
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io

# Check the controller is running
kubectl get pods -n lws-system

# Wait for the controller to be ready
kubectl wait --for=condition=Available deployment/lws-controller-manager -n lws-system --timeout=300s
```

#### Understanding LeaderWorkerSet Structure

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

### 3.2 Deploy Multi-Node vLLM with LeaderWorkerSet

#### 3.2.1 Understanding Pipeline Parallelism

While **Tensor Parallelism** splits *each layer* of a model across multiple GPUs, **Pipeline Parallelism** splits the model *between layers*. The model is divided into sequential stages, and each stage is assigned to a different GPU or group of GPUs.

**How It Works:**

1.  **Partitioning**: The model's layers are partitioned into contiguous blocks. For a 2-stage pipeline, the first half of the layers goes to Stage 0 (e.g., Node 1) and the second half goes to Stage 1 (e.g., Node 2).
2.  **Forward Pass**: An input batch is processed by Stage 0. The resulting intermediate activations are then sent over the network (using RDMA) to Stage 1, which completes the forward pass and produces the final output.
3.  **The "Pipeline Bubble"**: A naive implementation is inefficient. While Stage 1 is processing, Stage 0 is idle, waiting for the next batch. This idle time is known as the "pipeline bubble."
4.  **Micro-batching**: To solve this, the input batch is split into smaller "micro-batches." Stage 0 processes the first micro-batch and sends it to Stage 1. As Stage 1 works on the first micro-batch, Stage 0 immediately starts on the second micro-batch. This overlapping execution keeps all GPUs busy and maximizes throughput.

Here is a diagram illustrating the flow with micro-batches across two pipeline stages:

```
Time ->
         +------------------+------------------+------------------+
Stage 0: |  Micro-batch 1   |  Micro-batch 2   |  Micro-batch 3   | ...
         +------------------+------------------+------------------+
                   |                  |                  |
                   v (RDMA)           v (RDMA)           v (RDMA)
         +------------------+------------------+------------------+
Stage 1: |      Idle        |  Micro-batch 1   |  Micro-batch 2   | ...
         +------------------+------------------+------------------+
```

#### Combining Pipeline and Tensor Parallelism

vLLM allows you to combine both parallelism strategies for maximum scalability. This is essential for serving extremely large models.

-   **`tensor_parallel_size`**: The number of GPUs within a single stage that work together to execute a part of the model. Communication within this group happens over high-speed NVLink.
-   **`pipeline_parallel_size`**: The number of stages the model is split into. Communication between stages happens over the network (RDMA).

**Example**: To serve a very large model on 16 GPUs (2 nodes with 8 GPUs each), you would configure:
-   `pipeline_parallel_size=2`: The model is split into two stages.
-   `tensor_parallel_size=8`: Each stage runs on a full node of 8 GPUs using tensor parallelism.

This hybrid approach allows you to scale to models of virtually any size by adding more nodes (increasing `pipeline_parallel_size`) while fully utilizing the NVLink interconnect within each node.

#### 3.2.2 Deploy Multi-Node vLLM

Now let's deploy a multi-node vLLM setup that uses RDMA for inter-node communication.

```yaml
cat <<EOF| kubectl apply -f -
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-multi-node
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2 # 2 nodes total (1 leader + 1 worker)
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          role: leader
        annotations:
          networking.gke.io/default-interface: 'eth0'
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
      spec:
        containers:
        - name: vllm-leader
          image: vllm/vllm-openai:latest
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]
          command:
            - "/bin/bash"
            - "-c"
          args:
            - |
              set -e
              source /usr/local/gib/scripts/set_nccl_env.sh
              bash /vllm-workspace/examples/online_serving/multi-node-serving.sh leader --ray_cluster_size=\${LWS_GROUP_SIZE}
              python3 -m vllm.entrypoints.openai.api_server \
                --model google/gemma-3-27b-it \
                --port 8000 \
                --tensor-parallel-size 8 \
                --pipeline-parallel-size \${LWS_GROUP_SIZE} \
                --distributed-executor-backend ray
          env:
          - name: LD_LIBRARY_PATH
            value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
          - name: HUGGING_FACE_HUB_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-token
                key: token
          - name: NCCL_DEBUG
            value: "INFO"
          resources:
            requests:
              nvidia.com/gpu: 8
            limits:
              nvidia.com/gpu: 8
          ports:
          - containerPort: 8000
            name: http
          readinessProbe:
            tcpSocket:
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
          # Add volume mounts for the container
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
          - name: gib
            mountPath: /usr/local/gib
            readOnly: true
          - name: library-dir-host
            mountPath: /usr/local/nvidia
            readOnly: true
        nodeSelector:
          cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
        # Define volumes for the pod
        volumes:
        - name: shm
          emptyDir:
            medium: Memory
        - name: gib
          hostPath:
            path: /home/kubernetes/bin/gib
        - name: library-dir-host
          hostPath:
            path: /home/kubernetes/bin/nvidia
    workerTemplate:
      metadata:
        labels:
          role: worker
        annotations:
          networking.gke.io/default-interface: 'eth0'
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
      spec:
        containers:
        - name: vllm-worker
          image: vllm/vllm-openai:latest
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]
          command:
            - "/bin/bash"
            - "-c"
          args:
            - |
              set -e
              source /usr/local/gib/scripts/set_nccl_env.sh
              bash /vllm-workspace/examples/online_serving/multi-node-serving.sh worker --ray_address=\${LWS_LEADER_ADDRESS}
          env:
          - name: LD_LIBRARY_PATH
            value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
          - name: HUGGING_FACE_HUB_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-token
                key: token
          - name: NCCL_DEBUG
            value: "INFO"
          resources:
            requests:
              nvidia.com/gpu: 8
            limits:
              nvidia.com/gpu: 8
          # Add volume mounts for the container
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
          - name: gib
            mountPath: /usr/local/gib
            readOnly: true
          - name: library-dir-host
            mountPath: /usr/local/nvidia
            readOnly: true
        nodeSelector:
          cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
        # Define volumes for the pod
        volumes:
        - name: shm
          emptyDir:
            medium: Memory
        - name: gib
          hostPath:
            path: /home/kubernetes/bin/gib
        - name: library-dir-host
          hostPath:
            path: /home/kubernetes/bin/nvidia
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-leader
spec:
  ports:
    - name: http
      port: 8000
      protocol: TCP
      targetPort: 8000
  selector:
    leaderworkerset.sigs.k8s.io/name: vllm-multi-node
    role: leader
  type: ClusterIP
EOF
```

**Key Configuration Points:**
```yaml
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]
```
This security setting is important for high-performance communication. The `IPC_LOCK` capability allows a process to lock memory pages into RAM, preventing them from being swapped to disk.

Frameworks like vLLM and libraries like NCCL/RDMA need to "pin" memory for Direct Memory Access (DMA) operations by the GPU and network hardware. Pinning memory guarantees low-latency access and ensures that the physical address of the memory doesn't change, which is a requirement for DMA. Without this capability, attempts to pin memory would fail, leading to severe performance degradation or errors.

1. **Network Annotations**: Each pod requests all 10 network interfaces (default + gvnic + 8 RDMA)
2. **Tensor Parallel Size**: `16` = 8 GPUs per node × 2 nodes
3. **GPU Resources**: Each pod requests all 8 GPUs on its node
4. **NCCL Environment Variables**:
   - `NCCL_DEBUG=INFO`: Enables NCCL logging for troubleshooting

Under the hood `/vllm-workspace/examples/online_serving/multi-node-serving.sh leader --ray_cluster_size=\${LWS_GROUP_SIZE}` uses Ray to distribute the inferencing workload across the different nodes.

### 3.3 Deploy using RayService

TODO: What is Ray?
What is a RayService?

```yaml
cat <<EOF| kubectl apply -f -

apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vllm-rayservice
spec:
  rayClusterConfig:
    rayVersion: '2.9.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
            - name: ray-head
              image: vllm/vllm-openai:latest
              resources:
                limits:
                  nvidia.com/gpu: 0
    workerGroupSpecs:
      - groupName: worker-group
        replicas: 2
        rayStartParams: {}
        template:
          metadata:
            annotations:
              networking.gke.io/default-interface: 'eth0'
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
          spec:
            containers:
              - name: ray-worker
                image: vllm/vllm-openai:latest
                command: ["source /usr/local/gib/scripts/set_nccl_env.sh"]
                resources:
                  requests:
                    nvidia.com/gpu: 8
                  limits:
                    nvidia.com/gpu: 8
                env:
                  - name: NCCL_DEBUG
                    value: "INFO"
                  - name: LD_LIBRARY_PATH
                    value: /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64
                  - name: HUGGING_FACE_HUB_TOKEN
                    valueFrom:
                      secretKeyRef:
                        name: hf-token
                        key: token
                securityContext:
                  capabilities:
                    add: ["IPC_LOCK"]
                volumeMounts:
                - name: shm
                  mountPath: /dev/shm
                - name: gib
                  mountPath: /usr/local/gib
                  readOnly: true
                - name: library-dir-host
                  mountPath: /usr/local/nvidia
                  readOnly: true
            nodeSelector:
              cloud.google.com/gke-nodepool: ${NAME_PREFIX}-h200-pool
            volumes:
            - name: shm
              emptyDir:
                medium: Memory
            - name: gib
              hostPath:
                path: /home/kubernetes/bin/gib
            - name: library-dir-host
              hostPath:
                path: /home/kubernetes/bin/nvidia
EOF
```


### 3.4 Verify Multi-Node Deployment

```bash
# Check LeaderWorkerSet status
kubectl get leaderworkerset vllm-multi-node

# Watch pods come up
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-multi-node -w

# Check logs from the leader
kubectl logs vllm-multi-node-0

# Check logs from worker
kubectl logs vllm-multi-node-0-1

# Port forward to the leader to test
kubectl port-forward vllm-multi-node-0 8000:8000 &

# Test inference
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-405b",
    "prompt": "Explain distributed inference in simple terms:",
    "max_tokens": 200
  }' | jq
```

### 3.5 Troubleshooting Multi-Node Deployments

**Check RDMA connectivity:**
```bash
# Exec into leader pod
kubectl exec -it vllm-multi-node-0 -- bash

# Check RDMA interfaces
ibv_devices

# Test RDMA bandwidth
ib_write_bw
```

**Check NCCL logs:**
```bash
# Look for NCCL using RDMA transport
kubectl logs vllm-multi-node-0 | grep NCCL
# Should see: "NCCL INFO Using network RDMA"
```

---


## Section 5: Performance Considerations

### 5.1 Optimization Tips

**For Single-GPU:**
- Use quantization (INT8, INT4) to fit larger models
- Enable flash attention for longer sequences
- Consider smaller model variants (7B instead of 13B)

**For Multi-GPU Single-Node:**
- Use the maximum tensor parallel size that fits your model
- Enable continuous batching for higher throughput
- Monitor NVLink bandwidth with `nvidia-smi nvlink`
- Consider quantization to fit even larger models

**For Multi-Node:**
- Ensure RDMA is properly configured and tested first
- Use pipeline parallelism for very large models (1T+ parameters)
- Monitor NCCL bandwidth to verify RDMA is being used
- Use flash attention and other optimization techniques
- Start with 2 nodes, scale up as needed

---

## Next Steps

Now that you can deploy both single-node and multi-node inference workloads, Part 3 will cover:

- **Inference Gateway**: Intelligent routing and load balancing
- **Request queuing and prioritization**
- **Multi-model serving**
- **A/B testing and canary deployments**

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LeaderWorkerSet GitHub](https://github.com/kubernetes-sigs/lws)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [GKE Multi-Networking Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/setup-multinetwork-support-for-pods)

---

## Feedback and Contributions

Questions or suggestions? [Open an issue](https://github.com/maci0/gke-inference-from-scratch/issues) or reach out on [Twitter/LinkedIn](#).
