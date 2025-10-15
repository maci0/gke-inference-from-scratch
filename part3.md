# Building a Production Ready LLM Inferencing Platform on GKE from Scratch – Part 3: Inference Gateway

## Introduction

You've built an LLM application that works great in development. Now you need to serve it at scale. Thousands of concurrent requests with sub-second latency. Your prototype ran on a single GPU, but your production model won't fit on one machine. Time for some real infrastructure work.

This series walks through building a production LLM inference platform on GKE from scratch. We'll cover everything you need: networking, GPU clusters, inference frameworks, routing, and optimization.

Tools like `cluster toolkit` and `xpk` can automate much of this, but understanding what's happening under the hood matters. When your cluster starts acting weird at 3 AM, you'll actually know how to debug it.

We'll start with `gcloud` commands to see what's happening at each step, then automate it with Terraform for production use. With each step we will also explore and explain some of the underlying technologies that make everything work.

### Series Overview

This series covers:

1. [Base Infrastructure](part1.md) - GKE setup with GPU support and RDMA networking
2. [Inference Pattern](part2.md) - vLLM, single/multi GPU and multi node deployments
3. **[Inference Gateway](part3.md)** - Intelligent routing
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

## Part 3: Inference Gateway

In [Part 1](part1.md), we built the foundation with GKE, GPU nodes, and RDMA networking. In [Part 2](part2.md), we deployed vLLM inference workloads. Now it's time to add intelligent routing and load balancing to create a production-ready gateway.

### Prerequisites

Before starting this part, ensure you have:

1. **A GKE cluster with GPU support**: Complete [Part 1](part1.md) or use an existing cluster
2. **Inference workloads deployed**: Complete [Part 2](part2.md) to deploy vLLM

### What We'll Cover

In this part, we'll explore:

- **Gateway API for Kubernetes**
- **Intelligent routing and load balancing**
- **Request queuing and prioritization**
- **Multi-model serving**
- **A/B testing and canary deployments**
- **Monitoring and observability**

---

## Coming Soon

This section is under development. Check back soon for detailed content on building an intelligent inference gateway.

---

## Resources

- [GKE Gateway API Documentation](https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api)
- [Kubernetes Gateway API](https://gateway-api.sigs.k8s.io/)

---

## Feedback and Contributions

Questions or suggestions? [Open an issue](https://github.com/maci0/gke-inference-from-scratch/issues) or reach out on [Twitter/LinkedIn](#).

---