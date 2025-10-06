# Building a Production Ready LLM Inferencing Platform on GKE from Scratch – Part 2: Inference Frameworks

## Introduction

In this multi-part series of blog posts, we will explore how to build a production-ready LLM inference platform on Google Kubernetes Engine (GKE) from the ground up. Whether you're scaling from prototype to production or building new infrastructure, this guide will walk you through every component needed for a robust, performant inference system.

While there are plenty of tools available ( cluster toolkit, xpk, etc. ) to abstract this away. It is beneficial to have an understanding of every layer of the stack.

Throughout this series we will build those layers bottom to top. We will look at how to build them using regular `gcloud` command and how to build the automation using `terraform`.

### Series Overview

This series covers:

1. Base Infrastructure - GKE setup with GPU/TPU support and RDMA networking
2. **Inference Frameworks** - vLLM, TGI, TensorRT-LLM comparison and deployment
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

TBD

