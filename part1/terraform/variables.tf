variable "project_id" {
  description = "The Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "The Google Cloud region for the resources."
  type        = string
}

variable "zone" {
  description = "The zone for the GPU node pool."
  type        = string
}

variable "name_prefix" {
  description = "A common prefix for all created resources (e.g., 'mwy-llm-d')."
  type        = string
}

variable "gpu_machine_type" {
  description = "The machine type for the GPU node pool."
  type        = string
  default     = "a3-ultragpu-8g"
}

variable "gpu_accelerator_type" {
  description = "The type of GPU accelerator."
  type        = string
  default     = "nvidia-h200-141gb"
}

variable "gpu_accelerator_count" {
  description = "The number of GPUs per node."
  type        = number
  default     = 8
}

variable "autoscaling_min_nodes" {
  description = "The minimum number of nodes in the GPU node pool."
  type        = number
  default     = 0
}

variable "autoscaling_max_nodes" {
  description = "The maximum number of nodes in the GPU node pool."
  type        = number
  default     = 4
}

variable "reservation_name" {
  description = "The name of the specific reservation to use for the GPU node pool."
  type        = string
  default     = ""
}

variable "gke_release_channel" {
  description = "The GKE release channel to use for the cluster."
  type        = string
  default     = "STABLE"
}

variable "gke_version" {
  description = "The GKE version to use for the cluster. If not specified, the default version for the release channel will be used."
  type        = string
  default     = null
}

variable "gpu_driver_version" {
  description = "The GPU driver version to use for the GPU node pool."
  type        = string
  default     = "LATEST"
}

variable "create_reservation" {
  description = "Flag to determine if a reservation should be created"
  type        = bool
  default     = false
}

variable "reservation_instance_count" {
  description = "The number of instances to reserve"
  type        = number
  default     = 1
}
