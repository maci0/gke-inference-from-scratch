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
  initial_node_count       = 3

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
