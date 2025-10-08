locals {
  additional_networks = concat(
    [{
      network    = google_compute_network.gvnic_vpc.name
      subnetwork = google_compute_subnetwork.gvnic_sub.name
    }],
    [for sub in google_compute_subnetwork.rdma_sub : {
      network    = google_compute_network.rdma_vpc.name
      subnetwork = sub.name
    }]
  )
}

resource "google_container_node_pool" "gpu_nodes" {
  name           = "${var.name_prefix}-a3u-nodes"
  cluster        = google_container_cluster.primary.id
  location       = var.region
  node_locations = [var.zone]

  initial_node_count = 2
  autoscaling {
    min_node_count  = var.autoscaling_min_nodes
    max_node_count  = var.autoscaling_max_nodes
    location_policy = "ANY"
  }

  management {
    auto_repair  = false
    auto_upgrade = true
  }

  queued_provisioning {
    enabled = false
  }

  node_config {
    machine_type = var.gpu_machine_type

    ephemeral_storage_local_ssd_config {
      local_ssd_count = 32
    }


    # Don't NCCL fast socket, we use RDMA instead
    fast_socket {
      enabled = false
    }

    # Enable gvnic
    gvnic {
      enabled = true
    }

    reservation_affinity {
      consume_reservation_type = var.reservation_name == "" ? "NO_RESERVATION" : "SPECIFIC_RESERVATION"
      key                      = var.reservation_name == "" ? null : "compute.googleapis.com/reservation-name"
      values                   = var.reservation_name == "" ? null : [var.reservation_name]
    }

    advanced_machine_features {
      threads_per_core = 1
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    guest_accelerator {
      type  = var.gpu_accelerator_type
      count = var.gpu_accelerator_count
      gpu_driver_installation_config {
        gpu_driver_version = var.gpu_driver_version
      }
    }
  }

  network_config {
    dynamic "additional_node_network_configs" {
      for_each = local.additional_networks
      content {
        network    = additional_node_network_configs.value.network
        subnetwork = additional_node_network_configs.value.subnetwork
      }
    }
  }
}
