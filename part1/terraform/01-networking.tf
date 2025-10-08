# --- gVNIC Network ---
resource "google_compute_network" "gvnic_vpc" {
  name                    = "${var.name_prefix}-gvnic-net"
  auto_create_subnetworks = false

  provisioner "local-exec" {
    when    = destroy
    command = <<EOT
      FW_RULES=$(gcloud compute firewall-rules list --project=${self.project} --filter="network='${self.self_link}'" --format="value(name)")
      for rule in $FW_RULES; do
        gcloud compute firewall-rules delete "$rule" --project=${self.project} -q
      done
    EOT
  }
}

resource "google_compute_subnetwork" "gvnic_sub" {
  name          = "${var.name_prefix}-gvnic-sub"
  ip_cidr_range = "192.168.0.0/24"
  network       = google_compute_network.gvnic_vpc.id
  region        = var.region
}

resource "google_compute_firewall" "gvnic_internal" {
  name    = "${var.name_prefix}-gvnic-internal"
  network = google_compute_network.gvnic_vpc.name
  allow {
    protocol = "all"
  }
  source_ranges = ["192.168.0.0/16"]
}

resource "google_compute_subnetwork" "proxy_only_subnet" {
  name          = "${var.name_prefix}-proxy-only-sub"
  ip_cidr_range = "192.168.255.0/24"
  network       = google_compute_network.gvnic_vpc.id
  region        = var.region
  purpose       = "REGIONAL_MANAGED_PROXY"
  role          = "ACTIVE"
}

# --- RDMA High-Performance Network ---
# https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet
resource "google_compute_network" "rdma_vpc" {
  name                    = "${var.name_prefix}-rdma-net"
  auto_create_subnetworks = false
  network_profile         = "projects/${var.project_id}/global/networkProfiles/${var.zone}-vpc-roce"
  mtu                     = 8896

  provisioner "local-exec" {
    when    = destroy
    command = <<EOT
      FW_RULES=$(gcloud compute firewall-rules list --project=${self.project} --filter="network='${self.self_link}'" --format="value(name)")
      for rule in $FW_RULES; do
        gcloud compute firewall-rules delete "$rule" --project=${self.project} -q
      done
    EOT
  }
}

resource "google_compute_subnetwork" "rdma_sub" {
  count         = 8 # Creates 8 subnets, from 0 to 7
  name          = "${var.name_prefix}-rdma-sub-${count.index}"
  ip_cidr_range = "192.168.${count.index + 1}.0/24" # Starts from 192.168.1.0/24
  network       = google_compute_network.rdma_vpc.id
  region        = var.region
}
