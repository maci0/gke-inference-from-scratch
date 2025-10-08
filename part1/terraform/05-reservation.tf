resource "google_compute_reservation" "default" {
  count = var.create_reservation ? 1 : 0

  name = var.reservation_name
  zone = var.zone

  specific_reservation_required = true

  specific_reservation {
    count = var.reservation_instance_count
    instance_properties {
      machine_type = var.gpu_machine_type
      guest_accelerators {
        accelerator_count = var.gpu_accelerator_count
        accelerator_type  = var.gpu_accelerator_type
      }
    }
  }
}
