# The application VM. It is deliberately a separate host from the database so that the
# "bytes leaving the database" measurement crosses a real network, not a loopback interface.

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_core_images" "ol9" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Oracle Linux"
  operating_system_version = "9"
  shape                    = var.app_shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_instance" "app" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  display_name        = "${var.name}-app"
  shape               = var.app_shape

  shape_config {
    ocpus         = var.app_ocpus
    memory_in_gbs = var.app_memory_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
    hostname_label   = "app"
    nsg_ids          = [oci_core_network_security_group.app.id]
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ol9.images[0].id
    boot_volume_size_in_gbs = 100
  }

  metadata = {
    ssh_authorized_keys = file(pathexpand(var.ssh_public_key_path))
    user_data           = base64encode(file("${path.module}/cloud-init.yaml"))
  }
}
