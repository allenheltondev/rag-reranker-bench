# Autonomous Database on a private endpoint, reachable only from the application VM.
#
# Two settings here are benchmark decisions, not defaults:
#   - autoscaling is off, so the ECPU count cannot change between one iteration and the next;
#   - mTLS is off, so the application connects with a plain TLS descriptor and no wallet, which
#     keeps the connection setup out of the comparison and lets node-oracledb run in thin mode.

resource "oci_database_autonomous_database" "bench" {
  compartment_id = var.compartment_ocid
  db_name        = var.db_name
  display_name   = "${var.name}-adb"
  db_workload    = "OLTP"
  db_version     = var.db_version
  license_model  = "LICENSE_INCLUDED"

  compute_model                       = "ECPU"
  compute_count                       = var.db_ecpus
  data_storage_size_in_gb             = var.db_storage_gb
  is_auto_scaling_enabled             = false
  is_auto_scaling_for_storage_enabled = false
  is_free_tier                        = var.db_free_tier

  admin_password = var.db_admin_password

  subnet_id                   = oci_core_subnet.public.id
  nsg_ids                     = [oci_core_network_security_group.db.id]
  is_mtls_connection_required = false

  lifecycle {
    ignore_changes = [admin_password]
  }
}
