# One VCN, one public subnet. The database's private endpoint and the VM share the subnet,
# separated by network security groups: the VM may reach the database on 1522, nothing else
# may, and SSH to the VM is limited to one CIDR.

resource "oci_core_vcn" "bench" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.42.0.0/16"]
  display_name   = "${var.name}-vcn"
  dns_label      = "bench"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.bench.id
  display_name   = "${var.name}-igw"
  enabled        = true
}

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.bench.id
  display_name   = "${var.name}-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

# The VCN's default security list allows SSH from anywhere. This one allows nothing in, so
# the NSGs below are the only ingress path.
resource "oci_core_security_list" "locked" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.bench.id
  display_name   = "${var.name}-sl"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }
}

resource "oci_core_subnet" "public" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.bench.id
  cidr_block                 = "10.42.1.0/24"
  display_name               = "${var.name}-subnet"
  dns_label                  = "pub"
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.locked.id]
  prohibit_public_ip_on_vnic = false
}

resource "oci_core_network_security_group" "app" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.bench.id
  display_name   = "${var.name}-nsg-app"
}

resource "oci_core_network_security_group" "db" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.bench.id
  display_name   = "${var.name}-nsg-db"
}

resource "oci_core_network_security_group_security_rule" "ssh_in" {
  network_security_group_id = oci_core_network_security_group.app.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = var.allowed_ssh_cidr
  source_type               = "CIDR_BLOCK"

  tcp_options {
    destination_port_range {
      min = 22
      max = 22
    }
  }
}

resource "oci_core_network_security_group_security_rule" "app_out" {
  network_security_group_id = oci_core_network_security_group.app.id
  direction                 = "EGRESS"
  protocol                  = "all"
  destination               = "0.0.0.0/0"
  destination_type          = "CIDR_BLOCK"
}

# Only the application VM may open a database connection.
resource "oci_core_network_security_group_security_rule" "db_from_app" {
  network_security_group_id = oci_core_network_security_group.db.id
  direction                 = "INGRESS"
  protocol                  = "6"
  source                    = oci_core_network_security_group.app.id
  source_type               = "NETWORK_SECURITY_GROUP"

  tcp_options {
    destination_port_range {
      min = 1522
      max = 1522
    }
  }
}

resource "oci_core_network_security_group_security_rule" "db_out" {
  network_security_group_id = oci_core_network_security_group.db.id
  direction                 = "EGRESS"
  protocol                  = "all"
  destination               = "0.0.0.0/0"
  destination_type          = "CIDR_BLOCK"
}
