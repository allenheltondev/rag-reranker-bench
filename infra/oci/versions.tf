terraform {
  required_version = ">= 1.5"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0"
    }
  }
}

# Authenticates from ~/.oci/config. Set `oci_profile` if you use a non-default profile.
provider "oci" {
  region              = var.region
  config_file_profile = var.oci_profile
}
