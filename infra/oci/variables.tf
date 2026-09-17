variable "region" {
  description = "OCI region, e.g. us-ashburn-1. Both the database and the VM land here."
  type        = string
}

variable "oci_profile" {
  description = "Profile name in ~/.oci/config."
  type        = string
  default     = "DEFAULT"
}

variable "tenancy_ocid" {
  type = string
}

variable "compartment_ocid" {
  description = "Compartment everything is created in."
  type        = string
}

variable "name" {
  description = "Prefix for every resource's display name."
  type        = string
  default     = "reranker-bench"
}

# --- Database ---------------------------------------------------------------------------

variable "db_name" {
  description = "Autonomous Database name: letters and digits only, max 14 characters."
  type        = string
  default     = "rerankbench"
}

variable "db_version" {
  description = <<-EOT
    Database version to provision. The benchmark needs a release that can load an ONNX
    reranking model. If provisioning rejects this value, list what your region offers:
      oci db autonomous-db-version list --compartment-id <ocid> --db-workload OLTP
  EOT
  type        = string
  default     = "26ai"
}

variable "db_ecpus" {
  description = "ECPU count. Fixed for the life of the benchmark: autoscaling is disabled below so the number does not move mid-run."
  type        = number
  default     = 4
}

variable "db_storage_gb" {
  type    = number
  default = 128
}

variable "db_admin_password" {
  description = "ADMIN password. 12-30 chars, upper, lower, digit, no double quotes. Set via TF_VAR_db_admin_password."
  type        = string
  sensitive   = true
}

variable "db_free_tier" {
  description = "Provision an Always Free ADB instead. Fine for a smoke test; not what you want under a benchmark, and ONNX model loading may be constrained on it."
  type        = bool
  default     = false
}

# --- Application VM -------------------------------------------------------------------

variable "app_shape" {
  type    = string
  default = "VM.Standard.E5.Flex"
}

variable "app_ocpus" {
  description = "OCPUs for the VM that runs the application-side reranker. This number is recorded in every report; keep it fixed across runs."
  type        = number
  default     = 4
}

variable "app_memory_gb" {
  type    = number
  default = 32
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ssh_cidr" {
  description = "Where SSH to the VM is allowed from. Your IP with /32, not 0.0.0.0/0."
  type        = string
}

# --- Model storage ----------------------------------------------------------------------

variable "par_expires" {
  description = "Expiry (RFC3339) for the pre-authenticated read URL on the models bucket. The database loads ONNX files through it."
  type        = string
}
