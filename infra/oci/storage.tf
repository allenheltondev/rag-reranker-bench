# A bucket for the ONNX files, and a read-only pre-authenticated URL so the database can pull
# them without an IAM credential. The URL is what sql/03_load_models_adb.sql reads from.

data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_ocid
}

resource "oci_objectstorage_bucket" "models" {
  compartment_id = var.compartment_ocid
  namespace      = data.oci_objectstorage_namespace.ns.namespace
  name           = "${var.name}-models"
  access_type    = "NoPublicAccess"
}

resource "oci_objectstorage_preauthrequest" "models_read" {
  namespace    = data.oci_objectstorage_namespace.ns.namespace
  bucket       = oci_objectstorage_bucket.models.name
  name         = "${var.name}-models-read"
  access_type  = "AnyObjectRead"
  time_expires = var.par_expires
}
