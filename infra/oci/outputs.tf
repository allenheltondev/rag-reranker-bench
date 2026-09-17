output "app_public_ip" {
  value = oci_core_instance.app.public_ip
}

output "ssh" {
  value = "ssh opc@${oci_core_instance.app.public_ip}"
}

output "adb_connection_strings" {
  description = "Full TLS descriptors per service. Put the TP one in ORACLE_CONNECT_STRING: it is the fixed-degree service, so the database does not parallelise a statement differently from one run to the next."
  value       = oci_database_autonomous_database.bench.connection_strings[0].all_connection_strings
}

output "models_bucket" {
  value = oci_objectstorage_bucket.models.name
}

output "models_par_base_url" {
  description = "Prefix for object URLs the database can read. Append the file name. Goes in ORACLE_MODELS_PAR_URL."
  value       = "https://objectstorage.${var.region}.oraclecloud.com${oci_objectstorage_preauthrequest.models_read.access_uri}"
  sensitive   = true
}

output "upload_models" {
  value = "oci os object put --bucket-name ${oci_objectstorage_bucket.models.name} --file models/oracle/bge_reranker_base.onnx && oci os object put --bucket-name ${oci_objectstorage_bucket.models.name} --file models/oracle/all_MiniLM_L12_v2.onnx"
}
