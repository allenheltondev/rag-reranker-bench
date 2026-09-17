# Running the benchmark on Oracle Cloud

What this creates, in one compartment:

| Resource | Purpose |
|---|---|
| VCN, subnet, gateway | A network the two hosts share. Nothing else is reachable from outside. |
| Autonomous Database (OLTP, ECPU) | The in-database arm. Private endpoint, mTLS off, **autoscaling off**. |
| Compute VM (Oracle Linux 9) | The application arm. Separate host, so the transfer measurement crosses a real network. |
| Object Storage bucket + read PAR | Where the ONNX files go so the database can load them. |
| Two NSGs | VM ← SSH from your IP only; database ← port 1522 from the VM only. |

Two of those settings are benchmark decisions. Autoscaling is disabled so the ECPU count
cannot change between iterations. mTLS is disabled so the application connects with a plain
TLS descriptor in thin mode — no wallet, no Instant Client, and connection setup stays out of
the comparison.

## Prerequisites

- Terraform ≥ 1.5 and the OCI CLI, authenticated (`~/.oci/config`).
- An SSH key pair.
- The two ONNX files, exported locally: the augmented reranker for the database and the
  embedding model (see `sql/README.md`, "Two exports of the same model").

## Steps

```bash
cd infra/oci
cp terraform.tfvars.example terraform.tfvars      # fill in
export TF_VAR_db_admin_password='...'
terraform init && terraform apply
```

Then, from the repo root:

```bash
# 1. Upload the models. terraform output prints the exact command.
terraform -chdir=infra/oci output -raw upload_models | bash

# 2. Create the benchmark user, once, as ADMIN. Edit the DEFINEs at the top first.
sql admin@"$(terraform -chdir=infra/oci output -json adb_connection_strings | jq -r .TP)" @sql/00_user.sql

# 3. On the VM (cloud-init has cloned the repo and installed Node):
ssh opc@$(terraform -chdir=infra/oci output -raw app_public_ip)
cd rag-reranker-bench && npm install && cp .env.example .env
```

In `.env` on the VM:

```
ORACLE_USER=BENCH
ORACLE_PASSWORD=<what you set in 00_user.sql>
ORACLE_CONNECT_STRING=<the TP entry from `terraform output adb_connection_strings`>
ORACLE_TARGET=adb
ORACLE_MODELS_PAR_URL=<`terraform output -raw models_par_base_url`>
```

Then the normal sequence, with the application model exported on the VM so both arms run the
same checkpoint:

```bash
scripts/export-reranker-onnx.sh
npm run corpus && npm run load && npm run models && npm run doctor
npm run bench -- --repeats 3
```

## Which service to connect to

`adb_connection_strings` lists five. Use **TP**: it runs each statement at a fixed degree of
parallelism, so the database cannot parallelise the scoring step differently from one run to
the next. `HIGH` will parallelise `PREDICTION()` across rows, which is a genuinely interesting
number — the database doing something the application path cannot — but it is a separate
experiment. Do not mix the two in one report.

## Things to verify on first apply

This module was written without an OCI tenancy to apply it against. Three spots are the most
likely to need a version-specific adjustment:

- `db_version = "26ai"` — if provisioning rejects it, list the accepted values with
  `oci db autonomous-db-version list --compartment-id <ocid> --db-workload OLTP`.
- `data_storage_size_in_gb` — some provider versions want `data_storage_size_in_tbs` instead.
- `sql/03_load_models_adb.sql` uses `DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD` with a PAR URL and no
  credential. The comment in that file gives the fallback path if that call is not available.

`terraform validate` will catch the first two before anything is created.

## Cost

An ECPU ADB and a 4-OCPU flex VM are not free. `terraform destroy` when the run is done; the
results are in `results/` and the report records the shapes used.
