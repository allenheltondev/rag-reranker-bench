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

### 1. Provision

```bash
cd infra/oci
cp terraform.tfvars.example terraform.tfvars      # fill in
export TF_VAR_db_admin_password='...'
terraform init && terraform apply
```

### 2. Build both models once, on your machine

Both arms must run the same weights, so the models are built in one place and distributed,
rather than exported separately on each host.

```bash
npm run export:app-model        # the application copy
npm run augment:rerank-model    # the database copy, built FROM that export
```

`augment:rerank-model` prints the `ORACLE_INDB_SCORE_EXPR` for the model it produced. Keep it;
step 4 needs it verbatim.

### 3. Distribute

```bash
# The database reads its model from Object Storage.
terraform -chdir=infra/oci output -raw upload_models | bash

# The application arm gets the exact files, not a second export of the same checkpoint.
APP_IP=$(terraform -chdir=infra/oci output -raw app_public_ip)
scp -r models/bge-reranker-base opc@$APP_IP:~/rag-reranker-bench/models/
```

The embedding model does not need uploading if you point the loader at Oracle's own published
URL; otherwise put `all_MiniLM_L12_v2.onnx` in `models/oracle/` before the upload above.

### 4. Configure the VM

```bash
ssh opc@$APP_IP
cd rag-reranker-bench && npm install && cp .env.example .env
```

In `.env` on the VM:

```
ORACLE_USER=BENCH
ORACLE_PASSWORD=<choose one; bootstrap creates the user with it>
ORACLE_CONNECT_STRING=<the TP entry from `terraform output adb_connection_strings`>

# Autonomous has no customer SYSDBA, so bootstrap connects as ADMIN without a privilege flag.
ORACLE_SYS_USER=ADMIN
ORACLE_SYS_PASSWORD=<TF_VAR_db_admin_password>

# Models come from Object Storage rather than a directory object.
ORACLE_TARGET=adb
ORACLE_MODELS_PAR_URL=<`terraform output -raw models_par_base_url`>

# From step 2, verbatim. Set it ONCE - a second assignment of this key silently wins.
ORACLE_INDB_SCORE_EXPR=PREDICTION(BGE_RERANKER USING :qtext || '</s></s> ' || TITLE || '. ' || CONTENT AS DATA)
```

### 5. Load and check

```bash
npm run bootstrap && npm run corpus && npm run models && npm run load && npm run doctor
```

Order matters: the embedding model must exist before `load`, because rows are embedded on
insert.

### 6. Match the compute, then run

Read doctor's `database CPUs` line. It prints Autonomous's `cpu_count`, the VM's cores, and
the application arm's thread count. **Set `APP_RERANK_THREADS` to the database's `cpu_count`
and re-run doctor until it says `matched`.**

This is not a nicety. On the local Free container `cpu_count` is capped at 2 while the
application had 20 cores, and the resulting 9-13x latency ratio was mostly that imbalance.
A ratio measured without matching is a comparison of how much CPU each side was given.

```bash
npm run bench -- --candidates 10,40 --iterations 10 --repeats 3
npm run bench -- --retrievals hybrid-rrf --candidates 10,20,40,80 --iterations 5
```

Budget from the smoke test rather than from these commands: in-database scoring cost roughly
1.5 s per candidate at 2 CPUs, and scales with both candidate depth and inverse ECPU count.
Run `npm run bench -- --queries 2 --candidates 10 --iterations 2 --warmup 1` first and
multiply.

## Which service to connect to

`adb_connection_strings` lists five. Use **TP**: it runs each statement at a fixed degree of
parallelism, so the database cannot parallelise the scoring step differently from one run to
the next. `HIGH` will parallelise `PREDICTION()` across rows, which is a genuinely interesting
number — the database doing something the application path cannot — but it is a separate
experiment. Do not mix the two in one report.

## Sizing

Two numbers decide whether this works at all, and both are worth setting deliberately rather
than accepting:

**ECPUs.** They set the database's `cpu_count`, which is what the application arm has to match
for the comparison to mean anything, and they govern how long the in-database arm takes. Four
is the default here; more makes the run shorter and the comparison no less fair, as long as
the VM's threads are matched to it.

**Memory for the model.** A cross-encoder is held in memory to load and again per session to
score - locally that meant 1.1 GB of shared memory and a 4 GB PGA target. Autonomous manages
this itself, but the allowance scales with ECPUs, so too small an instance can fail to load
the model at all. If `npm run models:rerank` fails on memory, raise `db_ecpus` before reaching
for `--quantize`; quantizing means rebuilding both arms' weights to keep them identical.

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
