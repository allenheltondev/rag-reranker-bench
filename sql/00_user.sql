-- Create the benchmark schema user. Run this ONCE as ADMIN (Autonomous Database) or SYS
-- (Oracle Free container), from SQL*Plus or SQLcl:
--
--   sql admin@<connect>  @sql/00_user.sql
--
-- Everything else in this repo runs as this user through `npm run ...`.

DEFINE bench_user     = BENCH
DEFINE bench_password = ChangeMe_1
-- Autonomous Database: DATA. Oracle Free container: USERS.
DEFINE tablespace     = DATA

CREATE USER &bench_user IDENTIFIED BY "&bench_password"
  QUOTA UNLIMITED ON &tablespace;

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW, CREATE PROCEDURE TO &bench_user;

-- Loading an ONNX model creates a mining model object in the user's schema.
GRANT CREATE MINING MODEL TO &bench_user;

-- Oracle Text, for the lexical arm.
GRANT CTXAPP TO &bench_user;
GRANT EXECUTE ON CTXSYS.CTX_DDL TO &bench_user;

-- Vector utilities: embedding, model loading, reranking.
GRANT EXECUTE ON SYS.DBMS_VECTOR TO &bench_user;
GRANT EXECUTE ON SYS.DBMS_VECTOR_CHAIN TO &bench_user;

-- Autonomous Database only: pulling the ONNX files from Object Storage. Harmless elsewhere
-- if the package exists; comment out on a container database that lacks it.
GRANT EXECUTE ON DBMS_CLOUD TO &bench_user;

-- Oracle Free container only: the directory object the model loader reads from.
-- CREATE OR REPLACE DIRECTORY ONNX_DIR AS '/opt/oracle/onnx';
-- GRANT READ ON DIRECTORY ONNX_DIR TO &bench_user;
