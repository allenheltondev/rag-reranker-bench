-- Create the benchmark schema user. Run this ONCE as ADMIN (Autonomous Database) or SYS
-- (Oracle Free container). Everything else in this repo runs as this user through `npm run`.
--
--   Autonomous:  sql admin@<TP connect string> @sql/00_user.sql
--   Container:   docker compose cp sql/00_user.sql oracle:/tmp/00_user.sql
--                docker compose exec oracle bash -c "sed -i 's/\r$//' /tmp/00_user.sql && sqlplus -s 'sys/<pw>@localhost:1521/FREEPDB1 as sysdba' @/tmp/00_user.sql"
--   (the sed strips carriage returns in case the file was checked out with CRLF on Windows)
--
-- On the container, connect to the FREEPDB1 service as shown, not "/ as sysdba": that lands in
-- the root container, where a plain user name is rejected.

SET VERIFY OFF
WHENEVER SQLERROR CONTINUE

DEFINE bench_user     = BENCH
DEFINE bench_password = ChangeMe_1
-- Autonomous Database: DATA. Oracle Free container: USERS.
DEFINE tablespace     = USERS

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

-- Autonomous Database only: pulling the ONNX files from Object Storage. Skipped silently where
-- the package does not exist (the Free container).
BEGIN
  EXECUTE IMMEDIATE 'GRANT EXECUTE ON DBMS_CLOUD TO &bench_user';
EXCEPTION
  WHEN OTHERS THEN IF SQLCODE NOT IN (-4042, -942, -1917) THEN RAISE; END IF;
END;
/

-- Oracle Free container only: the directory object the model loader reads from. It maps to the
-- ./models/oracle mount in docker-compose.yml. Comment these two out on Autonomous.
CREATE OR REPLACE DIRECTORY ONNX_DIR AS '/opt/oracle/onnx';
GRANT READ ON DIRECTORY ONNX_DIR TO &bench_user;
