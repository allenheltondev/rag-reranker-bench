-- Create the benchmark schema user and its grants.
--
-- Run with `npm run bootstrap`, which connects with the elevated credentials from .env
-- (SYSDBA against the container, ADMIN against Autonomous) and substitutes the placeholders.
-- It is the one script that does not run as the benchmark user, because it creates them.
--
-- Tokens: ${BENCH_USER}, ${BENCH_PASSWORD}, ${TABLESPACE}

DECLARE
  user_exists NUMBER;
BEGIN
  SELECT COUNT(*) INTO user_exists FROM ALL_USERS WHERE USERNAME = '${BENCH_USER}';
  IF user_exists = 0 THEN
    EXECUTE IMMEDIATE 'CREATE USER ${BENCH_USER} IDENTIFIED BY "${BENCH_PASSWORD}"';
  ELSE
    -- Re-running must be harmless: reset the password to what .env says rather than failing.
    EXECUTE IMMEDIATE 'ALTER USER ${BENCH_USER} IDENTIFIED BY "${BENCH_PASSWORD}"';
  END IF;
  EXECUTE IMMEDIATE 'ALTER USER ${BENCH_USER} QUOTA UNLIMITED ON ${TABLESPACE}';
END;
/

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW, CREATE PROCEDURE TO ${BENCH_USER}
/

-- Loading an ONNX model creates a mining model object in the user's schema.
GRANT CREATE MINING MODEL TO ${BENCH_USER}
/

-- Oracle Text, for the lexical arm.
GRANT CTXAPP TO ${BENCH_USER}
/

GRANT EXECUTE ON CTXSYS.CTX_DDL TO ${BENCH_USER}
/

GRANT EXECUTE ON SYS.DBMS_VECTOR TO ${BENCH_USER}
/

BEGIN
  -- Present on some releases only; the benchmark does not require it.
  EXECUTE IMMEDIATE 'GRANT EXECUTE ON SYS.DBMS_VECTOR_CHAIN TO ${BENCH_USER}';
EXCEPTION
  WHEN OTHERS THEN IF SQLCODE NOT IN (-4042, -942) THEN RAISE; END IF;
END;
/

BEGIN
  -- So the benchmark can record the database's own CPU count, which is what governs how much
  -- of the machine in-database scoring can use. Reported in every run for provenance.
  EXECUTE IMMEDIATE 'GRANT SELECT ON V_$PARAMETER TO ${BENCH_USER}';
EXCEPTION
  WHEN OTHERS THEN NULL;
END;
/

BEGIN
  -- Autonomous Database only: reading the ONNX files from Object Storage.
  EXECUTE IMMEDIATE 'GRANT EXECUTE ON DBMS_CLOUD TO ${BENCH_USER}';
EXCEPTION
  WHEN OTHERS THEN IF SQLCODE NOT IN (-4042, -942, -1917) THEN RAISE; END IF;
END;
/
