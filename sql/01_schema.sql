-- Schema for the reranker benchmark.
--
-- Placeholders are substituted by src/db/load.ts from .env:
--   ${PREFIX}       ORACLE_SCHEMA_PREFIX  (default BENCH)
--   ${DIMS}         ORACLE_EMBED_DIMS     (default 384, must match the loaded ONNX embedder)
--   ${EMBED_MODEL}  ORACLE_EMBED_MODEL    (default DOC_EMBEDDER)
--
-- Statements are separated by a line containing only "/".

BEGIN
  EXECUTE IMMEDIATE 'DROP TABLE ${PREFIX}_CHUNKS PURGE';
EXCEPTION
  WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF;
END;
/

-- One row per retrievable chunk. Metadata lives beside the text and the vector, which is the
-- entire architectural point being tested: filtering, lexical retrieval, vector retrieval and
-- reranking can all happen without the candidate text leaving this table.
CREATE TABLE ${PREFIX}_CHUNKS (
  ID          VARCHAR2(64)   NOT NULL,
  DOC_ID      VARCHAR2(64)   NOT NULL,
  TITLE       VARCHAR2(400)  NOT NULL,
  CONTENT     CLOB           NOT NULL,
  TENANT      VARCHAR2(64)   NOT NULL,
  OWNER_ID    VARCHAR2(64),
  CREATED_AT  DATE           NOT NULL,
  EXPIRES_AT  DATE,
  TAGS        VARCHAR2(400),
  EMBEDDING   VECTOR(${DIMS}, FLOAT32),
  CONSTRAINT ${PREFIX}_CHUNKS_PK PRIMARY KEY (ID)
)
/

-- Metadata filters run before candidate generation, so they need to be cheap.
CREATE INDEX ${PREFIX}_CHUNKS_SCOPE_IX ON ${PREFIX}_CHUNKS (TENANT, EXPIRES_AT)
/

-- Lexical retrieval. SYNC (ON COMMIT) keeps the index current during the load; a production
-- system would usually sync on a schedule instead.
CREATE INDEX ${PREFIX}_CHUNKS_TEXT_IX ON ${PREFIX}_CHUNKS (CONTENT)
  INDEXTYPE IS CTXSYS.CONTEXT
  PARAMETERS ('SYNC (ON COMMIT)')
/
