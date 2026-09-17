-- Load the ONNX models into the database.
--
-- Both files must already be in a directory object the database can read. The usual setup is:
--
--   CREATE OR REPLACE DIRECTORY ONNX_DIR AS '/opt/oracle/onnx';
--   GRANT READ ON DIRECTORY ONNX_DIR TO <bench_user>;
--
-- and then copy the exported models there. scripts/export-reranker-onnx.sh produces the
-- application-side copy of the reranker; the in-database copy must be the augmented export
-- (tokenizer embedded in the graph) that DBMS_VECTOR.LOAD_ONNX_MODEL expects. See sql/README.md.
--
-- Tokens: ${EMBED_MODEL}, ${RERANK_MODEL}

-- Embedding model, used by VECTOR_EMBEDDING() during load and for vector retrieval.
BEGIN
  DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${EMBED_MODEL}', force => TRUE);
EXCEPTION
  WHEN OTHERS THEN NULL;
END;
/

BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    directory  => 'ONNX_DIR',
    file_name  => 'all_MiniLM_L12_v2.onnx',
    model_name => '${EMBED_MODEL}',
    metadata   => JSON('{
      "function"       : "embedding",
      "embeddingOutput": "embedding",
      "input"          : { "input": ["DATA"] }
    }')
  );
END;
/

-- Cross-encoder, invoked from SQL once it is a database object.
BEGIN
  DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${RERANK_MODEL}', force => TRUE);
EXCEPTION
  WHEN OTHERS THEN NULL;
END;
/

BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    directory  => 'ONNX_DIR',
    file_name  => 'bge_reranker_base.onnx',
    model_name => '${RERANK_MODEL}',
    metadata   => JSON('{
      "function" : "regression",
      "input"    : { "input": ["FIRST_INPUT", "SECOND_INPUT"] }
    }')
  );
END;
/
