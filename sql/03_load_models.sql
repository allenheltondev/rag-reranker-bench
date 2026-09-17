-- Load the ONNX models into the database from a directory object (Oracle Free container).
-- For Autonomous Database use 03_load_models_adb.sql. Run either with:
--
--   npm run models
--
-- which substitutes the placeholders from .env and runs as the benchmark user. The files must
-- already be in the directory (docker-compose mounts ./models/oracle at /opt/oracle/onnx).
-- The in-database reranker must be the augmented export (tokenizer embedded in the graph);
-- the application copy from scripts/export-reranker-onnx.sh is not that. See sql/README.md.
--
-- Tokens: ${EMBED_MODEL}, ${RERANK_MODEL}, ${ONNX_DIRECTORY}, ${EMBED_FILE}, ${RERANK_FILE}

BEGIN
  DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${EMBED_MODEL}', force => TRUE);
EXCEPTION
  WHEN OTHERS THEN NULL;
END;
/

BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    directory  => '${ONNX_DIRECTORY}',
    file_name  => '${EMBED_FILE}',
    model_name => '${EMBED_MODEL}',
    metadata   => JSON('{
      "function"       : "embedding",
      "embeddingOutput": "embedding",
      "input"          : { "input": ["DATA"] }
    }')
  );
END;
/

BEGIN
  DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${RERANK_MODEL}', force => TRUE);
EXCEPTION
  WHEN OTHERS THEN NULL;
END;
/

BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    directory  => '${ONNX_DIRECTORY}',
    file_name  => '${RERANK_FILE}',
    model_name => '${RERANK_MODEL}',
    metadata   => JSON('{
      "function" : "regression",
      "input"    : { "input": ["FIRST_INPUT", "SECOND_INPUT"] }
    }')
  );
END;
/
