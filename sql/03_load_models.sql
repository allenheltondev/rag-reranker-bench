-- Load the ONNX models into the database from a directory object (container).
-- For Autonomous Database use 03_load_models_adb.sql. Run either with:
--
--   npm run models                 both models
--   npm run models -- --only embed     just the embedding model
--   npm run models -- --only rerank    just the cross-encoder
--
-- Each model is one statement, so either can be loaded on its own. The files must already be
-- in the directory (docker-compose mounts ./models/oracle at /opt/oracle/onnx).
--
-- The cross-encoder must be the AUGMENTED export, with the tokenizer inside the graph:
-- PREDICTION() hands the model raw text and the graph has to tokenize it. The application
-- copy from scripts/export-reranker-onnx.sh is NOT that. See sql/README.md.
--
-- Tokens: ${EMBED_MODEL}, ${RERANK_MODEL}, ${ONNX_DIRECTORY}, ${EMBED_FILE}, ${RERANK_FILE},
--         ${RERANK_INPUT}

BEGIN
  BEGIN
    DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${EMBED_MODEL}', force => TRUE);
  EXCEPTION
    WHEN OTHERS THEN NULL;
  END;
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
  BEGIN
    DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${RERANK_MODEL}', force => TRUE);
  EXCEPTION
    WHEN OTHERS THEN NULL;
  END;
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    directory  => '${ONNX_DIRECTORY}',
    file_name  => '${RERANK_FILE}',
    model_name => '${RERANK_MODEL}',
    metadata   => JSON('{
      "function" : "regression",
      "input"    : ${RERANK_INPUT}
    }')
  );
END;
/
