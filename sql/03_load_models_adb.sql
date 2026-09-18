-- Load the ONNX models into Autonomous Database from Object Storage.
--
-- Autonomous has no filesystem directory you can copy into, so the files are read over the
-- pre-authenticated request URL that infra/oci creates for the models bucket:
--
--   ORACLE_TARGET=adb npm run models [-- --only embed|rerank]
--
-- VERIFY BEFORE RELYING ON IT: written without a live Autonomous instance to check against.
-- If LOAD_ONNX_MODEL_CLOUD is not in your release, the fallback is DBMS_CLOUD.GET_OBJECT into
-- DATA_PUMP_DIR followed by DBMS_VECTOR.LOAD_ONNX_MODEL with directory => 'DATA_PUMP_DIR',
-- exactly as 03_load_models.sql does.
--
-- Tokens: ${EMBED_MODEL}, ${RERANK_MODEL}, ${PAR_BASE_URL}, ${EMBED_FILE}, ${RERANK_FILE},
--         ${RERANK_INPUT}

BEGIN
  BEGIN
    DBMS_VECTOR.DROP_ONNX_MODEL(model_name => '${EMBED_MODEL}', force => TRUE);
  EXCEPTION
    WHEN OTHERS THEN NULL;
  END;
  DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD(
    model_name => '${EMBED_MODEL}',
    credential => NULL,
    uri        => '${PAR_BASE_URL}${EMBED_FILE}',
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
  DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD(
    model_name => '${RERANK_MODEL}',
    credential => NULL,
    uri        => '${PAR_BASE_URL}${RERANK_FILE}',
    metadata   => JSON('{
      "function" : "regression",
      "input"    : ${RERANK_INPUT}
    }')
  );
END;
/
