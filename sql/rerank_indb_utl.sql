-- ALTERNATIVE in-database reranking path: DBMS_VECTOR.UTL_TO_RERANK.
--
-- VERIFY THIS AGAINST YOUR DATABASE VERSION BEFORE QUOTING ANY NUMBER FROM IT. The
-- UTL_TO_RERANK signature, its params JSON, and the shape of its JSON result have moved
-- between releases, and this template was written without a live instance to check against.
-- `npm run doctor` exercises it and will tell you whether it works as written here.
--
-- Why it is interesting even so: the same call can route to an in-database ONNX model, or to
-- Cohere / Vertex AI / OCI Generative AI, by changing only the provider in the params. That is
-- the "same pipeline, different inference location" continuum, expressed as one API.
--
-- Tokens: ${RERANK_MODEL}
-- Binds:  :qtext :docs (JSON array of candidate text) :topk

SELECT
  jt.IDX,
  jt.SCORE
FROM JSON_TABLE(
  DBMS_VECTOR.UTL_TO_RERANK(
    :qtext,
    :docs,
    JSON('{ "provider": "database", "model": "${RERANK_MODEL}" }')
  ),
  '$[*]'
  COLUMNS (
    IDX   NUMBER PATH '$.index',
    SCORE NUMBER PATH '$.score'
  )
) jt
ORDER BY jt.SCORE DESC
FETCH FIRST :topk ROWS ONLY
