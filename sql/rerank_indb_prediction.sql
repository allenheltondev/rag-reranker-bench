-- In-database reranking: the whole pipeline, including cross-encoder scoring, in one statement.
--
-- The projection is deliberately narrow. Only IDs and scores cross the network; the candidate
-- text is read, tokenized and scored inside the database and never leaves it. That is the
-- architectural claim this benchmark measures the cost of.
--
-- Tokens: ${PREFIX}, ${EMBED_MODEL}, ${QUERY_EMBED_INPUT}, ${SCORE_EXPR}, ${VEC_SOURCE},
--         ${LEX_SOURCE}, ${FUSED_BODY}
-- Binds:  :qtext :contains :tenant :owner :pool :n :rrfk :topk
--
-- ${SCORE_EXPR} defaults to PREDICTION(...). If your exported cross-encoder loads as a
-- classification model rather than a regression, switch it to PREDICTION_PROBABILITY via
-- ORACLE_INDB_SCORE_EXPR; ordering by a predicted class value ranks nothing.

WITH qv AS (
  SELECT VECTOR_EMBEDDING(${EMBED_MODEL} USING ${QUERY_EMBED_INPUT} AS DATA) AS V FROM DUAL
),
vec AS (
  SELECT ID, ROWNUM AS RNK FROM (
${VEC_SOURCE}
  )
),
lex AS (
  SELECT ID, ROWNUM AS RNK FROM (
${LEX_SOURCE}
  )
),
fused AS (
${FUSED_BODY}
),
cand AS (
  SELECT c.ID, c.TITLE, c.CONTENT
  FROM fused f
  JOIN ${PREFIX}_CHUNKS c ON c.ID = f.ID
  ORDER BY f.SCORE DESC, f.ID
  FETCH FIRST :n ROWS ONLY
)
SELECT ID, ${SCORE_EXPR} AS SCORE
FROM cand
ORDER BY SCORE DESC, ID
FETCH FIRST :topk ROWS ONLY
