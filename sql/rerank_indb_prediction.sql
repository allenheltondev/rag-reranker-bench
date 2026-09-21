-- In-database reranking: the whole pipeline, including cross-encoder scoring, in one statement.
--
-- The projection is deliberately narrow. Only IDs and scores cross the network; the candidate
-- text is read, tokenized and scored inside the database and never leaves it. That is the
-- architectural claim this benchmark measures the cost of.
--
-- Tokens: ${PREFIX}, ${QUERY_VECTOR}, ${SCORE_EXPR}, ${VEC_SOURCE},
--         ${LEX_SOURCE}, ${FUSED_BODY}
-- Binds:  :qtext :qvec (separate-session mode) :contains :tenant :owner :pool :n :rrfk :topk
--
-- ${SCORE_EXPR} defaults to PREDICTION(...). If your exported cross-encoder loads as a
-- classification model rather than a regression, switch it to PREDICTION_PROBABILITY via
-- ORACLE_INDB_SCORE_EXPR; ordering by a predicted class value ranks nothing.

WITH qv AS (
  SELECT ${QUERY_VECTOR} AS V FROM DUAL
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
),
counted AS (
  -- Count after candidate limiting, before scoring and top-K. Keeping this in its own
  -- query block avoids projecting PREDICTION below the analytic operation.
  SELECT cand.*, COUNT(*) OVER () AS CANDIDATES_SCORED FROM cand
)
SELECT ID, ${SCORE_EXPR} AS SCORE, CANDIDATES_SCORED
FROM counted
ORDER BY SCORE DESC, ID
FETCH FIRST :topk ROWS ONLY
