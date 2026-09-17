-- Candidate generation: scope filter -> lexical + vector retrieval -> RRF fusion.
--
-- This one statement is the "retrieval held constant" part of the benchmark. Both reranker
-- implementations score exactly the candidates this produces, so any difference in the final
-- ordering is attributable to the reranker and not to retrieval.
--
-- Tokens: ${PREFIX}, ${EMBED_MODEL}, ${VEC_SOURCE}, ${LEX_SOURCE}, ${FUSED_BODY}
-- Binds:  :qtext :contains :tenant :owner :pool :n :rrfk
--
-- The three retrieval strategies are the same statement with different arms substituted in by
-- src/retrieval/candidates.ts: the unused arm becomes a no-op subquery rather than being left
-- in place, so vector-only never pays for a lexical scan it does not use. Run any stage with
-- `--dump-sql` to print the fully rendered statement.

WITH qv AS (
  SELECT VECTOR_EMBEDDING(${EMBED_MODEL} USING :qtext AS DATA) AS V FROM DUAL
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
)
SELECT f.ID, c.TITLE, c.CONTENT, f.SCORE, f.VRANK, f.LRANK
FROM fused f
JOIN ${PREFIX}_CHUNKS c ON c.ID = f.ID
ORDER BY f.SCORE DESC, f.ID
FETCH FIRST :n ROWS ONLY
