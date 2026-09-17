-- Optional approximate vector index.
--
-- Run this only if you want to benchmark approximate search. The default benchmark uses exact
-- search so that vector recall is not a moving part in the comparison: the question under test
-- is where the cross-encoder runs, not how well an ANN index is tuned.
--
-- It needs vector memory, which the Free container ships with set to zero. Without it you get
-- ORA-51962 and nothing else is affected - the corpus is still loaded and the benchmark still
-- runs. To enable it:
--
--   ALTER SYSTEM SET vector_memory_size = 512M SCOPE=SPFILE;
--   SHUTDOWN IMMEDIATE; STARTUP;
--
-- as SYSDBA, then `npm run load -- --vector-index`.
--
-- Tokens: ${PREFIX}

CREATE VECTOR INDEX ${PREFIX}_CHUNKS_VEC_IX ON ${PREFIX}_CHUNKS (EMBEDDING)
  ORGANIZATION INMEMORY NEIGHBOR GRAPH
  DISTANCE COSINE
  WITH TARGET ACCURACY 95
/
