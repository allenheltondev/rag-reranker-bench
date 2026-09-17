-- Container only: the directory object the model loader reads ONNX files from. It points at
-- the ./models/oracle mount declared in docker-compose.yml. Autonomous Database has no
-- filesystem to point at and uses Object Storage instead, so `npm run bootstrap` skips this
-- script when ORACLE_TARGET=adb.
--
-- Tokens: ${BENCH_USER}, ${ONNX_DIRECTORY}, ${ONNX_PATH}

CREATE OR REPLACE DIRECTORY ${ONNX_DIRECTORY} AS '${ONNX_PATH}'
/

GRANT READ ON DIRECTORY ${ONNX_DIRECTORY} TO ${BENCH_USER}
/
