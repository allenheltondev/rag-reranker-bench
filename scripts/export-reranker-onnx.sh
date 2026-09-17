#!/usr/bin/env bash
#
# Export BAAI/bge-reranker-base to ONNX for the APPLICATION side of the benchmark.
#
# The in-database copy is a separate export - see sql/README.md - because Oracle needs the
# tokenizer embedded in the graph, and ONNX Runtime in Node does not. Both must come from the
# same checkpoint, or the benchmark compares two models rather than two execution locations.
#
# Usage: scripts/export-reranker-onnx.sh [output-dir]
set -euo pipefail

MODEL="${MODEL:-BAAI/bge-reranker-base}"
OUT="${1:-./models/bge-reranker-base}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

echo "Setting up an isolated environment for the export..."
python3 -m venv .venv-export
# shellcheck disable=SC1091
source .venv-export/bin/activate
pip install --quiet --upgrade pip
pip install --quiet "optimum[exporters]" onnx onnxruntime

echo "Exporting ${MODEL} to ${OUT}..."
mkdir -p "$(dirname "${OUT}")"
optimum-cli export onnx \
  --model "${MODEL}" \
  --task text-classification \
  --opset 17 \
  "${OUT}"

# transformers.js looks for the graph under onnx/, alongside config.json and tokenizer.json.
mkdir -p "${OUT}/onnx"
if [ -f "${OUT}/model.onnx" ]; then
  mv "${OUT}/model.onnx" "${OUT}/onnx/model.onnx"
  [ -f "${OUT}/model.onnx_data" ] && mv "${OUT}/model.onnx_data" "${OUT}/onnx/model.onnx_data"
fi

deactivate
echo
echo "Exported to ${OUT}"
echo "Contents:"
ls -la "${OUT}" "${OUT}/onnx"
echo
echo "Next: set APP_RERANK_MODEL_PATH=${OUT} in .env and run 'npm run doctor -- --skip-oracle'."
