#!/usr/bin/env python3
"""
Augment an exported cross-encoder so Oracle AI Database can score raw text with it.

WHY THIS EXISTS
    VECTOR_EMBEDDING and PREDICTION hand a model raw text, so the tokenizer has to live
    inside the ONNX graph. `npm run export:app-model` produces the opposite: a plain graph
    that expects input_ids, with tokenization done by the caller. This script takes that
    export and prepends a tokenizer, producing a single graph the database can load.

    It deliberately augments the file the application arm already runs, so both arms of the
    benchmark execute bit-identical weights. That is the whole point of the comparison, and a
    separately-obtained prepared model cannot promise it.

ONE INPUT, NOT TWO
    Oracle's published example calls a reranker with two inputs:

        PREDICTION(model USING :q AS FIRST_INPUT, text AS SECOND_INPUT)

    That is not reproducible with the public tooling: every text tokenizer op in
    onnxruntime-extensions (BertTokenizer, HfJsonTokenizer, SentencepieceTokenizer) accepts
    at most ONE string input. Oracle's own converter must use a custom two-input op.

    So this builds a single-input graph and the pair is packed in SQL instead. That is not a
    compromise on correctness: an XLM-RoBERTa cross-encoder encodes a pair as

        <s> query </s></s> passage </s>

    and feeding "query</s></s>passage" as one string produces exactly that token sequence,
    because the tokenizer parses the separator from the text and adds the outer <s>/</s>
    itself. The script verifies this equivalence against the real tokenizer before writing
    anything, and refuses to emit a model if the two encodings differ.

USAGE
    npm run augment:rerank-model
    npm run augment:rerank-model -- --quantize    (if the fp32 graph is too large to load)

    The script prints the exact ORACLE_INDB_SCORE_EXPR to put in .env.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def fail(message: str) -> "NoReturn":
    print(f"\nError: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="models/bge-reranker-base",
                        help="Directory from `npm run export:app-model` (config.json, tokenizer.json, onnx/model.onnx).")
    parser.add_argument("--out", default="models/oracle/bge_reranker_base.onnx",
                        help="Where to write the augmented graph.")
    parser.add_argument("--quantize", action="store_true",
                        help="Emit int8. Use only if the fp32 graph exceeds the database's model size limit, "
                             "and then point APP_RERANK_DTYPE at the same quantization so both arms still match.")
    args = parser.parse_args()

    try:
        import onnx
        from onnx import TensorProto, helper
        import numpy as np
        from transformers import AutoTokenizer
        from onnxruntime_extensions import gen_processing_models
        import onnxruntime as ort
        import onnxruntime_extensions as ox
    except ImportError as exc:
        fail(f"missing dependency: {exc}.\nRun `npm run export:app-model` first; it builds the environment this uses.")

    model_dir = Path(args.model_dir)
    graph_path = model_dir / "onnx" / "model.onnx"
    if not graph_path.exists():
        fail(f"{graph_path} not found. Run `npm run export:app-model` first.")

    print(f"Loading tokenizer from {model_dir}")
    print(f"  files present: {sorted(p.name for p in model_dir.iterdir() if p.is_file())}")
    # Prefer the fast tokenizer: it carries tokenizer.json, which is the form the converter can
    # embed directly without needing the original sentencepiece model beside it.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    print(f"  {type(tokenizer).__name__}")

    # ---------------------------------------------------------------- separator
    # The string that has to sit between query and passage so that packing the pair into one
    # sequence reproduces the model's own pair encoding. Derived from the tokenizer rather
    # than assumed, because BERT-family models use a single [SEP] and RoBERTa-family use two.
    pair = tokenizer("QUERYSIDE", text_pair="PASSAGESIDE")["input_ids"]
    decoded = tokenizer.decode(pair)
    print(f"  native pair encoding: {decoded}")

    sep = tokenizer.sep_token or tokenizer.eos_token
    if sep is None:
        fail("the tokenizer defines neither a sep_token nor an eos_token, so a pair cannot be packed.")

    separator = None
    for candidate in (sep + sep, sep):
        packed = tokenizer(f"QUERYSIDE{candidate}PASSAGESIDE")["input_ids"]
        if packed == pair:
            separator = candidate
            break

    if separator is None:
        packed = tokenizer("QUERYSIDE" + sep + "PASSAGESIDE")["input_ids"]
        fail(
            "packing the pair into one string does not reproduce this model's pair encoding.\n"
            f"  native : {pair}\n"
            f"  packed : {packed}\n"
            "A single-input graph would therefore score something the model was not trained on.\n"
            "This model needs Oracle's own two-input converter; see sql/README.md."
        )
    print(f"  pair packs exactly with separator {separator!r}")

    # ---------------------------------------------------------------- tokenizer graph
    # ---------------------------------------------------------------- tokenizer graph
    # The converter takes several routes depending on the tokenizer family, and which one works
    # is not knowable in advance: the default path wants the original vocabulary file beside the
    # model, while schema_v2 embeds tokenizer.json into the graph instead. Try each and say
    # which succeeded, rather than failing on the first.
    print("Building the tokenizer graph ...")
    strategies = [
        ("tokenizer.json embedded (schema_v2)", {"schema_v2": True}),
        ("vocabulary file beside the model", {}),
    ]
    pre = None
    failures = []
    for label, kwargs in strategies:
        try:
            pre, _ = gen_processing_models(tokenizer, pre_kwargs={"CAST_TOKEN_ID": True}, **kwargs)
            print(f"  built via: {label}")
            break
        except Exception as exc:
            failures.append(f"  - {label}: {type(exc).__name__}: {exc}")

    if pre is None:
        detail = "\n".join(failures)
        fail(
            f"onnxruntime-extensions could not convert {type(tokenizer).__name__}.\n{detail}\n\n"
            "If every route failed on a missing vocabulary file, the export is missing the\n"
            "tokenizer's original model file. Re-run `npm run export:app-model`, which saves the\n"
            "full tokenizer beside the graph."
        )

    pre_outputs = [o.name for o in pre.graph.output]
    print(f"  emits {pre_outputs}")

    body = onnx.load(str(graph_path))
    body_inputs = [i.name for i in body.graph.input]
    print(f"  model body takes {body_inputs}")

    io_map = []
    for name in body_inputs:
        if name not in pre_outputs:
            fail(
                f"the model body wants '{name}' but the tokenizer graph emits {pre_outputs}.\n"
                "These have to line up before the two graphs can be merged."
            )
        io_map.append((name, name))

    # Drop tokenizer outputs the body does not consume, or merging leaves dangling graph
    # outputs that the database has no metadata for.
    keep = {name for name, _ in io_map}
    for output in [o for o in pre.graph.output if o.name not in keep]:
        pre.graph.output.remove(output)

    print("Merging tokenizer into the model ...")
    try:
        merged = onnx.compose.merge_models(pre, body, io_map=io_map)
    except Exception as exc:
        fail(f"merge failed: {exc}")

    # ---------------------------------------------------------------- shape the output
    # A cross-encoder emits logits of shape [batch, 1]; a regression model in the database is
    # expected to produce one number per row, so the trailing dimension is squeezed away.
    out_name = merged.graph.output[0].name
    squeezed = f"{out_name}_score"
    axes = helper.make_tensor(f"{squeezed}_axes", TensorProto.INT64, [1], [1])
    merged.graph.initializer.append(axes)
    merged.graph.node.append(
        helper.make_node("Squeeze", [out_name, f"{squeezed}_axes"], [squeezed], name="squeeze_logit")
    )
    merged.graph.output.append(helper.make_tensor_value_info(squeezed, TensorProto.FLOAT, [None]))
    for old in [o for o in merged.graph.output if o.name == out_name]:
        merged.graph.output.remove(old)

    merged.graph.input[0].name = "input"
    for node in merged.graph.node:
        node.input[:] = ["input" if i == pre.graph.input[0].name else i for i in node.input]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.quantize:
        print("Quantizing to int8 ...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        tmp = out_path.with_suffix(".fp32.onnx")
        onnx.save(merged, str(tmp))
        quantize_dynamic(str(tmp), str(out_path), weight_type=QuantType.QInt8)
        os.remove(tmp)
    else:
        onnx.save(merged, str(out_path), save_as_external_data=False)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.0f} MB)")

    # ---------------------------------------------------------------- smoke test
    print("Scoring a matching and a mismatched pair to check the graph runs ...")
    try:
        so = ort.SessionOptions()
        so.register_custom_ops_library(ox.get_library_path())
        sess = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        good = sess.run(None, {name: np.array([f"what colour is the sky{separator}The sky is blue."])})[0]
        bad = sess.run(None, {name: np.array([f"what colour is the sky{separator}Grass is green."])})[0]
        print(f"  matching pair   {float(good[0]):+.4f}")
        print(f"  mismatched pair {float(bad[0]):+.4f}")
        if float(good[0]) <= float(bad[0]):
            print("  WARNING: the matching pair did not score higher. Check the merge before trusting any ranking.")
        else:
            print("  ordering is correct")
    except Exception as exc:
        print(f"  could not run the graph locally: {exc}")
        print("  The file was still written; the database will be the real test.")

    sql_sep = separator.replace("'", "''")
    print(f"""
Next steps:

  1. npm run models:rerank

  2. Put this in .env - the pair is packed in SQL because the graph takes one input:

     ORACLE_INDB_SCORE_EXPR=PREDICTION({os.environ.get('ORACLE_RERANK_MODEL', 'BGE_RERANKER')} USING :qtext || '{sql_sep}' || TITLE || '. ' || CONTENT AS DATA)

  3. npm run doctor

     Doctor scores an obviously matching pair against an irrelevant one. If that check
     passes, the in-database arm is ready:

       npm run bench -- --candidates 10,40 --iterations 10 --repeats 3
""")


if __name__ == "__main__":
    main()
