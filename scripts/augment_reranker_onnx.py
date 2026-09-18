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
                        help="Quantize to int8. Quantizes the model BODY first and writes it back beside the "
                             "application's export, so both arms run the same quantized weights; then augments "
                             "that. Set APP_RERANK_DTYPE=q8 afterwards.")
    args = parser.parse_args()

    try:
        import onnx
        from onnx import TensorProto, helper
        import numpy as np
        from transformers import AutoTokenizer
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

    # ---------------------------------------------------------------- reference encoding
    # What the model actually expects for a pair. Everything below is judged against this.
    probe_q, probe_d = "what colour is the sky", "The sky is blue."
    native = list(tokenizer(probe_q, text_pair=probe_d)["input_ids"])
    print(f"  the model's own pair encoding is {len(native)} tokens: {native[:8]}...")

    # ---------------------------------------------------------------- tokenizer graph
    # Built directly from the two tokenizer files rather than through gen_processing_models,
    # which reaches the same operator only after trying to resolve the tokenizer by name
    # against Hugging Face. Everything needed is already on disk: HfJsonTokenizer takes
    # tokenizer.json and tokenizer_config.json as graph attributes.
    print("Building the tokenizer graph ...")
    try:
        from onnxruntime_extensions._cuops import SingleOpGraph
    except ImportError as exc:
        fail(f"onnxruntime-extensions is missing its graph builder: {exc}")

    vocab_file = model_dir / "tokenizer.json"
    config_file = model_dir / "tokenizer_config.json"
    for required in (vocab_file, config_file):
        if not required.exists():
            fail(f"{required} not found. Re-run `npm run export:app-model`.")

    tok_graph = SingleOpGraph.build_graph(
        "HfJsonTokenizer",
        tokenizer_vocab=vocab_file.read_text(encoding="utf-8"),
        tokenizer_config=config_file.read_text(encoding="utf-8"),
    )
    tok_node = tok_graph.node[0]
    print(f"  {tok_node.op_type} from {vocab_file.name} + {config_file.name}")

    if args.quantize:
        # Quantize the BODY, and write it where the application reranker will pick it up as its
        # q8 variant. Augmenting the quantized body then gives the database the same weights the
        # application runs. Quantizing the merged graph instead would leave the two arms with
        # separately-quantized models, which is exactly the thing this benchmark cannot have.
        from onnxruntime.quantization import quantize_dynamic, QuantType
        app_quantized = graph_path.parent / "model_quantized.onnx"
        print(f"Quantizing the model body to int8 -> {app_quantized}")
        quantize_dynamic(str(graph_path), str(app_quantized), weight_type=QuantType.QInt8)
        graph_path = app_quantized

    body = onnx.load(str(graph_path))
    body_inputs = [i.name for i in body.graph.input]
    print(f"  model body takes {body_inputs}")

    # HfJsonTokenizer emits token ids only. The body also wants an attention mask, and BERT
    # derivatives want segment ids, so both are synthesised from the ids' own shape. The mask
    # is all ones because the database scores one row at a time: there is no batch, therefore
    # no padding to mask out.
    TOK_IN, IDS = "input", "__ids"
    nodes = [helper.make_node(tok_node.op_type, [TOK_IN], [IDS], domain=tok_node.domain,
                              name="tokenize", **{a.name: helper.get_attribute_value(a) for a in tok_node.attribute})]
    nodes.append(helper.make_node("Cast", [IDS], ["__ids64"], to=TensorProto.INT64, name="cast_ids"))

    # The operator emits raw ids: it does not apply the tokenizer's post-processor, so the
    # sequence markers a pair is normally wrapped in are missing. They are added here as graph
    # constants rather than as text, because a special token written into the text makes the
    # metaspace pre-tokenizer emit a stray separator token ahead of it. The ids are exact and
    # position-independent; the text route is neither.
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    initializers = []
    wrapped = ["__ids64"]
    if bos_id is not None:
        initializers.append(helper.make_tensor("__bos", TensorProto.INT64, [1], [int(bos_id)]))
        wrapped.insert(0, "__bos")
    if eos_id is not None:
        initializers.append(helper.make_tensor("__eos", TensorProto.INT64, [1], [int(eos_id)]))
        wrapped.append("__eos")
    print(f"  wrapping ids with bos={bos_id} eos={eos_id}")
    nodes.append(helper.make_node("Concat", wrapped, ["__full"], axis=0, name="add_markers"))

    shape_1xN = helper.make_tensor("__shape_1xN", TensorProto.INT64, [2], [1, -1])
    initializers.append(shape_1xN)
    nodes.append(helper.make_node("Reshape", ["__full", "__shape_1xN"], ["input_ids"], name="as_batch_of_one"))
    nodes.append(helper.make_node("Shape", ["input_ids"], ["__shape"], name="ids_shape"))

    pre_outputs = ["input_ids"]
    for name, value in (("attention_mask", 1), ("token_type_ids", 0)):
        if name not in body_inputs:
            continue
        # The fill value rides on the node as an attribute; adding it to the graph's
        # initializers as well leaves an unreferenced tensor that the runtime warns about.
        const = helper.make_tensor(f"__{name}_value", TensorProto.INT64, [1], [value])
        nodes.append(helper.make_node("ConstantOfShape", ["__shape"], [name],
                                      value=const, name=f"make_{name}"))
        pre_outputs.append(name)

    pre_graph = helper.make_graph(
        nodes, "tokenize",
        [helper.make_tensor_value_info(TOK_IN, TensorProto.STRING, [None])],
        [helper.make_tensor_value_info(n, TensorProto.INT64, [1, None]) for n in pre_outputs],
        initializer=initializers,
    )
    # Match the body's IR version and default opset. onnx.compose refuses to merge models
    # that disagree on either, and the body's are whatever the exporter wrote, which is
    # generally older than the version of onnx doing the merge.
    body_opset = next((o.version for o in body.opset_import if o.domain in ("", "ai.onnx")), 17)
    pre = helper.make_model(pre_graph, opset_imports=[
        helper.make_opsetid("", body_opset), helper.make_opsetid(tok_node.domain, 1),
    ])
    pre.ir_version = body.ir_version
    print(f"  emits {pre_outputs} (IR {pre.ir_version}, opset {body_opset})")

    io_map = []
    for name in body_inputs:
        if name not in pre_outputs:
            fail(
                f"the model body wants '{name}' but the tokenizer graph emits {pre_outputs}.\n"
                "These have to line up before the two graphs can be merged."
            )
        io_map.append((name, name))

    # ------------------------------------------------------- choose the SQL packing
    # The operator emits raw token ids: it does NOT apply the tokenizer's post-processor, so
    # the markers a pair normally gets wrapped in have to arrive as part of the text. Rather
    # than assume a template, run the operator on each candidate and keep the one whose ids
    # equal what the model expects. If none match, the two arms would tokenize differently and
    # the comparison would be meaningless, so that is a hard failure.
    # Whitespace around the separator matters: the metaspace pre-tokenizer marks a word start,
    # so a passage that begins immediately after a special token tokenizes differently from the
    # same passage tokenized on its own. Enumerate the placements rather than reason about it.
    sep = tokenizer.sep_token or tokenizer.eos_token or ""
    candidates = []
    for count in (2, 1):
        joiner = sep * count
        for before in ("", " "):
            for after in ("", " "):
                label = f"query{before}{'</s>' * count}{after}passage".replace("</s>", sep)
                candidates.append((label, f"{{q}}{before}{joiner}{after}{{d}}"))

    print("Finding the packing whose in-database tokenization matches the model's ...")
    tok_only = helper.make_model(
        helper.make_graph([nodes[0]], "tok",
                          [helper.make_tensor_value_info(TOK_IN, TensorProto.STRING, [None])],
                          [helper.make_tensor_value_info(IDS, TensorProto.INT64, [None])]),
        opset_imports=[helper.make_opsetid("", body_opset), helper.make_opsetid(tok_node.domain, 1)],
    )
    tok_only.ir_version = body.ir_version
    so = ort.SessionOptions()
    so.register_custom_ops_library(ox.get_library_path())
    try:
        tok_sess = ort.InferenceSession(tok_only.SerializeToString(), so, providers=["CPUExecutionProvider"])
    except Exception as exc:
        fail(f"the tokenizer graph does not run: {exc}")

    template = None
    for label, pattern in candidates:
        packed = pattern.format(q=probe_q, d=probe_d)
        raw = tok_sess.run(None, {TOK_IN: np.array([packed])})[0].reshape(-1).tolist()
        # What the merged graph will actually feed the model, markers included.
        got = ([int(bos_id)] if bos_id is not None else []) + raw + ([int(eos_id)] if eos_id is not None else [])
        if got == native:
            print(f"  {label:34s} MATCH")
            template = pattern
            break
        # Show where it first diverges; the position tells you whether the problem is the
        # markers, the separator, or the passage's leading word.
        at = next((i for i, (a, b) in enumerate(zip(got, native)) if a != b), min(len(got), len(native)))
        print(f"  {label:34s} differs at token {at} ({len(got)} vs {len(native)} tokens)")

    if template is None:
        fail(
            "no packing reproduced the model's own pair encoding.\n"
            f"  expected : {native}\n"
            "The in-database arm would tokenize differently from the application arm, so their\n"
            "scores would not be comparable and the benchmark's central claim would be false.\n"
            "Report this output; it needs a graph-level fix rather than a different SQL string."
        )
    print(f"  using: {template}")

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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        good = sess.run(None, {name: np.array([template.format(q=probe_q, d="The sky is blue.")])})[0]
        bad = sess.run(None, {name: np.array([template.format(q=probe_q, d="Grass is green.")])})[0]
        print(f"  matching pair   {float(good[0]):+.4f}")
        print(f"  mismatched pair {float(bad[0]):+.4f}")
        if float(good[0]) <= float(bad[0]):
            print("  WARNING: the matching pair did not score higher. Check the merge before trusting any ranking.")
        else:
            print("  ordering is correct")
    except Exception as exc:
        print(f"  could not run the graph locally: {exc}")
        print("  The file was still written; the database will be the real test.")

    def sql_literal(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    # Turn the winning template into a SQL concatenation of literals and columns.
    sql_expr = (
        template.replace("{q}", "\x00Q\x00").replace("{d}", "\x00D\x00")
    )
    parts = []
    for chunk in sql_expr.split("\x00"):
        if chunk == "Q":
            parts.append(":qtext")
        elif chunk == "D":
            parts.append("TITLE || '. ' || CONTENT")
        elif chunk:
            parts.append(sql_literal(chunk))
    packed_sql = " || ".join(parts)
    quantize_note = ("""
  2b. Set APP_RERANK_DTYPE=q8 in .env. The application arm must run the quantized weights this
      just produced, or the two arms are no longer the same model."""
                     if args.quantize else "")

    print(f"""
Next steps:

  1. npm run models:rerank

  2. Put this in .env - the pair is packed in SQL because the graph takes one input:

     ORACLE_INDB_SCORE_EXPR=PREDICTION({os.environ.get('ORACLE_RERANK_MODEL', 'BGE_RERANKER')} USING {packed_sql} AS DATA)

{quantize_note}
  3. npm run doctor

     Doctor scores an obviously matching pair against an irrelevant one. If that check
     passes, the in-database arm is ready:

       npm run bench -- --candidates 10,40 --iterations 10 --repeats 3
""")


if __name__ == "__main__":
    main()
