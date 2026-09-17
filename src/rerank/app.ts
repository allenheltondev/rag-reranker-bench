import { basename, dirname } from 'node:path';
import { app } from '../config.js';
import type { Candidate, RankedResult } from '../types.js';

export interface RerankTimings {
  tokenize: number;
  infer: number;
  sort: number;
}

export interface AppRerankOutcome {
  results: RankedResult[];
  timings: RerankTimings;
}

/**
 * Application-side cross-encoder.
 *
 * This deliberately runs the *same ONNX file* the database runs, through ONNX Runtime, so the
 * comparison is about where inference happens rather than which model won. Anything that would
 * change the model itself - quantization, a different checkpoint, a GPU execution provider -
 * is configuration, and whatever is configured gets recorded in the report.
 */
export class AppReranker {
  private tokenizer: any = null;
  private model: any = null;
  private loaded = false;

  async load(): Promise<void> {
    if (this.loaded) return;
    // Imported lazily: the fixture backend must work on a machine with no model and no
    // native ONNX Runtime binary available.
    const tf = await import('@huggingface/transformers');
    const { AutoTokenizer, AutoModelForSequenceClassification, env } = tf as any;

    // Never reach for the network. If the export step has not been run, fail with a message
    // that says so instead of silently downloading a different checkpoint than the database has.
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = dirname(app.modelPath);

    const id = basename(app.modelPath);
    const options: Record<string, unknown> = { local_files_only: true };
    if (app.intraOpThreads !== undefined) {
      options['session_options'] = { intraOpNumThreads: app.intraOpThreads };
    }

    try {
      this.tokenizer = await AutoTokenizer.from_pretrained(id, { local_files_only: true });
      this.model = await AutoModelForSequenceClassification.from_pretrained(id, {
        ...options,
        dtype: app.dtype,
        device: app.device,
      });
    } catch (err) {
      throw new Error(
        `Could not load the application-side reranker from ${app.modelPath}.\n` +
        `Run scripts/export-reranker-onnx.sh first, and make sure APP_RERANK_MODEL_PATH points ` +
        `at a directory containing config.json, tokenizer.json and onnx/model.onnx.\n\n` +
        `${(err as Error).message}`,
      );
    }
    this.loaded = true;
  }

  describe(): Record<string, unknown> {
    return {
      modelPath: app.modelPath,
      dtype: app.dtype,
      device: app.device,
      intraOpThreads: app.intraOpThreads ?? 'default',
      maxLength: app.maxLength,
      batchSize: app.batchSize,
    };
  }

  /**
   * Score every (query, candidate) pair and return the top K.
   *
   * Tokenization and inference are timed separately because they answer different questions:
   * tokenization is work the database also does but never reports, and inference is the part
   * a GPU would change.
   */
  async rerank(queryText: string, candidates: readonly Candidate[], topK: number): Promise<AppRerankOutcome> {
    await this.load();
    const timings: RerankTimings = { tokenize: 0, infer: 0, sort: 0 };
    const scores: number[] = new Array(candidates.length).fill(0);

    for (let start = 0; start < candidates.length; start += app.batchSize) {
      const batch = candidates.slice(start, start + app.batchSize);
      // The same string the in-database path scores: "TITLE. CONTENT".
      const pairs = batch.map((c) => `${c.title}. ${c.content}`);

      const t0 = performance.now();
      const inputs = this.tokenizer(new Array(batch.length).fill(queryText), {
        text_pair: pairs,
        padding: true,
        truncation: true,
        max_length: app.maxLength,
      });
      const t1 = performance.now();

      const output = await this.model(inputs);
      const t2 = performance.now();

      timings.tokenize += t1 - t0;
      timings.infer += t2 - t1;

      const logits = output.logits ?? output.output ?? output;
      const data: ArrayLike<number> = logits.data ?? logits;
      const perItem = data.length / batch.length;
      for (let i = 0; i < batch.length; i++) {
        // bge-reranker emits a single relevance logit per pair. If a checkpoint emits two
        // class logits, rank by the positive class instead of the raw first value.
        scores[start + i] = perItem === 1 ? Number(data[i]) : Number(data[i * perItem + (perItem - 1)]);
      }
    }

    const t3 = performance.now();
    const ordered = candidates
      .map((c, i) => ({ chunkId: c.chunkId, score: scores[i]!, tieBreak: c.chunkId }))
      .sort((a, b) => (b.score - a.score) || a.tieBreak.localeCompare(b.tieBreak))
      .slice(0, topK)
      .map((r, i): RankedResult => ({ chunkId: r.chunkId, rank: i + 1, score: r.score }));
    timings.sort = performance.now() - t3;

    return { results: ordered, timings };
  }

  async close(): Promise<void> {
    if (this.model?.dispose) await this.model.dispose();
    this.model = null;
    this.tokenizer = null;
    this.loaded = false;
  }
}
