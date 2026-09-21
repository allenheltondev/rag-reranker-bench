import { cpus } from 'node:os';
import oracledb from 'oracledb';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { app, buildStages, oracle, paths, runConfigFromEnv } from './config.js';
import { generateCorpus } from './corpus/generate.js';
import { assertIdentifier, closePool, describeOracle, hintFor, withConnection } from './db/oracle.js';
import { countChunks, loadChunks, runScript } from './db/load.js';
import { AppReranker } from './rerank/app.js';
import { InDbReranker, describeAttributeMismatch, scoreExpr } from './rerank/indb.js';
import { bindsFor } from './retrieval/candidates.js';
import { usedBinds } from './db/sql.js';
import { buildDeps, pipelineFor, runBenchmark, verifyCandidateParity } from './bench/harness.js';
import { renderCostsCsv, renderCsv, renderMarkdown } from './bench/report.js';
import { renderInspection } from './bench/inspect.js';
import { exportAppModel } from './tools/export-app-model.js';
import { readZip } from './tools/unzip.js';
import type { BenchRun, Chunk, Query, RunConfig } from './types.js';

const args = process.argv.slice(2);
const command = args[0] ?? 'help';
const flag = (name: string): boolean => args.includes(`--${name}`);
const opt = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};

const log = (msg = ''): void => { process.stdout.write(`${msg}\n`); };

function overridesFromFlags(): Partial<RunConfig> {
  const o: Partial<RunConfig> = {};
  const backend = opt('backend');
  if (backend === 'oracle' || backend === 'fixture') o.backend = backend;
  const iterations = opt('iterations');
  if (iterations) o.iterations = Number(iterations);
  const warmup = opt('warmup');
  if (warmup) o.warmup = Number(warmup);
  const topK = opt('top-k');
  if (topK) o.topK = Number(topK);
  const candidates = opt('candidates');
  if (candidates) o.candidateCounts = candidates.split(',').map(Number);
  const rerankers = opt('rerankers');
  if (rerankers) o.rerankers = rerankers.split(',') as RunConfig['rerankers'];
  const retrievals = opt('retrievals');
  if (retrievals) o.retrievals = retrievals.split(',') as RunConfig['retrievals'];
  const queries = opt('queries');
  if (queries) o.queries = Number(queries);
  const repeats = opt('repeats');
  if (repeats) o.repeats = Number(repeats);
  const quiesce = opt('quiesce-ms');
  if (quiesce) o.quiesceMs = Number(quiesce);
  const reset = opt('reset-cmd');
  if (reset) o.resetCommand = reset;
  return o;
}

function readCorpus(): { chunks: Chunk[]; queries: Query[] } {
  try {
    return {
      chunks: JSON.parse(readFileSync(paths.corpus, 'utf8')) as Chunk[],
      queries: JSON.parse(readFileSync(paths.queries, 'utf8')) as Query[],
    };
  } catch {
    throw new Error(`Corpus not found at ${paths.corpus}. Run \`npm run corpus\` first.`);
  }
}

async function cmdCorpus(): Promise<void> {
  const cfg = runConfigFromEnv(overridesFromFlags());
  const { chunks, queries } = generateCorpus(cfg.corpusSize, cfg.seed);
  mkdirSync(resolve(paths.corpus, '..'), { recursive: true });
  writeFileSync(paths.corpus, `${JSON.stringify(chunks, null, 2)}\n`);
  writeFileSync(paths.queries, `${JSON.stringify(queries, null, 2)}\n`);

  const graded = queries.reduce((a, q) => a + Object.keys(q.judgments).length, 0);
  const expired = chunks.filter((c) => c.expiresAt !== null).length;
  const otherTenant = chunks.filter((c) => c.tenant !== 'acme').length;
  log(`Wrote ${chunks.length} chunks to ${paths.corpus}`);
  log(`Wrote ${queries.length} queries to ${paths.queries}`);
  log(`  graded chunks: ${graded}`);
  log(`  expired chunks (must be filtered out): ${expired}`);
  log(`  other-tenant chunks (must never be returned): ${otherTenant}`);
  const byKind = queries.reduce<Record<string, number>>((acc, q) => {
    acc[q.kind] = (acc[q.kind] ?? 0) + 1;
    return acc;
  }, {});
  log(`  queries by hazard: ${Object.entries(byKind).map(([k, v]) => `${k}=${v}`).join(' ')}`);
}

async function cmdLoad(): Promise<void> {
  const { chunks } = readCorpus();
  log('Creating schema...');
  await runScript('01_schema.sql');
  log(`Loading ${chunks.length} chunks and embedding them with ${oracle.embedModel}...`);
  const inserted = await loadChunks(chunks);
  log(`Inserted ${inserted} rows; table now holds ${await countChunks()}.`);
  if (flag('vector-index')) {
    log('Creating the approximate vector index...');
    await runScript('02_vector_index.sql');
  } else {
    log('Skipped the approximate vector index (exact search keeps recall out of the comparison).');
    log('Pass --vector-index to create it anyway.');
  }
}

/**
 * Augment the exported cross-encoder so the database can score raw text with it.
 *
 * Runs in the virtualenv `npm run export:app-model` already built, because it needs the same
 * onnx version that produced the graph it is modifying.
 */
async function cmdAugmentRerankModel(): Promise<void> {
  const { spawnSync } = await import('node:child_process');
  const { findPython } = await import('./tools/export-app-model.js');

  if (!existsSync(resolve(app.modelPath, 'onnx', 'model.onnx'))) {
    throw new Error(`No export found at ${app.modelPath}. Run \`npm run export:app-model\` first.`);
  }

  // A separate environment from the export's, for two reasons: the augmentation needs no
  // torch (so this stays small), and onnxruntime-extensions publishes wheels for a narrower
  // set of Python versions than the exporter does, so it may need an older interpreter than
  // the export ran on.
  const venvDir = resolve('.venv-augment');
  const venvPython = process.platform === 'win32'
    ? resolve(venvDir, 'Scripts', 'python.exe')
    : resolve(venvDir, 'bin', 'python');

  if (!existsSync(venvPython)) {
    const python = findPython(['3.13', '3.12', '3.11', '3.10']);
    log(`Creating ${venvDir} with ${python.join(' ')} ...`);
    const venv = spawnSync(python[0]!, [...python.slice(1), '-m', 'venv', venvDir], { stdio: 'inherit' });
    if (venv.status !== 0) throw new Error('Could not create the augmentation environment.');
  }

  log('Installing onnx, onnxruntime, tokenizers and onnxruntime-extensions==0.15.0 ...');
  const install = spawnSync(
    venvPython,
    ['-m', 'pip', 'install', '--quiet', '--upgrade', 'pip'],
    { stdio: 'inherit' },
  );
  if (install.status !== 0) throw new Error('pip upgrade failed.');
  // Pinned: 0.15.2 publishes no Windows wheels at all, and 0.15.0's top out at CPython 3.13.
  const deps = spawnSync(
    venvPython,
    // Deliberately not `transformers`: this needs only the tokenizer, and that library costs
    // tens of seconds of import time on every run while scanning its model registry.
    ['-m', 'pip', 'install', '--quiet', 'onnx', 'onnxruntime', 'tokenizers',
      'onnxruntime-extensions==0.15.0'],
    { stdio: 'inherit' },
  );
  if (deps.status !== 0) {
    throw new Error(
      'Could not install the augmentation dependencies.\n'
      + 'onnxruntime-extensions ships wheels for CPython 3.10-3.13 only. Delete .venv-augment, '
      + 'install one of those versions, and run this again.',
    );
  }

  const res = spawnSync(venvPython, [resolve('scripts/augment_reranker_onnx.py'), ...args.slice(1)], { stdio: 'inherit' });
  if (res.status !== 0) throw new Error(`Augmentation failed with exit code ${res.status}.`);
}

/** Export the application-side cross-encoder. Cross-platform; see src/tools/export-app-model.ts. */
async function cmdExportAppModel(): Promise<void> {
  exportAppModel({
    model: opt('hf-model') ?? 'BAAI/bge-reranker-base',
    outDir: opt('out') ?? app.modelPath,
    log,
  });
}

/**
 * Download a database-side ONNX model and put it where the database can read it.
 *
 * Exists because the useful form of these models is published as a zip, and unpacking one
 * into the right place is three fiddly steps that differ per platform. Takes a URL because
 * the download locations move between releases; the README says where to find the current one.
 */
async function cmdFetchModel(): Promise<void> {
  const size = (bytes: number): string =>
    bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  const url = opt('url') ?? args[1];
  if (!url || !/^https?:\/\//.test(url)) {
    throw new Error('Usage: npm run fetch:model -- <url> [--as filename.onnx]');
  }
  mkdirSync(paths.oracleModels, { recursive: true });

  log(`Downloading ${url}`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`Download failed: HTTP ${res.status} ${res.statusText}`);
  const buf = Buffer.from(await res.arrayBuffer());
  log(`  ${size(buf.length)} received`);

  // A login page or an error page is the most common thing to get instead of a model, and it
  // would otherwise be written out and fail much later with an unhelpful Oracle error.
  const head = buf.subarray(0, 512).toString('utf8').trimStart().toLowerCase();
  if (head.startsWith('<!doctype html') || head.startsWith('<html')) {
    throw new Error(
      'That URL returned an HTML page, not a file. It is probably a login or landing page; '
      + 'use the direct download link.',
    );
  }

  const written: string[] = [];
  const isZip = buf.length > 4 && buf.readUInt32LE(0) === 0x04034b50;
  if (isZip) {
    const { entries, names } = readZip(buf, (n) => n.toLowerCase().endsWith('.onnx'));
    if (entries.length === 0) {
      throw new Error(`No .onnx file inside the archive. It contains: ${names.join(', ')}`);
    }
    for (const entry of entries) {
      // Flatten: the database reads a directory, not a tree, and archive paths vary.
      const name = opt('as') ?? basename(entry.name);
      const dest = resolve(paths.oracleModels, name);
      writeFileSync(dest, entry.data);
      written.push(`${dest} (${size(entry.data.length)})`);
    }
  } else {
    const name = opt('as') ?? (basename(new URL(url).pathname) || 'model.onnx');
    const dest = resolve(paths.oracleModels, name);
    writeFileSync(dest, buf);
    written.push(`${dest} (${size(buf.length)})`);
  }

  log('');
  for (const w of written) log(`Wrote ${w}`);
  log('');
  log('The database reads these from inside its own container. Confirm it can see them:');
  log('  docker compose exec oracle ls -l /opt/oracle/onnx');
  log('');
  log(`Then: npm run models:embed   (expects ${oracle.embedFile})`);
  log(`  or: npm run models:rerank  (expects ${oracle.rerankFile})`);
  log('If the file landed under a different name, set ORACLE_EMBED_FILE / ORACLE_RERANK_FILE in .env.');
}

/**
 * Create the benchmark user. The one command that uses elevated credentials.
 *
 * Safe to re-run: an existing user has its password reset to what .env says rather than
 * failing, so a forgotten password is fixed by running this again.
 */
async function cmdBootstrap(): Promise<void> {
  const target = flag('adb') ? 'adb' : oracle.target;
  if (!oracle.user || !oracle.password) {
    throw new Error('ORACLE_USER and ORACLE_PASSWORD must be set. Copy .env.example to .env first.');
  }
  const user = assertIdentifier(oracle.user, 'ORACLE_USER');
  // The password goes into DDL inside double quotes, so a double quote in it would terminate
  // the identifier. Oracle does not allow one in a password anyway; fail clearly rather than
  // emitting SQL that means something other than intended.
  if (oracle.password.includes('"')) {
    throw new Error('ORACLE_PASSWORD cannot contain a double quote.');
  }
  const tablespace = assertIdentifier(
    oracle.tablespace || (target === 'adb' ? 'DATA' : 'USERS'),
    'ORACLE_TABLESPACE',
  );

  log(`Creating ${user} on ${oracle.connectString} as ${oracle.sysUser}${target === 'adb' ? '' : ' (SYSDBA)'}...`);
  await runScript('00_user.sql', {
    BENCH_USER: user,
    BENCH_PASSWORD: oracle.password,
    TABLESPACE: tablespace,
  }, 'elevated');
  log(`  user and grants applied, quota on ${tablespace}`);

  if (target === 'adb') {
    log('  skipping the directory object: Autonomous reads models from Object Storage.');
  } else {
    const dir = assertIdentifier(oracle.onnxDirectory, 'ORACLE_ONNX_DIRECTORY');
    await runScript('00_user_local.sql', {
      BENCH_USER: user,
      ONNX_DIRECTORY: dir,
      ONNX_PATH: oracle.onnxPath.replace(/'/g, "''"),
    }, 'elevated');
    log(`  directory ${dir} -> ${oracle.onnxPath}, readable by ${user}`);
  }

  log('');
  log('Next: put the ONNX files in ./models/oracle, then `npm run models`.');
}

/** Load the ONNX models into the database, from a directory object or from Object Storage. */
async function cmdModels(): Promise<void> {
  const target = flag('adb') ? 'adb' : oracle.target;
  // `--only` is also an npm config flag, and npm can consume it before the script sees it.
  // `--model` is the documented spelling; npm run models:embed / models:rerank use it.
  const only = opt('model') ?? opt('only');
  if (only !== undefined && only !== 'embed' && only !== 'rerank') {
    throw new Error(`--model takes 'embed' or 'rerank', got ${JSON.stringify(only)}.`);
  }
  // Each model is one statement in the script, embedding first.
  const indices = only === 'embed' ? [0] : only === 'rerank' ? [1] : undefined;
  const what = only === 'embed' ? oracle.embedFile : only === 'rerank' ? oracle.rerankFile
    : `${oracle.embedFile} and ${oracle.rerankFile}`;

  if (target === 'adb') {
    if (!oracle.modelsParUrl) {
      throw new Error('ORACLE_MODELS_PAR_URL is required for the adb target (terraform output models_par_base_url).');
    }
    const base = oracle.modelsParUrl.endsWith('/') ? oracle.modelsParUrl : `${oracle.modelsParUrl}/`;
    log(`Loading ${what} from Object Storage...`);
    await runScript('03_load_models_adb.sql', {
      PAR_BASE_URL: base.replace(/'/g, "''"),
      EMBED_FILE: oracle.embedFile,
      RERANK_FILE: oracle.rerankFile,
      RERANK_INPUT: oracle.rerankInputSpec,
    }, 'bench', indices);
  } else {
    log(`Loading ${what} from directory ${oracle.onnxDirectory}...`);
    await runScript('03_load_models.sql', {
      ONNX_DIRECTORY: assertIdentifier(oracle.onnxDirectory, 'ORACLE_ONNX_DIRECTORY'),
      EMBED_FILE: oracle.embedFile,
      RERANK_FILE: oracle.rerankFile,
      RERANK_INPUT: oracle.rerankInputSpec,
    }, 'bench', indices);
  }
  const info = await describeOracle();
  log(`Models now in schema: ${info.models.join(', ') || '(none)'}`);
  await closePool();
}

/** Check every moving part before a run, and say precisely which one is not ready. */
async function cmdDoctor(): Promise<void> {
  let failures = 0;
  const check = async (name: string, fn: () => Promise<string>): Promise<void> => {
    try {
      log(`  ok   ${name}: ${await fn()}`);
    } catch (err) {
      failures++;
      log(`  FAIL ${name}: ${(err as Error).message.split('\n')[0]}`);
    }
  };

  log('Corpus');
  await check('data files', async () => {
    const { chunks, queries } = readCorpus();
    return `${chunks.length} chunks, ${queries.length} queries`;
  });

  if (!flag('skip-oracle')) {
    log('Oracle');
    await check('connection', async () => {
      const info = await describeOracle();
      return `${info.version} (${info.clientMode} client)`;
    });
    // cpu_count, not the container's nproc: an edition cap or an instance setting can hold it
    // below what the OS exposes, and it governs how much of the machine in-database scoring
    // can actually use.
    await check('database memory', async () => {
      const info = await describeOracle();
      // Judge the PGA against the model that has to fit in it, not against a fixed number:
      // a quantized model fits comfortably where an fp32 one cannot.
      const modelPath = resolve(paths.oracleModels, oracle.rerankFile);
      const modelMb = existsSync(modelPath) ? Math.round(statSync(modelPath).size / 1024 / 1024) : null;
      const tight = info.pgaTargetMb !== null && modelMb !== null && info.pgaTargetMb < modelMb * 1.5;
      return `pga_aggregate_target=${info.pgaTargetMb ?? '?'} MB · pga_aggregate_limit=${info.pgaLimitMb ?? '?'} MB`
        + ` · pool max=${oracle.poolMax}`
        + (modelMb === null ? '' : ` · model=${modelMb} MB`)
        + (tight ? ' · too small to hold that model with room to work; see the ORA-04036 guidance' : '');
    });
    await check('database CPUs', async () => {
      const info = await describeOracle();
      const host = cpus().length;
      // An empty string in .env is not the same as unset to JavaScript, but it is to Oracle.
      const configured = (process.env['APP_RERANK_THREADS'] ?? '').trim();
      const matched = configured ? Number(configured) === info.cpuCount : info.cpuCount === host;
      return `cpu_count=${info.cpuCount ?? 'unknown'} · host=${host} · app threads=${configured || `default (up to ${host})`}`
        + (matched ? ' · matched' : ' · NOT matched, so a latency comparison is partly a CPU comparison');
    });
    await check('chunk table', async () => `${await countChunks()} rows`);
    await check('embedding model loaded', async () => {
      const info = await describeOracle();
      if (!info.models.includes(oracle.embedModel.toUpperCase())) {
        throw new Error(`${oracle.embedModel} missing — run \`npm run models -- --only embed\``);
      }
      return oracle.embedModel;
    });
    await check('embedding works', async () => withConnection(async (conn) => {
      const r = await conn.execute<[unknown]>(
        `SELECT VECTOR_EMBEDDING(${oracle.embedModel.toUpperCase()} USING 'hello' AS DATA) FROM DUAL`,
      );
      return r.rows?.[0] ? `returned a vector (expecting ${oracle.embedDims} dims)` : 'no row';
    }));

    // The cross-encoder is the hard half of the setup. Report its absence as a skip rather
    // than a failure, so that a run of the retrieval and application-side stages is not
    // gated on it: `npm run bench -- --rerankers none,app` works without it.
    const info = await describeOracle();
    if (!info.models.includes(oracle.rerankModel.toUpperCase())) {
      log(`  skip ${oracle.rerankModel} not loaded — in-database reranking unavailable.`);
      log(`       Everything else works: npm run bench -- --rerankers none,app`);
    } else {
      // Print what is actually in force. A scoring expression comes from .env, and the most
      // common failure is that the variable never reached the process at all, which looks
      // identical to having configured it wrongly.
      const envSet = process.env['ORACLE_INDB_SCORE_EXPR'];
      log(`  note scoring expression (${envSet ? 'from ORACLE_INDB_SCORE_EXPR' : 'built-in default; ORACLE_INDB_SCORE_EXPR is unset or empty'}):`);
      log(`       ${scoreExpr()}`);

      const mismatch = describeAttributeMismatch();
      if (mismatch) {
        failures++;
        log(`  FAIL in-DB scoring: ${mismatch.split('\n')[0]}`);
        for (const line of mismatch.split('\n').slice(1)) log(`       ${line}`);
      } else
      await check(`in-DB scoring via ${oracle.indbRerankApi}`, async () => withConnection(async (conn) => {
        const r = await conn.execute<[number]>(
          `SELECT ${scoreExpr().replace(/TITLE \|\| '\. ' \|\| CONTENT/, `'the sky is blue'`)} FROM DUAL`,
          { qtext: 'what colour is the sky' },
        );
        const score = r.rows?.[0]?.[0];
        if (typeof score !== 'number') throw new Error(`expected a number, got ${typeof score}`);
        return `scored ${score.toFixed(4)}`;
      }));
    }
  }

  if (!flag('skip-app')) {
    log('Application reranker');
    await check('model loads', async () => {
      const reranker = new AppReranker();
      await reranker.load();
      const out = await reranker.rerank('what colour is the sky', [
        { chunkId: 'a', title: 'Sky', content: 'The sky is blue.', rank: 1, score: 1 },
        { chunkId: 'b', title: 'Grass', content: 'Grass is green.', rank: 2, score: 0.5 },
      ], 2);
      await reranker.close();
      const top = out.results[0]?.chunkId;
      if (top !== 'a') throw new Error(`ranked '${top}' above the matching passage — check the score expression`);
      // The one difference that makes the benchmark meaningless is the two arms running
      // differently-quantized weights, and nothing about a successful load reveals it.
      const quantizedExists = existsSync(resolve(app.modelPath, 'onnx', 'model_quantized.onnx'));
      const usingQuantized = app.dtype === 'q8' || app.dtype === 'int8' || app.dtype === 'uint8';
      if (quantizedExists && !usingQuantized) {
        throw new Error(
          `a quantized model exists at ${app.modelPath}/onnx/model_quantized.onnx but `
          + `APP_RERANK_DTYPE=${app.dtype}. The database is scoring with the quantized weights and `
          + `this arm would not be. Set APP_RERANK_DTYPE=q8.`,
        );
      }
      return `${app.modelPath} · dtype=${app.dtype} · scored 2 pairs correctly`;
    });
  }

  log('');
  log(failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`);
  if (failures > 0) process.exitCode = 1;
}

async function cmdBench(): Promise<void> {
  const cfg = runConfigFromEnv(overridesFromFlags());
  // There is no database in fixture mode, so there is nothing for an in-database stage to
  // mean. Drop it loudly rather than failing several stages into a run.
  if (cfg.backend === 'fixture' && cfg.rerankers.includes('in-db')) {
    cfg.rerankers = cfg.rerankers.filter((r) => r !== 'in-db');
    log('Fixture backend: skipping in-database stages (they need Oracle).');
  }
  const { chunks, queries: allQueries } = readCorpus();
  const queries = cfg.queries > 0 ? allQueries.slice(0, cfg.queries) : allQueries;
  const stages = buildStages(cfg);
  const deps = buildDeps(cfg, chunks);

  if (flag('dump-sql')) {
    for (const stage of stages) {
      const sql = pipelineFor(stage, deps).sql?.();
      log(`\n${'='.repeat(78)}\n-- ${stage.id}\n${'='.repeat(78)}`);
      log(sql ?? '(fixture backend: no SQL)');
    }
    await closePool().catch(() => {});
    return;
  }

  log(`Backend: ${cfg.backend} · ${queries.length} queries · ${stages.length} stages · ${cfg.iterations} iterations (+${cfg.warmup} warmup) · ${cfg.repeats} repeat(s)`);
  log('');

  let parity;
  if (cfg.backend === 'oracle' && cfg.rerankers.includes('in-db') && cfg.rerankers.includes('app')) {
    log('Verifying both reranking paths see identical candidates...');
    parity = await verifyCandidateParity(queries, cfg, deps);
    log(parity.mismatches.length === 0
      ? `  ${parity.checked} combinations checked, all identical.`
      : `  WARNING: ${parity.mismatches.length}/${parity.checked} combinations differ. The comparison is not controlled.`);
    log('');
  }

  const run = await runBenchmark(stages, queries, cfg, deps, { onProgress: log });

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const dir = resolve(paths.results, stamp);
  mkdirSync(dir, { recursive: true });
  const payload: BenchRun & { parity?: typeof parity } = { ...run };
  if (parity) payload.parity = parity;
  writeFileSync(resolve(dir, 'raw.json'), `${JSON.stringify(payload, null, 2)}\n`);
  writeFileSync(resolve(dir, 'summary.md'), renderMarkdown(run, queries, parity, chunks));
  writeFileSync(resolve(dir, 'summary.csv'), `${renderCsv(run, queries, chunks)}\n`);
  writeFileSync(resolve(dir, 'scoring-costs.csv'), `${renderCostsCsv(run)}\n`);
  writeFileSync(resolve(dir, 'inspection.md'), renderInspection(run));

  log('');
  log(`Results written to ${dir}`);
  log('');
  log(renderMarkdown(run, queries, parity, chunks));
  await closePool().catch(() => {});
}

async function cmdReport(): Promise<void> {
  const file = opt('run') ?? args[1];
  if (!file) throw new Error('Usage: npm run report -- --run results/<stamp>/raw.json');
  const run = JSON.parse(readFileSync(resolve(file), 'utf8')) as BenchRun & { parity?: never };
  const { chunks, queries } = readCorpus();
  log(renderMarkdown(run, queries, run.parity, chunks));
}

async function cmdInspect(): Promise<void> {
  const file = opt('run') ?? args[1];
  if (!file) throw new Error('Usage: npm run inspect -- --run results/<stamp>/raw.json');
  const run = JSON.parse(readFileSync(resolve(file), 'utf8')) as BenchRun;
  log(renderInspection(run));
}

async function cmdExplain(): Promise<void> {
  const cfg = runConfigFromEnv(overridesFromFlags());
  const retrieval = (opt('retrieval') ?? 'vector') as 'vector' | 'lexical' | 'hybrid-rrf';
  if (!['vector', 'lexical', 'hybrid-rrf'].includes(retrieval)) {
    throw new Error(`--retrieval takes vector, lexical or hybrid-rrf, got ${JSON.stringify(retrieval)}.`);
  }
  const n = Number(opt('candidates') ?? 10);
  const topK = Number(opt('top-k') ?? Math.min(cfg.topK, n));
  const control = flag('control');
  const warmup = Number(opt('warmup') ?? 3);
  const { queries } = readCorpus();
  const wanted = opt('query');
  const query = wanted ? queries.find((q) => q.id === wanted) : queries[0];
  if (!query) throw new Error(`No query ${JSON.stringify(wanted)} in the corpus.`);

  const sql = new InDbReranker(cfg.rrfK).sqlFor(retrieval, control);
  const binds = usedBinds(sql, bindsFor(query, n, cfg.rrfK, topK));

  log(`${retrieval} · ${control ? 'control' : 'rerank'} · N=${n} · top-K=${topK} · ${query.id} · ${warmup} warmup`);
  log('');

  await withConnection(async (conn) => {
    // Row-source statistics are per session and off by default; without them the plan comes
    // back with estimates only, and estimates are exactly what is in question here.
    await conn.execute('ALTER SESSION SET STATISTICS_LEVEL = ALL');

    // Every invocation of this command gets a fresh session, and the first execution of a
    // statement in one loads the ONNX models it references into that session and hard-parses.
    // Left cold, a plan attributes seconds of model loading to whichever step happened to
    // touch the model first, which is exactly the step under examination. The benchmark warms
    // up before measuring; so does this. ALLSTATS LAST reports the final execution only.
    const runs: number[] = [];
    let rows: { ID: string; SCORE: number }[] = [];
    for (let i = 0; i <= warmup; i++) {
      const started = performance.now();
      const res = await conn.execute<{ ID: string; SCORE: number }>(sql, binds, {
        outFormat: oracledb.OUT_FORMAT_OBJECT,
      });
      runs.push(performance.now() - started);
      rows = res.rows ?? [];
    }
    log(`Returned ${rows.length} row(s).`);
    log(`Executions: ${runs.map((m) => `${m.toFixed(0)} ms`).join(' → ')}`);
    if (warmup > 0) {
      log(`The plan below is the last of these. The first is cold: it loads the ONNX models into`);
      log(`the session. Only the last one is comparable with the benchmark's warm figures.`);
    }
    log('');

    // DISPLAY_CURSOR with no sql_id describes the previous statement on this session, so this
    // has to run on the same connection as the statement above.
    let plan: string[] = [];
    try {
      const out = await conn.execute<{ PLAN_TABLE_OUTPUT: string }>(
        `SELECT PLAN_TABLE_OUTPUT FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(NULL, NULL, 'ALLSTATS LAST +PREDICATE'))`,
        [],
        { outFormat: oracledb.OUT_FORMAT_OBJECT },
      );
      plan = (out.rows ?? []).map((r) => r.PLAN_TABLE_OUTPUT);
    } catch (err) {
      log(`Could not read the plan: ${(err as Error).message}`);
      log('Re-run `npm run bootstrap` to pick up the V$ grants this needs, then try again.');
      return;
    }
    if (plan.length === 0 || plan.join('').includes('cannot be displayed')) {
      log('The plan came back empty. The benchmark user is probably missing SELECT on V_$SQL_PLAN,');
      log('V_$SQL_PLAN_STATISTICS_ALL, V_$SQL and V_$SESSION. Re-run `npm run bootstrap` to grant them.');
    }
    for (const line of plan) log(line);
    log('');
    // Deliberately after the plan is read: DISPLAY_CURSOR with no SQL_ID describes the
    // previous statement on this session, so any query run before it becomes the one
    // described. Running this lookup first made every plan above the lookup's own.
    // Which candidates the statement scored, and how much text each one carried. Sequence
    // length drives cross-encoder cost, so two retrieval arms scoring the same count of
    // candidates can still be doing different amounts of work. The control's SCORE is the
    // length of the scored text, because that is what the control expression computes.
    const lengths = await conn.execute<{ ID: string; LEN: number }>(
      `SELECT ID, LENGTH(TITLE) + LENGTH(CONTENT) AS LEN FROM ${oracle.schemaPrefix.toUpperCase()}_CHUNKS`
      + ` WHERE ID IN (${rows.map((_, i) => `:id${i}`).join(', ') || 'NULL'})`,
      Object.fromEntries(rows.map((r, i) => [`id${i}`, r.ID])),
      { outFormat: oracledb.OUT_FORMAT_OBJECT },
    ).catch(() => ({ rows: [] as { ID: string; LEN: number }[] }));
    const lenById = new Map((lengths.rows ?? []).map((r) => [r.ID, r.LEN]));
    const chars = rows.map((r) => lenById.get(r.ID) ?? 0);
    if (chars.length > 0) {
      const total = chars.reduce((a, b) => a + b, 0);
      log(`Scored text: ${total} characters over ${rows.length} candidate(s),`
        + ` ${Math.round(total / rows.length)} on average.`);
      log(rows.map((r, i) => `  ${r.ID} (${chars[i]} chars)`).join('\n'));
    }
    log(`A-Rows is what each step really produced. Every step should show ${n}; more would mean`);
    log('the cross-encoder scored rows the candidate limit then discarded.');
    log('');
    log('A-Time is cumulative: a step includes its children. Subtract a step from its child to');
    log('get what that step itself cost. The scoring expression is projected by the outermost');
    log('SORT ORDER BY STOPKEY, so that step minus the one below it is the cost of scoring.');
  });
  await closePool().catch(() => {});
}

function cmdHelp(): void {
  log(`rag-reranker-bench

  npm run corpus                      Generate the corpus and query set into data/
  npm run bootstrap                   Create the benchmark user (uses ORACLE_SYS_PASSWORD)
  npm run fetch:model -- <url>        Download a database-side ONNX model (unzips if needed)
  npm run export:app-model            Export the application-side cross-encoder to ONNX
  npm run augment:rerank-model        Add a tokenizer to that export so the database can load it
  npm run load                        Create the schema and load the corpus into Oracle
  npm run models                      Load both ONNX models into the database
  npm run models:embed                Load only the embedding model
  npm run models:rerank               Load only the cross-encoder
  npm run doctor                      Check Oracle, the models, and the app reranker
  npm run bench                       Run the benchmark
  npm run report -- --run <raw.json>  Re-render a report from a previous run
  npm run inspect -- --run <raw.json> Per-iteration forensics: candidates scored, outliers
  npm run explain -- --retrieval vector --candidates 10
                                      Execution plan of one in-database rerank, with real row counts

Flags for bench:
  --backend oracle|fixture   Where retrieval and reranking run (default: oracle)
  --retrievals a,b,c         vector, lexical, hybrid-rrf
  --rerankers a,b,c          none, in-db, app
  --candidates 10,20,40,80   Candidate pool sizes to sweep
  --top-k 10                 Results returned to the model
  --iterations 20            Measured iterations per query per stage
  --warmup 5                 Unmeasured iterations per query per stage
  --queries N                Use only the first N queries
  --repeats R                Full repetitions of the protocol; report shows between-repeat spread
  --quiesce-ms 5000          Idle time after each reset between isolation batches
  --reset-cmd "<shell>"      Command to run at each reset, e.g. "docker compose restart oracle && sleep 90"
  --dump-sql                 Print the rendered SQL for every stage and exit

Flags for doctor:
  --skip-oracle              Skip the database checks
  --skip-app                 Skip the application reranker checks
`);
}

const commands: Record<string, () => Promise<void> | void> = {
  corpus: cmdCorpus,
  bootstrap: cmdBootstrap,
  'fetch-model': cmdFetchModel,
  'export-app-model': cmdExportAppModel,
  'augment-rerank-model': cmdAugmentRerankModel,
  load: cmdLoad,
  models: cmdModels,
  doctor: cmdDoctor,
  bench: cmdBench,
  report: cmdReport,
  inspect: cmdInspect,
  explain: cmdExplain,
  help: cmdHelp,
};

const handler = commands[command];
if (!handler) {
  log(`Unknown command: ${command}`);
  cmdHelp();
  process.exit(1);
}

/**
 * Leave deliberately rather than waiting for the event loop to drain.
 *
 * Two things here outlive their owner: the Oracle connection pool holds `poolMin` connections
 * open by design, and ONNX Runtime keeps native threads alive after the model is disposed.
 * Neither is a leak that matters inside a command, but both keep Node from exiting, which
 * looks to the person running it like the CLI has hung. Closing the pool centrally also means
 * a command added later cannot forget to.
 */
async function finish(code: number): Promise<never> {
  await closePool().catch(() => {});
  // process.exit can truncate output that is still buffered, which matters when stdout is a
  // pipe rather than a terminal, so flush before leaving.
  await new Promise<void>((done) => {
    if (process.stdout.writableLength === 0) done();
    else process.stdout.write('', () => done());
  });
  process.exit(code);
}

try {
  await handler();
  await finish(process.exitCode === undefined ? 0 : Number(process.exitCode));
} catch (err) {
  const message = (err as Error).message;
  log('');
  log(`Error: ${message}${hintFor(message)}`);
  await finish(1);
}
