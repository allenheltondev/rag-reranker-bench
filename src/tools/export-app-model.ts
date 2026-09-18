import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, renameSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * Export the application-side cross-encoder to ONNX.
 *
 * This is the copy ONNX Runtime loads in the application arm. It is a plain export - the
 * tokenizer stays a separate tokenizer.json - which is the opposite of what the database
 * needs, where the tokenizer has to be inside the graph. Both must come from the same
 * checkpoint or the benchmark compares two models instead of two execution locations.
 *
 * Driven from Node rather than a shell script because the benchmark runs on Windows as well
 * as Linux, and venv layouts, executable suffixes and shell quoting all differ.
 */

const isWindows = process.platform === 'win32';
const venvDir = resolve('.venv-export');

/** Where a venv puts executables, which is the one thing that always differs. */
const venvBin = (name: string): string =>
  resolve(venvDir, isWindows ? 'Scripts' : 'bin', isWindows ? `${name}.exe` : name);

function run(cmd: string, args: string[], label: string): void {
  const res = spawnSync(cmd, args, { stdio: 'inherit' });
  if (res.error) throw new Error(`${label} could not start (${cmd}): ${res.error.message}`);
  if (res.status !== 0) throw new Error(`${label} failed with exit code ${res.status}.`);
}

export function pythonVersion(candidate: readonly string[]): string | null {
  const [cmd, ...rest] = candidate;
  const res = spawnSync(cmd!, [...rest, '--version'], { stdio: 'pipe', encoding: 'utf8' });
  if (res.status !== 0) return null;
  const text = `${res.stdout ?? ''}${res.stderr ?? ''}`.trim();
  return /Python (\d+\.\d+)/.exec(text)?.[1] ?? null;
}

/**
 * Find a usable Python, optionally restricted to versions a dependency has wheels for.
 *
 * Whatever `python` happens to be on PATH is often newer than the compiled wheels a package
 * publishes, and pip's message for that case ("no matching distribution ... from versions:
 * none") does not mention the version at all. On Windows the `py` launcher can select an
 * older interpreter explicitly, so ask it for one before falling back.
 */
export function findPython(allowed?: readonly string[]): string[] {
  const explicit: string[][] = isWindows && allowed
    ? allowed.map((v) => ['py', `-${v}`])
    : [];
  const generic: string[][] = isWindows
    ? [['py', '-3'], ['python'], ['python3']]
    : [['python3'], ['python']];

  for (const candidate of [...explicit, ...generic]) {
    const version = pythonVersion(candidate);
    if (version === null) continue;
    if (allowed && !allowed.includes(version)) continue;
    return candidate;
  }

  const found = generic.map((c) => `${c.join(' ')} -> ${pythonVersion(c) ?? 'not found'}`);
  throw new Error(
    (allowed
      ? `No Python ${allowed.join(' / ')} found, and the dependency publishes wheels only for those.\n`
      : 'No Python 3 found.\n')
    + `Interpreters tried: ${found.join(', ')}\n`
    + (isWindows
      ? 'Install one from python.org; the `py` launcher will then be able to select it.'
      : 'Install one with your package manager.'),
  );
}

export interface ExportOptions {
  model: string;
  outDir: string;
  log: (msg: string) => void;
}

export function exportAppModel({ model, outDir, log }: ExportOptions): void {
  const out = resolve(outDir);

  if (!existsSync(venvBin('python'))) {
    const python = findPython();
    log(`Creating an isolated Python environment in ${venvDir} ...`);
    run(python[0]!, [...python.slice(1), '-m', 'venv', venvDir], 'venv creation');
  } else {
    log(`Reusing the Python environment in ${venvDir}`);
  }

  const py = venvBin('python');
  log('Installing the exporter (optimum-onnx, onnx, onnxruntime) ...');
  run(py, ['-m', 'pip', 'install', '--quiet', '--upgrade', 'pip'], 'pip upgrade');
  // The ONNX exporter lives in optimum-onnx, not in optimum itself: the extras that used to
  // provide it (`optimum[exporters]`) were dropped in optimum 2.x, and without this package
  // `optimum-cli export onnx` is not a registered subcommand at all.
  run(py, ['-m', 'pip', 'install', '--quiet', 'optimum-onnx', 'onnx', 'onnxruntime'], 'pip install');

  // The exact versions matter for reproducing a published number, so record them where the
  // person running the export will see them.
  const versions = spawnSync(py, ['-m', 'pip', 'list'], { stdio: 'pipe', encoding: 'utf8' });
  const relevant = (versions.stdout ?? '')
    .split('\n')
    .filter((line) => /^(optimum|optimum-onnx|onnx|onnxruntime|transformers|torch)\s/.test(line));
  if (relevant.length > 0) {
    log('Exporter versions:');
    for (const line of relevant) log(`  ${line.trim()}`);
  }

  log(`Exporting ${model} to ${out} ...`);
  mkdirSync(out, { recursive: true });
  const cli = venvBin('optimum-cli');
  // The console script is the documented interface; the module form is the fallback for
  // installs that do not put scripts on the venv's path.
  const exportArgs = ['--model', model, '--task', 'text-classification', '--opset', '17', out];
  try {
    if (existsSync(cli)) run(cli, ['export', 'onnx', ...exportArgs], 'ONNX export');
    else run(py, ['-m', 'optimum.exporters.onnx', ...exportArgs], 'ONNX export');
  } catch (err) {
    throw new Error(
      `${(err as Error).message}\n\n`
      + 'Two things usually cause this. If the output mentions huggingface.co, the download was\n'
      + 'blocked - check your network or proxy. If it says "unrecognized arguments", the exporter\n'
      + `package did not install; run "${py} -m pip install optimum-onnx" and try again.`,
    );
  }

  // transformers.js looks for the graph under onnx/, beside config.json and tokenizer.json.
  const onnxDir = resolve(out, 'onnx');
  mkdirSync(onnxDir, { recursive: true });
  for (const file of ['model.onnx', 'model.onnx_data']) {
    const from = resolve(out, file);
    if (existsSync(from)) renameSync(from, resolve(onnxDir, file));
  }

  const required = ['config.json', 'tokenizer.json', 'onnx/model.onnx'];
  const missing = required.filter((f) => !existsSync(resolve(out, f)));
  if (missing.length > 0) {
    throw new Error(
      `The export finished but ${missing.join(', ')} ${missing.length === 1 ? 'is' : 'are'} missing from ${out}.\n`
      + 'The application reranker cannot load without them.',
    );
  }
  log('');
  log(`Exported to ${out}`);
  log('Verify it with: npm run doctor -- --skip-oracle');
}
