import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { listZip, readZip } from '../src/tools/unzip.js';

/** Build a real archive with Python's zipfile rather than a hand-rolled fixture. */
function makeZip(files: Record<string, string>, compress = true): Buffer {
  const dir = mkdtempSync(join(tmpdir(), 'unzip-test-'));
  const zipPath = join(dir, 'a.zip');
  const script = `
import zipfile, json, sys
files = json.loads(sys.argv[1])
mode = zipfile.ZIP_DEFLATED if sys.argv[3] == 'y' else zipfile.ZIP_STORED
with zipfile.ZipFile(sys.argv[2], 'w', mode) as z:
    for name, body in files.items():
        z.writestr(name, body)
`;
  const scriptPath = join(dir, 's.py');
  writeFileSync(scriptPath, script);
  execFileSync('python3', [scriptPath, JSON.stringify(files), zipPath, compress ? 'y' : 'n']);
  return readFileSync(zipPath);
}

test('lists every entry without decompressing', () => {
  const zip = makeZip({ 'a.txt': 'hello', 'sub/b.onnx': 'model bytes' });
  assert.deepEqual(listZip(zip).sort(), ['a.txt', 'sub/b.onnx']);
});

test('extracts deflated entries matching the filter', () => {
  // Repetitive content so deflate actually compresses and the deflate path is exercised.
  const body = 'onnx'.repeat(5000);
  const zip = makeZip({ 'readme.txt': 'ignore me', 'model/all_MiniLM_L12_v2.onnx': body });
  const { entries } = readZip(zip, (n: string) => n.endsWith('.onnx'));
  assert.equal(entries.length, 1);
  assert.equal(entries[0]!.name, 'model/all_MiniLM_L12_v2.onnx');
  assert.equal(entries[0]!.data.toString('utf8'), body);
});

test('extracts stored (uncompressed) entries', () => {
  const zip = makeZip({ 'x.onnx': 'stored payload' }, false);
  const { entries } = readZip(zip, () => true);
  assert.equal(entries[0]!.data.toString('utf8'), 'stored payload');
});

test('skips directory entries even when the filter accepts them', () => {
  const zip = makeZip({ 'dir/': '', 'dir/x.onnx': 'body' });
  const { entries } = readZip(zip, () => true);
  assert.deepEqual(entries.map((e: { name: string }) => e.name), ['dir/x.onnx']);
});

test('rejects data that is not a ZIP archive', () => {
  assert.throws(() => listZip(Buffer.from('this is an onnx file, not a zip')), /Not a ZIP archive/);
});
