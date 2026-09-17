import { inflateRawSync } from 'node:zlib';

/**
 * Minimal ZIP reader: enough to pull files out of a model archive, and nothing more.
 *
 * Written rather than taken as a dependency because it is the only archive handling in the
 * repo and the alternatives are platform-specific (PowerShell's Expand-Archive, bsdtar's zip
 * support, GNU tar's lack of it). Only the two compression methods that occur in practice are
 * supported - stored and deflate - and anything else is reported rather than silently skipped.
 */

export interface ZipEntry {
  name: string;
  data: Buffer;
}

const EOCD_SIG = 0x06054b50;
const CENTRAL_SIG = 0x02014b50;
const LOCAL_SIG = 0x04034b50;

/** Locate the end-of-central-directory record, which sits at the end, after any comment. */
function findEocd(buf: Buffer): number {
  // The comment is at most 0xffff bytes, plus the 22-byte record itself.
  const from = Math.max(0, buf.length - (0xffff + 22));
  for (let i = buf.length - 22; i >= from; i--) {
    if (buf.readUInt32LE(i) === EOCD_SIG) return i;
  }
  throw new Error('Not a ZIP archive: no end-of-central-directory record found.');
}

export function listZip(buf: Buffer): string[] {
  return readZip(buf, () => false).names;
}

/**
 * Read entries whose name passes `wanted`. Entries are decompressed eagerly, so the filter
 * exists to avoid materialising an archive's worth of data to extract one file from it.
 */
export function readZip(
  buf: Buffer,
  wanted: (name: string) => boolean,
): { entries: ZipEntry[]; names: string[] } {
  const eocd = findEocd(buf);
  const count = buf.readUInt16LE(eocd + 10);
  let offset = buf.readUInt32LE(eocd + 16);

  if (offset === 0xffffffff || count === 0xffff) {
    throw new Error('ZIP64 archives are not supported. Extract this one with your platform tools.');
  }

  const entries: ZipEntry[] = [];
  const names: string[] = [];

  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(offset) !== CENTRAL_SIG) {
      throw new Error(`Corrupt ZIP: expected a central directory entry at byte ${offset}.`);
    }
    const method = buf.readUInt16LE(offset + 10);
    const compressedSize = buf.readUInt32LE(offset + 20);
    const nameLen = buf.readUInt16LE(offset + 28);
    const extraLen = buf.readUInt16LE(offset + 30);
    const commentLen = buf.readUInt16LE(offset + 32);
    const localOffset = buf.readUInt32LE(offset + 42);
    const name = buf.toString('utf8', offset + 46, offset + 46 + nameLen);
    names.push(name);

    if (wanted(name) && !name.endsWith('/')) {
      if (buf.readUInt32LE(localOffset) !== LOCAL_SIG) {
        throw new Error(`Corrupt ZIP: no local header for ${name}.`);
      }
      // The local header's name and extra lengths can differ from the central directory's.
      const localNameLen = buf.readUInt16LE(localOffset + 26);
      const localExtraLen = buf.readUInt16LE(localOffset + 28);
      const start = localOffset + 30 + localNameLen + localExtraLen;
      const raw = buf.subarray(start, start + compressedSize);
      if (method === 0) entries.push({ name, data: Buffer.from(raw) });
      else if (method === 8) entries.push({ name, data: inflateRawSync(raw) });
      else throw new Error(`${name}: unsupported ZIP compression method ${method}.`);
    }
    offset += 46 + nameLen + extraLen + commentLen;
  }
  return { entries, names };
}
