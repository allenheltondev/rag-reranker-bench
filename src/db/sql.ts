import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { oracle } from '../config.js';
import { assertIdentifier } from './oracle.js';

const sqlDir = resolve(dirname(fileURLToPath(import.meta.url)), '../../sql');

/**
 * Substitute ${TOKEN} placeholders, and refuse to return SQL that still has one.
 *
 * Comment lines are left exactly as written. A template's header comment names its own
 * placeholders, and substituting a multi-line SQL fragment into that sentence turns the
 * documentation into noise - which matters here, because `--dump-sql` output is meant to be
 * read by a person.
 */
export function render(template: string, tokens: Record<string, string>): string {
  const lines = template.split('\n');
  const rendered = lines.map((line) =>
    line.trim().startsWith('--')
      ? line
      : line.replace(/\$\{([A-Z_]+)\}/g, (match, name: string) => tokens[name] ?? match),
  );
  const leftover = rendered
    .filter((line) => !line.trim().startsWith('--'))
    .join('\n')
    .match(/\$\{[A-Z_]+\}/g);
  if (leftover) {
    throw new Error(`Unsubstituted SQL tokens: ${[...new Set(leftover)].join(', ')}`);
  }
  return rendered.join('\n');
}

/**
 * The expression handed to VECTOR_EMBEDDING for the query.
 *
 * A model needing an instruction prefix gets it concatenated in SQL rather than bound, so the
 * rendered statement shows exactly what was embedded. Single quotes are doubled; the prefix
 * comes from .env, not from anything a query can influence.
 */
export function queryEmbedInput(): string {
  const prefix = oracle.embedQueryPrefix;
  return prefix ? `'${prefix.replace(/'/g, "''")}' || :qtext` : ':qtext';
}

/** Tokens every template shares, derived from config with identifiers validated. */
export function baseTokens(): Record<string, string> {
  return {
    QUERY_EMBED_INPUT: queryEmbedInput(),
    PREFIX: assertIdentifier(oracle.schemaPrefix, 'ORACLE_SCHEMA_PREFIX'),
    EMBED_MODEL: assertIdentifier(oracle.embedModel, 'ORACLE_EMBED_MODEL'),
    RERANK_MODEL: assertIdentifier(oracle.rerankModel, 'ORACLE_RERANK_MODEL'),
    DIMS: String(oracle.embedDims),
  };
}

export function loadSql(name: string, extra: Record<string, string> = {}): string {
  const raw = readFileSync(resolve(sqlDir, name), 'utf8');
  return render(raw, { ...baseTokens(), ...extra });
}

/**
 * Split a script into executable statements on lines containing only "/", the SQL*Plus
 * convention. Comment-only fragments are dropped so an empty statement never reaches the
 * server.
 */
export function splitStatements(script: string): string[] {
  return script
    .split(/^\s*\/\s*$/m)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .filter((s) => s.split('\n').some((line) => line.trim() !== '' && !line.trim().startsWith('--')));
}

/**
 * Keep only the binds the statement actually references.
 *
 * The retrieval strategies are the same template with arms swapped out, so a vector-only
 * statement has no lexical subquery and never mentions :contains. node-oracledb's thin mode
 * rejects a bind that the SQL does not use, so the bind set has to follow the rendered text
 * rather than the caller's intent.
 *
 * Deriving it from the SQL rather than branching on the strategy means an arm can be edited
 * without remembering to update a bind list somewhere else. Comments are stripped first
 * because the templates document their own binds in a header ("Binds: :qtext :contains ..."),
 * and string literals are stripped because an embedding instruction prefix may contain a colon.
 */
export function usedBinds<T extends Record<string, unknown>>(sql: string, binds: T): Partial<T> {
  const executable = sql
    .split('\n')
    .filter((line) => !line.trim().startsWith('--'))
    .join('\n')
    .replace(/'(?:[^']|'')*'/g, "''");

  const used = new Set<string>();
  for (const match of executable.matchAll(/:([A-Za-z][A-Za-z0-9_]*)/g)) {
    used.add(match[1]!.toLowerCase());
  }
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(binds)) {
    if (used.has(key.toLowerCase())) out[key] = value;
  }
  return out as Partial<T>;
}
