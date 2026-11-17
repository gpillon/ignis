#!/usr/bin/env node
// Transcribes an audio file with ignis: POST /v1/audio/transcriptions.
//
//   node transcribe.mjs <audio> [--out file]
//
// Node 18 or later, no dependencies. IGNIS_BASE_URL (default
// http://127.0.0.1:8000) says where ignis is; IGNIS_API_KEY is sent as a
// bearer key only when set. The transcript goes to stdout, or to --out.
// Exit 0 on success, 1 on any failure, the reason on stderr: a server's
// error is printed as the JSON it sent.

import { readFile, writeFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { call, fail, parseArgs, refuse } from './lib.mjs';

const TOOL = 'transcribe';
const USAGE = 'usage: transcribe.mjs <audio> [--out file]';

const { options, positional } = parseArgs(TOOL, USAGE, process.argv.slice(2), ['--out']);
if (positional.length === 0) fail(TOOL, `no audio file given\n${USAGE}`);
if (positional.length > 1) fail(TOOL, `one audio file at a time\n${USAGE}`);
const [audio] = positional;
const out = options['--out'];

let bytes;
try {
  bytes = await readFile(audio);
} catch (error) {
  fail(TOOL, `cannot read ${audio}: ${error.code === 'ENOENT' ? 'no such file' : error.message}`);
}

const form = new FormData();
form.append('file', new Blob([bytes]), basename(audio));

const answer = await call(TOOL, 'POST', '/v1/audio/transcriptions', { form });
refuse(TOOL, answer);
const body = answer.body.toString('utf8');

let text;
try {
  text = JSON.parse(body).text;
} catch {
  text = undefined;
}
if (typeof text !== 'string') fail(TOOL, `ignis's answer holds no transcript:\n${body.trim()}`);

if (out !== undefined) {
  try {
    await writeFile(out, `${text}\n`);
  } catch (error) {
    fail(TOOL, `cannot write ${out}: ${error.message}`);
  }
} else {
  process.stdout.write(`${text}\n`);
}
