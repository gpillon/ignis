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
import { call, refuse } from './lib.mjs';

const USAGE = 'usage: transcribe.mjs <audio> [--out file]';

function fail(message) {
  process.stderr.write(`transcribe: ${message}\n`);
  process.exit(1);
}

let audio;
let out;
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--out') {
    out = args[++i];
    if (out === undefined) fail(`--out needs a file\n${USAGE}`);
  } else if (args[i].startsWith('--')) {
    fail(`unknown option ${args[i]}\n${USAGE}`);
  } else if (audio === undefined) {
    audio = args[i];
  } else {
    fail(`one audio file at a time\n${USAGE}`);
  }
}
if (audio === undefined) fail(`no audio file given\n${USAGE}`);

let bytes;
try {
  bytes = await readFile(audio);
} catch (error) {
  fail(`cannot read ${audio}: ${error.code === 'ENOENT' ? 'no such file' : error.message}`);
}

const form = new FormData();
form.append('file', new Blob([bytes]), basename(audio));

const answer = await call('transcribe', 'POST', '/v1/audio/transcriptions', { form });
refuse('transcribe', answer);
const body = answer.body.toString('utf8');

let text;
try {
  text = JSON.parse(body).text;
} catch {
  text = undefined;
}
if (typeof text !== 'string') fail(`ignis's answer holds no transcript:\n${body.trim()}`);

if (out !== undefined) {
  try {
    await writeFile(out, `${text}\n`);
  } catch (error) {
    fail(`cannot write ${out}: ${error.message}`);
  }
} else {
  process.stdout.write(`${text}\n`);
}
