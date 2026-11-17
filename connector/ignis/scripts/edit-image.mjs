#!/usr/bin/env node
// Edits pictures with ignis: POST /v1/images/edits (multipart), the full-size
// PNG written to a file.
//
//   node edit-image.mjs <prompt> <out.png> <input...> [--model id]
//
// Each input is a PNG, JPEG or WebP file, sent in order as an `image[]` part.
// Node 18 or later, no dependencies. IGNIS_BASE_URL (default
// http://127.0.0.1:8000) says where ignis is; IGNIS_API_KEY is sent as a
// bearer key only when set. Without --model the image model is picked from
// the server's catalog. Exit 0 on success, 1 on any failure, the reason on
// stderr: a server's error is printed as the JSON it sent.

import { readFile } from 'node:fs/promises';
import { basename, extname } from 'node:path';
import { call, fail, parseArgs, pickImageModel, refuse, saveImage } from './lib.mjs';

const TOOL = 'edit-image';
const USAGE = 'usage: edit-image.mjs <prompt> <out.png> <input...> [--model id]';
const TYPES = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' };

const { options, positional } = parseArgs(TOOL, USAGE, process.argv.slice(2), ['--model']);
if (positional.length < 3) fail(TOOL, `a prompt, an output file and at least one input image are needed\n${USAGE}`);
const [prompt, out, ...inputs] = positional;

const form = new FormData();
for (const input of inputs) {
  let bytes;
  try {
    bytes = await readFile(input);
  } catch (error) {
    fail(TOOL, `cannot read ${input}: ${error.code === 'ENOENT' ? 'no such file' : error.message}`);
  }
  const type = TYPES[extname(input).toLowerCase()] ?? 'application/octet-stream';
  form.append('image[]', new Blob([bytes], { type }), basename(input));
}

const model = options['--model'] ?? (await pickImageModel(TOOL));
form.append('model', model);
form.append('prompt', prompt);
form.append('response_format', 'b64_json');

const answer = await call(TOOL, 'POST', '/v1/images/edits', { form });
refuse(TOOL, answer);
await saveImage(TOOL, answer, out);
