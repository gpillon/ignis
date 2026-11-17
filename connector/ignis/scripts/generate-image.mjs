#!/usr/bin/env node
// Generates an image with ignis: POST /v1/images/generations, the full-size
// PNG written to a file.
//
//   node generate-image.mjs <prompt> <out.png> [--size WxH] [--model id]
//
// Node 18 or later, no dependencies. IGNIS_BASE_URL (default
// http://127.0.0.1:8000) says where ignis is; IGNIS_API_KEY is sent as a
// bearer key only when set. Without --model the image model is picked from
// the server's catalog. Exit 0 on success, 1 on any failure, the reason on
// stderr: a server's error is printed as the JSON it sent.

import { call, fail, parseArgs, pickImageModel, refuse, saveImage } from './lib.mjs';

const TOOL = 'generate-image';
const USAGE = 'usage: generate-image.mjs <prompt> <out.png> [--size WxH] [--model id]';

const { options, positional } = parseArgs(TOOL, USAGE, process.argv.slice(2), ['--size', '--model']);
if (positional.length < 2) fail(TOOL, `a prompt and an output file are needed\n${USAGE}`);
if (positional.length > 2) fail(TOOL, `one prompt and one output file only (quote the prompt)\n${USAGE}`);
const [prompt, out] = positional;
const size = options['--size'];
if (size !== undefined && !/^(auto|\d+x\d+)$/.test(size)) {
  fail(TOOL, `--size is WxH, such as 1024x1024, or auto\n${USAGE}`);
}

const model = options['--model'] ?? (await pickImageModel(TOOL));
const body = { model, prompt, response_format: 'b64_json' };
if (size !== undefined) body.size = size;

const answer = await call(TOOL, 'POST', '/v1/images/generations', { json: body });
refuse(TOOL, answer);
await saveImage(TOOL, answer, out);
