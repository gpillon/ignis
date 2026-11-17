// What the skill's scripts share: ignis's address and key, one HTTP call, the
// image model's pick from the catalog, and the way a script fails.
//
// The call is built on node:http and node:https, not fetch: fetch (undici)
// gives up on a response whose headers take more than about 300 seconds, which
// a long transcription on the CPU or a queued image job outlasts. node:http
// has no such limit.

import http from 'node:http';
import https from 'node:https';
import { writeFile } from 'node:fs/promises';

/** Prints `<tool>: <message>` on stderr and exits 1. */
export function fail(tool, message) {
  process.stderr.write(`${tool}: ${message}\n`);
  process.exit(1);
}

/** Where ignis is (IGNIS_BASE_URL, no trailing slash) and its key, if any. */
export function config() {
  const base = (process.env.IGNIS_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
  return { base, key: process.env.IGNIS_API_KEY };
}

function unreachable(tool, base, error) {
  const why = error.code || error.errors?.[0]?.code || error.message;
  fail(tool, `cannot reach ignis at ${base} (${why}); set IGNIS_BASE_URL to where it runs`);
}

/**
 * One request to ignis: `{ status, body }` with `body` a Buffer. `json` is
 * sent as a JSON body, `form` (a FormData) as multipart. An unreachable
 * server fails the script, naming the address.
 */
export async function call(tool, method, path, { json, form } = {}) {
  const { base, key } = config();
  const headers = {};
  if (key) headers.Authorization = `Bearer ${key}`;
  let payload;
  if (form !== undefined) {
    // A Response serialises a FormData into its body and its content-type.
    const serialised = new Response(form);
    headers['content-type'] = serialised.headers.get('content-type');
    payload = Buffer.from(await serialised.arrayBuffer());
  } else if (json !== undefined) {
    headers['content-type'] = 'application/json';
    payload = Buffer.from(JSON.stringify(json));
  }
  if (payload !== undefined) headers['content-length'] = payload.length;
  let url;
  try {
    url = new URL(`${base}${path}`);
  } catch {
    return fail(tool, `IGNIS_BASE_URL is not a URL: ${base}`);
  }
  const transport = url.protocol === 'https:' ? https : http;
  return new Promise((resolve) => {
    const request = transport.request(url, { method, headers }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolve({ status: response.statusCode, body: Buffer.concat(chunks) }));
      response.on('error', (error) => unreachable(tool, base, error));
    });
    request.on('error', (error) => unreachable(tool, base, error));
    request.end(payload);
  });
}

/**
 * Exits 1 with the server's own JSON when `answer` is not a 2xx; a 401 with
 * IGNIS_API_KEY unset says so too, since the key is then what is missing.
 */
export function refuse(tool, answer) {
  if (answer.status >= 200 && answer.status < 300) return;
  process.stderr.write(`${tool}: ignis answered ${answer.status}\n${answer.body.toString('utf8').trim()}\n`);
  if (answer.status === 401 && !config().key) {
    process.stderr.write(`${tool}: IGNIS_API_KEY is not set: set it to the server's API key\n`);
  }
  process.exit(1);
}

/** The answer's body as JSON, or a failure naming what was received. */
export function parse(tool, answer) {
  try {
    return JSON.parse(answer.body.toString('utf8'));
  } catch {
    return fail(tool, `ignis's answer is not JSON:\n${answer.body.toString('utf8').trim()}`);
  }
}

/**
 * The id of the image model to ask, from GET /v1/models/catalog: the one
 * loaded if it is an image model, else the first image model complete on
 * disk this card can load. The server tool generate_image picks by the same
 * rule.
 */
export async function pickImageModel(tool) {
  const answer = await call(tool, 'GET', '/v1/models/catalog');
  refuse(tool, answer);
  const rows = parse(tool, answer).data;
  if (!Array.isArray(rows)) {
    return fail(tool, `ignis's catalog holds no list of models:\n${answer.body.toString('utf8').trim()}`);
  }
  const images = rows.filter((row) => row.kind === 'image');
  const pick = images.find((row) => row.loaded) || images.find((row) => row.on_disk === 'complete' && row.available !== false);
  if (pick === undefined) {
    return fail(tool, 'ignis has no image model on disk; download one first, or name one with --model');
  }
  return pick.id;
}

/** Writes the first image of an images answer to `out`, and says so. */
export async function saveImage(tool, answer, out) {
  const body = parse(tool, answer);
  const b64 = body?.data?.[0]?.b64_json;
  if (typeof b64 !== 'string') {
    return fail(tool, `ignis's answer holds no image:\n${answer.body.toString('utf8').trim()}`);
  }
  try {
    await writeFile(out, Buffer.from(b64, 'base64'));
  } catch (error) {
    return fail(tool, `cannot write ${out}: ${error.message}`);
  }
  for (const warning of body.warnings || []) process.stderr.write(`${tool}: ${warning}\n`);
  process.stdout.write(`wrote ${out} (${body.size}, seed ${body.seed})\n`);
}

/** Splits `args` into the options named in `values` (each takes a value) and the positional rest. */
export function parseArgs(tool, usage, args, values) {
  const options = {};
  const positional = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
      const name = args[i];
      if (!values.includes(name)) fail(tool, `unknown option ${name}\n${usage}`);
      options[name] = args[++i];
      if (options[name] === undefined) fail(tool, `${name} needs a value\n${usage}`);
    } else {
      positional.push(args[i]);
    }
  }
  return { options, positional };
}
