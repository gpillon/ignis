// `node --test connector/`: runs the skill's transcribe script as a process
// against a node:http stub of ignis's POST /v1/audio/transcriptions. Node 18+,
// no dependencies.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { run as runScript, stub } from './stub.mjs';

const SCRIPT = join(dirname(fileURLToPath(import.meta.url)), 'ignis', 'scripts', 'transcribe.mjs');
const run = (args, env) => runScript(SCRIPT, args, env);
const AUDIO = Buffer.from('RIFF....WAVEfmt fake audio bytes');

async function recording() {
  const dir = await mkdtemp(join(tmpdir(), 'ignis-skill-'));
  const file = join(dir, 'meeting.wav');
  await writeFile(file, AUDIO);
  return { dir, file };
}

const said = () => [200, { text: 'hello from ignis', usage: { type: 'duration', seconds: 1 } }];

test('prints the transcript, sending the file as a multipart `file` part', async () => {
  const server = await stub(said);
  const { file } = await recording();
  const { code, stdout } = await run([file], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.equal(code, 0);
  assert.equal(stdout, 'hello from ignis\n');
  const [request] = server.requests;
  assert.equal(request.method, 'POST');
  assert.equal(request.url, '/v1/audio/transcriptions');
  assert.match(request.headers['content-type'], /^multipart\/form-data; boundary=/);
  const body = request.body.toString('latin1');
  assert.match(body, /name="file"; filename="meeting\.wav"/);
  assert.ok(request.body.includes(AUDIO), 'the audio bytes arrive whole');
});

test('--out writes the transcript to the file and prints nothing', async () => {
  const server = await stub(said);
  const { dir, file } = await recording();
  const out = join(dir, 'said.txt');
  const { code, stdout } = await run([file, '--out', out], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.equal(code, 0);
  assert.equal(stdout, '');
  assert.equal(await readFile(out, 'utf8'), 'hello from ignis\n');
});

test('IGNIS_BASE_URL unset means http://127.0.0.1:8000', async (t) => {
  // The default port may be a running ignis's: the test needs it free.
  const server = http.createServer((req, res) => {
    req.resume();
    req.on('end', () => res.end(JSON.stringify({ text: 'from the default' })));
  });
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(8000, '127.0.0.1', resolve);
    });
  } catch {
    return t.skip('127.0.0.1:8000 is in use');
  }
  const { file } = await recording();
  const { code, stdout } = await run([file], {});
  await new Promise((resolve) => server.close(resolve));
  assert.equal(code, 0);
  assert.equal(stdout, 'from the default\n');
});

test('a missing audio file is named, with a non-zero exit', async () => {
  const server = await stub(said);
  const { code, stderr } = await run(['/no/such/recording.wav'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.match(stderr, /\/no\/such\/recording\.wav/);
  assert.equal(server.requests.length, 0, 'nothing was sent');
});

test('no audio argument is an error with the usage', async () => {
  const { code, stderr } = await run([], {});
  assert.notEqual(code, 0);
  assert.match(stderr, /usage: transcribe\.mjs <audio>/);
});

test('an unreachable server is named, with a non-zero exit', async () => {
  const closed = await stub(said);
  const url = closed.url;
  await closed.close();
  const { file } = await recording();
  const { code, stderr } = await run([file], { IGNIS_BASE_URL: url });
  assert.notEqual(code, 0);
  assert.ok(stderr.includes(url), stderr);
  assert.match(stderr, /cannot reach ignis/);
});

test('a server error prints the server JSON, with a non-zero exit', async () => {
  const refusal = { error: { message: 'the audio could not be decoded', type: 'invalid_request_error', code: 'invalid_audio' } };
  const server = await stub(() => [400, refusal]);
  const { file } = await recording();
  const { code, stdout, stderr } = await run([file], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.equal(stdout, '');
  assert.match(stderr, /400/);
  assert.ok(stderr.includes(JSON.stringify(refusal)), stderr);
});

test('the key is sent as a bearer only when set', async () => {
  const server = await stub(said);
  const { file } = await recording();
  await run([file], { IGNIS_BASE_URL: server.url });
  await run([file], { IGNIS_BASE_URL: server.url, IGNIS_API_KEY: 'sk-test-key' });
  await run([file], { IGNIS_BASE_URL: server.url, IGNIS_API_KEY: '' });
  await server.close();
  const sent = server.requests.map((request) => request.headers.authorization);
  assert.deepEqual(sent, [undefined, 'Bearer sk-test-key', undefined]);
});

test('a trailing slash on IGNIS_BASE_URL does not double the path', async () => {
  const server = await stub(said);
  const { file } = await recording();
  await run([file], { IGNIS_BASE_URL: `${server.url}/` });
  await server.close();
  assert.equal(server.requests[0].url, '/v1/audio/transcriptions');
});

test('a 401 with IGNIS_API_KEY unset says the variable is unset, beside the server JSON', async () => {
  const refusal = { error: { message: 'invalid API key', type: 'invalid_request_error', code: 'invalid_api_key' } };
  const server = await stub(() => [401, refusal]);
  const { file } = await recording();
  const unset = await run([file], { IGNIS_BASE_URL: server.url });
  const wrong = await run([file], { IGNIS_BASE_URL: server.url, IGNIS_API_KEY: 'sk-wrong' });
  await server.close();
  assert.notEqual(unset.code, 0);
  assert.match(unset.stderr, /IGNIS_API_KEY is not set/);
  assert.ok(unset.stderr.includes(JSON.stringify(refusal)), unset.stderr);
  assert.notEqual(wrong.code, 0);
  assert.doesNotMatch(wrong.stderr, /IGNIS_API_KEY/);
  assert.ok(wrong.stderr.includes(JSON.stringify(refusal)), wrong.stderr);
});
