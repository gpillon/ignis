// `node --test` in connector/: runs the skill's generate-image and edit-image
// scripts as processes against a node:http stub of ignis's catalog and image
// routes. Node 18+, no dependencies.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { run as runScript, stub } from './stub.mjs';

const SCRIPTS = join(dirname(fileURLToPath(import.meta.url)), 'ignis', 'scripts');
const generate = (args, env) => runScript(join(SCRIPTS, 'generate-image.mjs'), args, env);
const edit = (args, env) => runScript(join(SCRIPTS, 'edit-image.mjs'), args, env);

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 1, 2, 3, 250, 251, 252, 253]);
const row = (id, kind, extra = {}) => ({ id, source: 'built-in', on_disk: 'complete', kind, loaded: false, available: true, ...extra });
const CATALOG = {
  object: 'list',
  data: [
    row('ignis-27b', 'text', { loaded: true }),
    row('ignis-image-absent', 'image', { on_disk: 'absent' }),
    row('ignis-image-big', 'image', { available: false }),
    row('ignis-image-2.1-turbo', 'image'),
  ],
};
const MADE = { created: 1, data: [{ b64_json: PNG.toString('base64') }], seed: 7, size: '1024x1024' };

/** A stub serving the catalog and both image routes. */
function ignis({ catalog = [200, CATALOG], image = [200, MADE] } = {}) {
  return stub((request) => (request.url === '/v1/models/catalog' ? catalog : image));
}

async function workdir() {
  return mkdtemp(join(tmpdir(), 'ignis-skill-'));
}

/** The text parts and file parts of a multipart body, in order. */
function parts(request) {
  const boundary = /boundary=(.+)$/.exec(request.headers['content-type'])[1];
  const body = request.body.toString('latin1');
  return body
    .split(`--${boundary}`)
    .slice(1, -1)
    .map((raw) => {
      const [head, ...rest] = raw.replace(/^\r\n/, '').split('\r\n\r\n');
      const content = rest.join('\r\n\r\n').replace(/\r\n$/, '');
      return {
        name: /name="([^"]*)"/.exec(head)[1],
        filename: /filename="([^"]*)"/.exec(head)?.[1],
        type: /content-type: (.+)/i.exec(head)?.[1],
        content,
      };
    });
}

test('generate-image writes the full-size PNG the server answered', async () => {
  const server = await ignis();
  const dir = await workdir();
  const out = join(dir, 'logo.png');
  const { code, stdout } = await generate(['a red fox', out], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.equal(code, 0);
  assert.deepEqual(await readFile(out), PNG);
  assert.match(stdout, /wrote .*logo\.png \(1024x1024, seed 7\)/);
  const post = server.requests.find((request) => request.method === 'POST');
  assert.equal(post.url, '/v1/images/generations');
  assert.match(post.headers['content-type'], /^application\/json/);
  assert.deepEqual(JSON.parse(post.body), {
    model: 'ignis-image-2.1-turbo',
    prompt: 'a red fox',
    response_format: 'b64_json',
  });
});

test('generate-image sends --size and --model, and asks no catalog with --model', async () => {
  const server = await ignis();
  const out = join(await workdir(), 'a.png');
  const { code } = await generate(['a fox', out, '--size', '2048x2048', '--model', 'my-image'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.equal(code, 0);
  assert.deepEqual(server.requests.map((request) => request.url), ['/v1/images/generations']);
  assert.deepEqual(JSON.parse(server.requests[0].body), {
    model: 'my-image',
    prompt: 'a fox',
    response_format: 'b64_json',
    size: '2048x2048',
  });
});

test('the catalog pick takes the loaded image model, else the first one on disk this card loads', async () => {
  const loaded = { object: 'list', data: [row('ignis-27b', 'text'), row('img-a', 'image'), row('img-b', 'image', { loaded: true })] };
  const server = await ignis({ catalog: [200, loaded] });
  const out = join(await workdir(), 'a.png');
  await generate(['a fox', out], { IGNIS_BASE_URL: server.url });
  await server.close();
  const post = server.requests.find((request) => request.method === 'POST');
  assert.equal(JSON.parse(post.body).model, 'img-b');
});

test('no image model on disk fails naming --model, sending nothing', async () => {
  const server = await ignis({ catalog: [200, { object: 'list', data: [row('ignis-27b', 'text')] }] });
  const out = join(await workdir(), 'a.png');
  const { code, stderr } = await generate(['a fox', out], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.match(stderr, /generate-image: ignis has no image model on disk/);
  assert.match(stderr, /--model/);
  assert.equal(server.requests.length, 1, 'only the catalog was asked');
});

test('a catalog error prints the server JSON', async () => {
  const refusal = { error: { message: 'warming up', type: 'server_error', code: 'server_not_ready' } };
  const server = await ignis({ catalog: [503, refusal] });
  const out = join(await workdir(), 'a.png');
  const { code, stderr } = await generate(['a fox', out], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.match(stderr, /503/);
  assert.ok(stderr.includes(JSON.stringify(refusal)), stderr);
});

test('generate-image: a server error prints the server JSON, writes no file', async () => {
  const refusal = { error: { message: 'size `9x9` is not served', type: 'invalid_request_error', code: 'invalid_value', param: 'size' } };
  const server = await ignis({ image: [400, refusal] });
  const out = join(await workdir(), 'a.png');
  const { code, stdout, stderr } = await generate(['a fox', out, '--size', '9x9'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.equal(stdout, '');
  assert.match(stderr, /400/);
  assert.ok(stderr.includes(JSON.stringify(refusal)), stderr);
  await assert.rejects(readFile(out), { code: 'ENOENT' });
});

test('generate-image: an answer with no image is an error', async () => {
  const server = await ignis({ image: [200, { created: 1, data: [] }] });
  const out = join(await workdir(), 'a.png');
  const { code, stderr } = await generate(['a fox', out, '--model', 'm'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.match(stderr, /holds no image/);
});

test('generate-image: usage errors and an unreachable server', async () => {
  const none = await generate([], {});
  assert.notEqual(none.code, 0);
  assert.match(none.stderr, /usage: generate-image\.mjs <prompt> <out\.png>/);
  const one = await generate(['only a prompt'], {});
  assert.notEqual(one.code, 0);
  assert.match(one.stderr, /usage: generate-image\.mjs/);
  const odd = await generate(['a', 'b.png', '--bogus', 'x'], {});
  assert.match(odd.stderr, /unknown option --bogus/);
  const badSize = await generate(['a', 'b.png', '--size', 'big'], {});
  assert.notEqual(badSize.code, 0);
  assert.match(badSize.stderr, /--size is WxH/);

  const closed = await ignis();
  const url = closed.url;
  await closed.close();
  const { code, stderr } = await generate(['a fox', join(await workdir(), 'a.png')], { IGNIS_BASE_URL: url });
  assert.notEqual(code, 0);
  assert.ok(stderr.includes(url), stderr);
  assert.match(stderr, /cannot reach ignis/);
});

test('the key is sent as a bearer only when set, on the catalog and the image call alike', async () => {
  const server = await ignis();
  const out = join(await workdir(), 'a.png');
  await generate(['a fox', out], { IGNIS_BASE_URL: server.url });
  await generate(['a fox', out], { IGNIS_BASE_URL: `${server.url}/`, IGNIS_API_KEY: 'sk-test-key' });
  await server.close();
  assert.deepEqual(
    server.requests.map((request) => [request.url, request.headers.authorization]),
    [
      ['/v1/models/catalog', undefined],
      ['/v1/images/generations', undefined],
      ['/v1/models/catalog', 'Bearer sk-test-key'],
      ['/v1/images/generations', 'Bearer sk-test-key'],
    ],
  );
});

test('edit-image sends the pictures as image[] parts with the prompt and model, and writes the PNG', async () => {
  const server = await ignis();
  const dir = await workdir();
  const first = join(dir, 'photo.jpg');
  const second = join(dir, 'logo.png');
  const jpg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3]);
  await writeFile(first, jpg);
  await writeFile(second, PNG);
  const out = join(dir, 'edited.png');
  const { code, stdout } = await edit(['make it night', out, first, second], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.equal(code, 0);
  assert.deepEqual(await readFile(out), PNG);
  assert.match(stdout, /wrote .*edited\.png/);
  const post = server.requests.find((request) => request.method === 'POST');
  assert.equal(post.url, '/v1/images/edits');
  assert.match(post.headers['content-type'], /^multipart\/form-data; boundary=/);
  const sent = parts(post);
  assert.deepEqual(
    sent.map((part) => [part.name, part.filename, part.type]),
    [
      ['image[]', 'photo.jpg', 'image/jpeg'],
      ['image[]', 'logo.png', 'image/png'],
      ['model', undefined, undefined],
      ['prompt', undefined, undefined],
      ['response_format', undefined, undefined],
    ],
  );
  assert.equal(sent[0].content, jpg.toString('latin1'));
  assert.equal(sent[1].content, PNG.toString('latin1'));
  assert.equal(sent[2].content, 'ignis-image-2.1-turbo');
  assert.equal(sent[3].content, 'make it night');
  assert.equal(sent[4].content, 'b64_json');
});

test('edit-image --model skips the catalog', async () => {
  const server = await ignis();
  const dir = await workdir();
  const input = join(dir, 'a.png');
  await writeFile(input, PNG);
  const { code } = await edit(['more blue', join(dir, 'o.png'), input, '--model', 'my-image'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.equal(code, 0);
  assert.deepEqual(server.requests.map((request) => request.url), ['/v1/images/edits']);
  assert.equal(parts(server.requests[0]).find((part) => part.name === 'model').content, 'my-image');
});

test('edit-image: a missing input is named and nothing is sent', async () => {
  const server = await ignis();
  const dir = await workdir();
  const { code, stderr } = await edit(['x', join(dir, 'o.png'), '/no/such/photo.png'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.match(stderr, /edit-image: cannot read \/no\/such\/photo\.png: no such file/);
  assert.equal(server.requests.length, 0);
});

test('edit-image: usage errors and a server error', async () => {
  const none = await edit(['only a prompt', 'o.png'], {});
  assert.notEqual(none.code, 0);
  assert.match(none.stderr, /usage: edit-image\.mjs <prompt> <out\.png> <input\.\.\.>/);

  const refusal = { error: { message: 'mask is not supported', type: 'invalid_request_error', code: 'invalid_value' } };
  const server = await ignis({ image: [400, refusal] });
  const dir = await workdir();
  const input = join(dir, 'a.webp');
  await writeFile(input, PNG);
  const out = join(dir, 'o.png');
  const { code, stdout, stderr } = await edit(['x', out, input, '--model', 'm'], { IGNIS_BASE_URL: server.url });
  await server.close();
  assert.notEqual(code, 0);
  assert.equal(stdout, '');
  assert.ok(stderr.includes(JSON.stringify(refusal)), stderr);
  await assert.rejects(readFile(out), { code: 'ENOENT' });
  assert.equal(parts(server.requests[0])[0].type, 'image/webp');
});

test('edit-image: an unreachable server is named', async () => {
  const closed = await ignis();
  const url = closed.url;
  await closed.close();
  const dir = await workdir();
  const input = join(dir, 'a.png');
  await writeFile(input, PNG);
  const { code, stderr } = await edit(['x', join(dir, 'o.png'), input, '--model', 'm'], { IGNIS_BASE_URL: url });
  assert.notEqual(code, 0);
  assert.ok(stderr.includes(url), stderr);
  assert.match(stderr, /cannot reach ignis/);
});
