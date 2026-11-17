// Shared by the skill's tests: a node:http stub of ignis and a runner for a script.

import http from 'node:http';
import { spawn } from 'node:child_process';

/** A stub ignis: records each request, answers `reply(request)`. */
export async function stub(reply) {
  const requests = [];
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      const request = { method: req.method, url: req.url, headers: req.headers, body: Buffer.concat(chunks) };
      requests.push(request);
      const [status, body] = reply(request);
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(body));
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  return { url, requests, close: () => new Promise((resolve) => server.close(resolve)) };
}

/** Runs a script with `env` over a clean slate: exit code, stdout, stderr. */
export function run(script, args, env) {
  return new Promise((resolve) => {
    const clean = { ...process.env };
    delete clean.IGNIS_BASE_URL;
    delete clean.IGNIS_API_KEY;
    const child = spawn(process.execPath, [script, ...args], { env: { ...clean, ...env } });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));
    child.on('close', (code) => resolve({ code, stdout, stderr }));
  });
}
