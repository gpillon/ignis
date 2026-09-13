//! `ignis serve`'s stdout is one log stream, rendered by the one logging
//! system (ADR 0025): under `IGNIS_LOG_FORMAT=pretty` — what an interactive
//! terminal gets by default — a request's scheduler activity must not leave
//! raw JSONL lines behind. Before ADR 0025 the scheduler interval counters
//! bypassed `ignis-logging` and wrote `{"kind":"interval",...}` straight to
//! stdout on every step, whatever format the operator had chosen.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Kills the child (and reaps it) when dropped: the server never exits on
/// its own.
struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// An ephemeral localhost port the OS just handed out. Released before the
/// server binds it — a narrow race, but `--bind 127.0.0.1:0` would leave
/// the test no way to learn the port the server actually got.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("reserve a port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// Wait for a stdout line containing `needle`, returning every line seen up
/// to and including it.
fn read_until(rx: &mpsc::Receiver<String>, needle: &str, seen: &mut Vec<String>) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let line = rx
            .recv_timeout(remaining)
            .unwrap_or_else(|_| panic!("timed out waiting for `{needle}`; saw: {seen:#?}"));
        let matched = line.contains(needle);
        seen.push(line);
        if matched {
            return;
        }
    }
}

/// One non-streaming chat completion over a raw HTTP/1.1 connection,
/// returning the status line.
fn chat_completion(port: u16) -> String {
    let body = r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hello"}],"max_tokens":4}"#;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    write!(
        stream,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("send request");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read response");
    response.lines().next().unwrap_or_default().to_owned()
}

#[test]
fn a_served_request_leaves_no_jsonl_line_on_a_pretty_stdout() {
    let port = free_port();
    let mut command = Command::new(env!("CARGO_BIN_EXE_ignis-server"));
    command
        .args(["--bind", &format!("127.0.0.1:{port}")])
        .env("IGNIS_LOG_FORMAT", "pretty")
        .env("IGNIS_LOG_LEVEL", "info")
        .env_remove("IGNIS_ARTIFACT")
        .stdout(Stdio::piped());
    let mut child = command.spawn().expect("spawn ignis-server");
    let stdout = child.stdout.take().expect("piped stdout");
    let _guard = KillOnDrop(child);

    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut seen = Vec::new();
    read_until(&rx, "ignis.process.started", &mut seen);
    let status = chat_completion(port);
    assert!(status.contains("200"), "the completion must succeed: {status}");
    // `ignis.request.done` is emitted by the same telemetry consumer that
    // handles every scheduler tick of the request, after all of them — so by
    // the time it is on stdout, every line those ticks produced is too.
    read_until(&rx, "ignis.request.done", &mut seen);

    let json_lines: Vec<&String> =
        seen.iter().filter(|line| line.trim_start().starts_with('{')).collect();
    assert!(
        json_lines.is_empty(),
        "a pretty stdout must carry only pretty log lines, found JSONL: {json_lines:#?}"
    );
}
