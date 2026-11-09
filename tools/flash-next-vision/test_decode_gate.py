"""The phase gate's driver and verdict (CPU, no GPU): what a leg measures from a stream, and what four legs
are judged by, before any leg has run."""
import http.server
import json
import threading
import time

import pytest

import decode_gate


class StreamingServer(http.server.BaseHTTPRequestHandler):
    """A stand-in that streams `TOKENS` one-token chunks `GAP` seconds apart, then the usage chunk."""
    TOKENS, GAP = 20, 0.005
    bodies = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = json.dumps({"data": [{"id": "qwen3.8-flash-next", "decode_lanes": 3}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        StreamingServer.bodies.append(body)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        tokens = min(self.TOKENS, body["max_tokens"])
        for i in range(tokens):
            time.sleep(self.GAP)
            chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        usage = {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": tokens}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_a_leg_measures_one_lane_then_three_at_once_greedy_and_to_the_cap(tmp_path, server):
    StreamingServer.bodies = []
    log = tmp_path / "server.log"
    log.write_text("\n".join(json.dumps(line) for line in [
        {"event_name": "ignis.runtime.flash_next_plan", "attributes": {"expert_cache_bytes": 1, "lanes": 3}},
        {"event_name": "ignis.request.admitted", "attributes": {}},
        {"event_name": "ignis.runtime.flash_next_plan", "attributes": {"expert_cache_bytes": 2, "lanes": 3}},
    ]) + "\n")
    rc = decode_gate.main(["run", "--url", server, "--label", "off-1", "--vision", "off", "--out", str(tmp_path),
                           "--max-tokens", "20", "--rounds", "1", "--server-log", str(log)])
    assert rc == 0
    leg = json.loads((tmp_path / "off-1.json").read_text())
    assert (leg["label"], leg["vision"], leg["model"]) == ("off-1", False, "qwen3.8-flash-next")
    prompts = len(decode_gate.PROMPTS)
    assert len(leg["c1"]["requests"]) == prompts and len(leg["c3"]["rounds"]) == 1
    # One-lane decode excludes the time to the first token: 19 gaps of ~5 ms after it.
    assert all(r["tokens"] == 20 for r in leg["c1"]["requests"])
    assert 20 < leg["c1"]["tok_s"] < 19 / 0.005
    assert leg["c3"]["tok_s"] > 0 and leg["c3"]["rounds"][0]["tokens"] == 20 * prompts
    # The last plan the log holds is the leg's.
    assert leg["plan"]["expert_cache_bytes"] == 2
    measured = [b for b in StreamingServer.bodies if b["max_tokens"] == 20]
    assert len(measured) == 2 * prompts
    assert all(b["temperature"] == 0 and b["ignore_eos"] is True and b["enable_thinking"] is False and b["stream"]
               for b in measured)


def leg(label, vision, c1, c3, cache=18_074_667_600, text="same"):
    return {"label": label, "vision": vision, "c1": {"tok_s": c1}, "c3": {"tok_s": c3},
            "texts": {str(i): text for i in range(3)}, "plan": {"expert_cache_bytes": cache, "lanes": 3}}


def write(tmp_path, legs):
    paths = []
    for value in legs:
        path = tmp_path / f"{value['label']}.json"
        path.write_text(json.dumps(value))
        paths.append(str(path))
    return paths


def test_the_gate_passes_when_each_adjacent_pair_holds_99_percent_at_both_widths(tmp_path):
    legs = [leg("off-1", False, 100.0, 250.0), leg("on-1", True, 99.5, 249.0),
            leg("on-2", True, 101.0, 252.0), leg("off-2", False, 100.5, 251.0)]
    verdict = decode_gate.verdict(write(tmp_path, legs))
    assert verdict["pass"] is True
    assert verdict["c1"]["pairs"] == pytest.approx([0.995, 101.0 / 100.5])
    assert verdict["c1"]["pooled"] == pytest.approx((99.5 + 101.0) / (100.0 + 100.5))
    assert verdict["expert_cache_equal"] is True and verdict["texts_identical"] is True


def test_one_pair_below_99_percent_fails_the_gate_and_says_where(tmp_path):
    legs = [leg("off-1", False, 100.0, 250.0), leg("on-1", True, 100.0, 245.0),
            leg("on-2", True, 100.0, 251.0), leg("off-2", False, 100.0, 250.0)]
    verdict = decode_gate.verdict(write(tmp_path, legs))
    assert verdict["pass"] is False
    assert verdict["c3"]["pairs"][0] == pytest.approx(0.98)
    assert any("c3" in reason and "off-1" in reason for reason in verdict["reasons"])


def test_an_expert_cache_or_a_text_that_moves_with_vision_is_reported(tmp_path):
    legs = [leg("off-1", False, 100.0, 250.0), leg("on-1", True, 100.0, 250.0, cache=17_000_000_000),
            leg("on-2", True, 100.0, 250.0, text="other"), leg("off-2", False, 100.0, 250.0)]
    verdict = decode_gate.verdict(write(tmp_path, legs))
    assert verdict["expert_cache_equal"] is False and verdict["texts_identical"] is False
    assert verdict["pass"] is False


def test_legs_not_in_abba_order_are_refused(tmp_path):
    legs = [leg("off-1", False, 100.0, 250.0), leg("off-2", False, 100.0, 250.0),
            leg("on-1", True, 100.0, 250.0), leg("on-2", True, 100.0, 250.0)]
    with pytest.raises(decode_gate.Refused, match="off, on, on, off"):
        decode_gate.verdict(write(tmp_path, legs))
