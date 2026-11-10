"""The known-answer set's scorer, its comparison, its fetch and its manifest (CPU, no network, no GPU): what a
reply is scored as, before any model has answered."""
import base64
import hashlib
import http.server
import io
import json
import os
import threading

import pytest

import fetch
import score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def test_a_reply_is_right_when_every_needle_group_appears_as_a_whole_word():
    assert score.matches("The square in the image is **Red**.", [["red"]])
    assert score.matches("There are three circles.", [["3", "three"]])
    assert score.matches("3", [["3", "three"]])
    assert score.matches('The text says "error: config.yaml not found".', [["config.yaml"], ["not found"]])
    assert not score.matches("The text says config.yaml.", [["config.yaml"], ["not found"]])
    # A needle inside a longer word or number is not the answer.
    assert not score.matches("It is covered in snow.", [["red"]])
    assert not score.matches("The number is 147.", [["47"]])
    assert not score.matches("A category of animals.", [["cat"]])
    assert score.matches("A cat-like animal: a cat.", [["cat"]])


def entry(entry_id="object-cat-01", kind="object", needles=(("cat", "kitten"),)):
    return {"id": entry_id, "kind": kind, "question": "What animal is shown in the image? Answer with one word.",
            "needles": [list(n) for n in needles]}


def test_the_request_asks_greedily_with_thinking_off_the_question_then_the_image():
    body = score.request_body(entry(), b"\x89PNG\r\n\x1a\nxyz", "image/png", "qwen3.8-flash-next", 64)
    assert (body["model"], body["temperature"], body["enable_thinking"], body["max_tokens"]) == \
        ("qwen3.8-flash-next", 0, False, 64)
    (message,) = body["messages"]
    text, image = message["content"]
    assert text == {"type": "text", "text": entry()["question"]}
    assert image["type"] == "image_url"
    assert image["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nxyz").decode()


class FakeServer(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible stand-in that answers each question from `ANSWERS`."""
    ANSWERS = {}
    seen = []

    def log_message(self, *args):
        pass

    def _send(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeServer.seen.append(body)
        question = body["messages"][0]["content"][0]["text"]
        self._send({"choices": [{"message": {"content": FakeServer.ANSWERS[question]}}],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 3}})


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def small_set(tmp_path):
    """A manifest of a canary and two photos, with their images on disk as fetch leaves them."""
    images = tmp_path / "images"
    for name in ("object-cat-01", "colour-door-01"):
        os.makedirs(images / name)
        (images / name / "scored.png").write_bytes(b"png:" + name.encode())
    canary = {"id": "canary-colour", "kind": "canary",
              "question": "What colour is the square in the image? Answer in one short sentence.",
              "needles": [["red"]],
              "source": {"repo": "crates/server/tests/fixtures/vision_canary/colour.png"}}
    door = dict(entry("colour-door-01", "colour", (("blue",),)), question="What colour is the door? Answer with one word.")
    manifest = {"schema": score.SCHEMA, "images": [canary, entry(), door]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_a_run_writes_every_reply_its_verdict_and_the_accuracy_and_raises_the_alarm_on_a_wrong_canary(
        tmp_path, server, capsys):
    manifest = small_set(tmp_path)
    FakeServer.ANSWERS = {manifest["images"][0]["question"]: "The square is blue.",
                          manifest["images"][1]["question"]: "Cat.",
                          manifest["images"][2]["question"]: "Blue"}
    FakeServer.seen = []
    rc = score.main(["run", "--url", server, "--manifest", str(tmp_path / "manifest.json"),
                     "--images", str(tmp_path / "images"), "--label", "fn", "--out", str(tmp_path / "out")])
    assert rc == 0
    rows = [json.loads(line) for line in (tmp_path / "out" / "fn.jsonl").read_text().splitlines()]
    assert [(r["id"], r["right"]) for r in rows] == [("canary-colour", False), ("object-cat-01", True),
                                                     ("colour-door-01", True)]
    assert rows[1]["answer"] == "Cat." and rows[1]["completion_tokens"] == 3
    assert rows[0]["image_sha256"] == hashlib.sha256(
        open(os.path.join(REPO_ROOT, "crates/server/tests/fixtures/vision_canary/colour.png"), "rb").read()).hexdigest()
    summary = json.loads((tmp_path / "out" / "fn.summary.json").read_text())
    assert summary["model"] == "fake-model" and summary["images"] == 3 and summary["right"] == 2
    assert summary["by_kind"]["colour"] == {"images": 1, "right": 1}
    assert summary["alarms"] == ["canary canary-colour answered wrongly: 'The square is blue.'"]
    assert "ALARM" in capsys.readouterr().out
    assert all(b["model"] == "fake-model" and b["temperature"] == 0 for b in FakeServer.seen)


def write_run(path, verdicts):
    with open(path, "w") as f:
        for i, right in enumerate(verdicts):
            f.write(json.dumps({"id": f"i{i:03}", "kind": "canary" if i < 4 else "object", "right": right,
                                "answer": "x"}) + "\n")


def test_the_comparison_is_paired_and_names_a_clear_gap_to_the_27b(tmp_path):
    write_run(tmp_path / "even-fn.jsonl", [True] * 4 + [True, False] * 30)
    write_run(tmp_path / "even-27b.jsonl", [True] * 4 + [False, True] * 30)
    even = score.compare(str(tmp_path / "even-fn.jsonl"), str(tmp_path / "even-27b.jsonl"))
    assert (even["images"], even["flash_next"]["right"], even["qwen27b"]["right"]) == (64, 34, 34)
    assert (even["flash_next_only"], even["qwen27b_only"]) == (30, 30)
    assert even["bootstrap"]["low"] < 0 < even["bootstrap"]["high"]
    assert even["alarms"] == []
    write_run(tmp_path / "low-fn.jsonl", [True, True, False, True] + [False] * 40 + [True] * 20)
    write_run(tmp_path / "low-27b.jsonl", [True] * 4 + [True] * 50 + [False] * 10)
    low = score.compare(str(tmp_path / "low-fn.jsonl"), str(tmp_path / "low-27b.jsonl"))
    assert low["bootstrap"]["high"] < 0 and low["mcnemar_p"] < 0.001
    assert any("canary i002" in a for a in low["alarms"])
    assert any("clearly below" in a for a in low["alarms"])


def test_fetch_keeps_the_original_it_pins_and_scores_a_copy_no_longer_than_the_bound(tmp_path, monkeypatch):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (3000, 1000), (200, 30, 30)).save(buffer, format="PNG")
    original = buffer.getvalue()
    item = {"id": "colour-wall-01", "source": {"url": "https://upload.example/wall.png",
                                               "sha1": hashlib.sha1(original).hexdigest(), "bytes": len(original)}}
    monkeypatch.setattr(fetch, "download", lambda url: original)
    path = fetch.fetch_one(item, str(tmp_path))
    assert path == os.path.join(str(tmp_path), "colour-wall-01", "scored.png")
    assert Image.open(path).size == (fetch.MAX_SIDE, round(1000 * fetch.MAX_SIDE / 3000))
    assert open(os.path.join(tmp_path, "colour-wall-01", "original.png"), "rb").read() == original
    # A second fetch reads nothing; an original whose bytes are not the pinned ones is refused and removed.
    monkeypatch.setattr(fetch, "download", lambda url: pytest.fail("downloaded again"))
    assert fetch.fetch_one(item, str(tmp_path)) == path
    bad = dict(item, id="colour-wall-02")
    monkeypatch.setattr(fetch, "download", lambda url: original + b"x")
    with pytest.raises(fetch.Refused, match="sha1"):
        fetch.fetch_one(bad, str(tmp_path))
    assert not os.path.exists(os.path.join(tmp_path, "colour-wall-02", "original.png"))


def test_a_transparent_original_is_scored_on_white(tmp_path):
    # A transparent pixel's colour channels are whatever the file holds (here black): what is seen is white.
    from PIL import Image
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    image.paste((230, 120, 20, 255), (0, 0, 20, 20))
    image.save(tmp_path / "pumpkin.png")
    fetch.derive(str(tmp_path / "pumpkin.png"), str(tmp_path / "scored.png"))
    scored = Image.open(tmp_path / "scored.png")
    assert scored.mode == "RGB"
    assert scored.getpixel((5, 5)) == (230, 120, 20) and scored.getpixel((35, 5)) == (255, 255, 255)


def test_the_committed_manifest_is_a_set_the_scorer_can_use_freely():
    manifest = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))
    assert manifest["schema"] == score.SCHEMA
    images = manifest["images"]
    ids = [i["id"] for i in images]
    assert len(ids) == len(set(ids))
    canaries = [i for i in images if i["kind"] == "canary"]
    public = [i for i in images if i["kind"] != "canary"]
    assert len(canaries) == 4 and 50 <= len(public) <= 100
    for i in canaries:
        assert os.path.isfile(os.path.join(REPO_ROOT, i["source"]["repo"]))
    for i in public:
        s = i["source"]
        assert i["kind"] in score.KINDS and i["needles"] and all(i["needles"])
        assert s["page"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert s["url"].startswith("https://upload.wikimedia.org/") and "?" not in s["url"]
        assert len(s["sha1"]) == 40 and s["bytes"] > 0 and s["author"]
        assert fetch.license_allowed(s["license"]), (i["id"], s["license"])
