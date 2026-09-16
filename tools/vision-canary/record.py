"""Record the vision canary (GitHub #178) from the reference engine.

Generates the fixed canary images (deterministic: PIL drawing + Arial, no
randomness), asks the reference one short, unambiguous question about each
with the image as a base64 `data:` URI, and writes the fixture the GPU test
(`crates/server/tests/vision_canary_gpu.rs`) scores ignis against.

The reference must be serving the same artifact, with vision, greedy and
thinking off:

    ninfer-serve.exe <artifact> --host 127.0.0.1 --port 8080 --vision --greedy \
        --no-thinking --max-context 8192 --max-concurrency 1

    python tools/vision-canary/record.py crates/server/tests/fixtures/vision_canary

The answers are stored as text; the test tokenizes them with the artifact's
tokenizer, the same way `ignis-bench oracle record` builds the text canary.
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-27b"
MAX_TOKENS = 32


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("arial.ttf", size)


def number() -> Image.Image:
    # 448x448: 196 merged tokens, wider than a 128-token prefill chunk.
    image = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(image).text((224, 224), "47", fill="black", font=font(220), anchor="mm")
    return image


def colour() -> Image.Image:
    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).rectangle((64, 64, 191, 191), fill=(220, 20, 20))
    return image


def circles() -> Image.Image:
    image = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(image)
    for cx in (72, 192, 312):
        draw.ellipse((cx - 44, 84, cx + 44, 172), fill=(30, 60, 220))
    return image


def screenshot_text() -> Image.Image:
    image = Image.new("RGB", (640, 160), (245, 245, 245))
    ImageDraw.Draw(image).text((24, 56), "error: config.yaml not found", fill=(20, 20, 20), font=font(40))
    return image


# Unambiguous questions, answered in a sentence: the fact is still checkable
# by eye, and the answer is long enough for the teacher-forced score to have
# resolution -- at one or two tokens per answer a single near-tie flip moves
# the suite by 7%, which is noise, not a floor.
CANARIES = [
    ("number", number, "What number is shown in the image? Answer in one short sentence."),
    ("colour", colour, "What colour is the square in the image? Answer in one short sentence."),
    ("circles", circles, "How many circles are in the image? Answer in one short sentence."),
    ("text", screenshot_text, "What does the text in the image say? Answer in one short sentence."),
]


def ask(png: bytes, question: str) -> str:
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    return reply["choices"][0]["message"]["content"]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fixture = {
        "engine": "ninfer-serve --vision --greedy --no-thinking",
        "max_tokens": MAX_TOKENS,
        "canaries": [],
    }
    for name, draw, question in CANARIES:
        path = out / f"{name}.png"
        draw().save(path, optimize=False)
        text = ask(path.read_bytes(), question)
        print(f"{name}: {text!r}")
        fixture["canaries"].append({"id": name, "image": path.name, "question": question, "text": text})
    (out / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
