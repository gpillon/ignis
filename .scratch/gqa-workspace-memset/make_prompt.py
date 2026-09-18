"""Build one long, non-repeating prompt file for the prefill A/B.

Repetitive text would let the tokenizer collapse it; the point is a real
~64K-token span, so the body is drawn from the repo's own prose and source.
"""
import json
import os
import sys

ROOT = sys.argv[1]
TARGET_CHARS = int(sys.argv[2])
OUT = sys.argv[3]

pieces = []
total = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in
                   (".git", "target", "build", "node_modules", ".scratch", "models", "vendor")]
    for name in sorted(filenames):
        if not name.endswith((".rs", ".md", ".cu", ".cuh", ".h", ".cpp", ".ts")):
            continue
        path = os.path.join(dirpath, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        if not text.strip():
            continue
        pieces.append("\n\n===== %s =====\n%s" % (os.path.relpath(path, ROOT), text))
        total += len(pieces[-1])
        if total >= TARGET_CHARS:
            break
    if total >= TARGET_CHARS:
        break

body = "".join(pieces)[:TARGET_CHARS]
request = {
    "model": "ignis",
    "messages": [{"role": "user", "content": body + "\n\nReply with the single word: ok"}],
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(request, handle)
print("wrote %s: %d chars of prompt" % (OUT, len(body)))
