"""Body for run-c.sh: one chat completion over the 4096px screenshot,
`max_tokens: 1`, sent identically to ignis and ninfer-serve.

    python .scratch/vision-fanout/244/make-chat-body.py
"""
import base64, json, pathlib

b = base64.b64encode(pathlib.Path('.scratch/decide-live/s4096.png').read_bytes()).decode()
body = {"model": "qwen3.8-27b", "max_tokens": 1, "temperature": 0, "stream": False,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}},
            {"type": "text", "text": "What is in this screenshot?"}]}]}
pathlib.Path('.scratch/vision-fanout/244/chat_4096.json').write_text(json.dumps(body))
