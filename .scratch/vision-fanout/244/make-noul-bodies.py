"""Bodies for run-a.sh: one `noul` decision over the rescaled screenshot.

    python .scratch/vision-fanout/244/make-noul-bodies.py
"""
import base64, json, pathlib

src = pathlib.Path('.scratch/decide-live')
out = pathlib.Path('.scratch/vision-fanout/244')
for name in ['s768', 's1536', 's4096']:
    b = base64.b64encode((src / f'{name}.png').read_bytes()).decode()
    body = {
        "state": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}],
        "questions": {"legible": {"type": "noul",
                                  "instructions": "the screenshot contains a blue button"}},
    }
    (out / f'noul_{name}.json').write_text(json.dumps(body))
