"""Bodies for run-b2.sh: the three pointing scenes at six sizes, plus a
5120px oversize probe. point + box + noul per request.

    python .scratch/vision-fanout/244/make-size-bodies.py
"""
import base64, io, json, pathlib
from PIL import Image

fix = pathlib.Path('crates/server/tests/fixtures/pointing')
out = pathlib.Path('.scratch/vision-fanout/244/size')
out.mkdir(exist_ok=True)
man = json.loads((fix / 'manifest.json').read_text())
SIZES = [4096, 3072, 2048, 1536, 1024, 768, 5120]
for scene in man['scenes']:
    im = Image.open(fix / scene['image']).convert('RGB')
    for s in SIZES:
        r = im if s == 4096 else im.resize((s, s), Image.BICUBIC)
        buf = io.BytesIO()
        r.save(buf, format='PNG')
        b = base64.b64encode(buf.getvalue()).decode()
        body = {
            "state": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}],
            "questions": {
                "where": {"type": "point", "instructions": "click the blue button"},
                "bounds": {"type": "box", "instructions":
                           "Locate the blue Save button in this screenshot and output its bounding box."},
                "legible": {"type": "noul", "instructions": "the button text is legible"},
            },
        }
        (out / f"q_{scene['id']}_{s}.json").write_text(json.dumps(body))
