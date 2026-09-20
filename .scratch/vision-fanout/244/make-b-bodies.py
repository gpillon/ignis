import base64, json, pathlib
fix = pathlib.Path('crates/server/tests/fixtures/pointing')
out = pathlib.Path('.scratch/vision-fanout/244')
man = json.loads((fix/'manifest.json').read_text())
for scene in man['scenes']:
    b = base64.b64encode((fix/scene['image']).read_bytes()).decode()
    body = {
        "state": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}],
        "questions": {
            "where": {"type": "point", "instructions": "click the blue button"},
            "bounds": {"type": "box", "instructions":
                       "Locate the blue Save button in this screenshot and output its bounding box."},
            "legible": {"type": "noul", "instructions": "the button text is legible"},
        },
    }
    (out/f"b_{scene['id']}.json").write_text(json.dumps(body))
    print(scene['id'], scene['blue_box'], scene['blue_centre'])
