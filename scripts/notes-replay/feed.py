"""Feeds a public recording to the Playground's File transcription and takes back its `.md` export
(GitHub #429), with no download: GET /<name> serves a file of the audio folder, POST /save/<name> writes
the body into the output folder. Bound to 127.0.0.1, with CORS for the page.

    python scripts/notes-replay/feed.py <audio folder> <output folder> [port]

In the page (the Transcribe view, source File, Speakers on, Suggest while listening off), a console snippet
fetches http://127.0.0.1:<port>/<name> into the file input (a DataTransfer) and dispatches `change`; once the
file is transcribed, a hook on HTMLAnchorElement.click POSTs the Download menu's `.md` to /save/<name>.
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

AUDIO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8765


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(AUDIO), **k)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        name = Path(self.path.removeprefix("/save/")).name
        body = self.rfile.read(int(self.headers["Content-Length"]))
        (OUT / name).write_bytes(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"saved")


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
