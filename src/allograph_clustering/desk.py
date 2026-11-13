"""The desk: one run's report served on this machine, classified again whenever its categories file changes.

Edit `categories/<code>.yaml`, reload the page, and the categories, the comparison with the classifier
clusters and the counts follow. The part of the page that depends only on the run (thumbnails, the tree,
the 3-D map) is built once.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import yaml

from . import pipeline, report


class Desk:
    def __init__(self, code: str, run: Path, data: dict, categories: Path):
        self.code, self.run, self.data, self.categories = code, run, data, categories
        self.char = chr(int(code.removeprefix("U+"), 16))
        self.of_run: dict | None = None
        self.stamp: int | None = None
        self.html = ""
        self.lock = Lock()

    def page(self) -> str:
        with self.lock:
            stamp = self.categories.stat().st_mtime_ns
            if stamp != self.stamp:
                if self.of_run is None:
                    self.of_run = report.run_data(self.run, self.data)
                chosen, _, category, stats = pipeline.label(self.run, self.data, self.categories)
                page = {**self.of_run, **report.category_data(self.of_run, chosen, category, stats)}
                self.html = report.render(self.run, chosen, self.categories.parent / "drawings", page, self.char)
                self.stamp = stamp
            return self.html


def serve(desk: Desk, port: int, log=print) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/":
                self.send_error(404)
                return
            try:
                status, kind, body = 200, "text/html", desk.page()
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                # A categories file caught half-edited: say what is wrong with it and keep serving.
                status, kind, body = 500, "text/plain", f"{desk.categories}: {error}"
            payload = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    log(f"desk for {desk.code} at http://127.0.0.1:{port}/ (categories: {desk.categories})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
