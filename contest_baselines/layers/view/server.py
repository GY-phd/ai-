# -*- coding: utf-8 -*-
"""The View layer's HTTP transport: a read-only server.

The API is useful only if something can reach it.  This module puts the
``View``'s queries on a socket, using nothing but the standard library,
so the evaluation image does not gain a web framework.

The server is **read-only by construction**
-------------------------------------------
It serves ``GET`` only.  There is no route that writes, so a browser
that finds the port cannot change the model, the ledger or the state.
``POST`` and every other method is refused with ``405``.

Where it runs
-------------
The evaluation machine runs one process per round, offline, and the
platform owns the protocol's stdout.  So the server is **off by
default** and is started only when something asks for it:

* ``CONTEST_VIEW_PORT`` set  -> the server starts on that port;
* ``python -m contest_baselines.layers.view.server`` -> it starts and
  waits, which is what a local dashboard does.

The offline evaluation never sets the variable, so the submission's
runtime has no listening socket at all - one less way for the round to
go wrong.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from .api import View

#: The environment variable that turns the server on.
PORT_ENV = "CONTEST_VIEW_PORT"

#: The directory the server reads when the caller does not name one.
STATE_ENV = "CONTEST_VIEW_STATE"


class _Handler(BaseHTTPRequestHandler):
    """One request: route it, answer it, close it."""

    server_version = "zhixun-view/1.0"

    def __init__(self, *args, view=None, **kwargs):
        self._view = view
        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------- routing
    def _routes(self):
        """The endpoint table.

        Each route is ``(path, handler)`` where the handler takes the
        view and the query string and returns the answer.
        """
        view = self._view

        def _ledger(_view, _query):
            return view.ledger()

        def _rounds(_view, _query):
            return view.rounds()

        def _metrics(_view, _query):
            return view.metrics()

        def _overview(_view, _query):
            return view.overview()

        def _champion(_view, _query):
            return view.champion()

        def _hypothesis(_view, _query):
            return view.hypothesis()

        def _predict(_view, query):
            payload = query.get("inputs", [None])[0]
            if payload is None:
                return {"kind": "error", "version": 1,
                        "error": "the predict endpoint needs an 'inputs' parameter"}
            try:
                inputs = json.loads(payload)
            except ValueError:
                return {"kind": "error", "version": 1,
                        "error": "the 'inputs' parameter is not valid JSON"}
            return view.predict(inputs)

        return {
            "/api/ledger": _ledger,
            "/api/rounds": _rounds,
            "/api/metrics": _metrics,
            "/api/overview": _overview,
            "/api/champion": _champion,
            "/api/hypothesis": _hypothesis,
            "/api/predict": _predict,
        }

    # -------------------------------------------------------------- methods
    def do_GET(self):  # noqa: N802 - the name is the base class's
        """Answer a read."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = self._routes().get(parsed.path)
        if route is None:
            self._send(404, {"kind": "error", "version": 1,
                              "error": "no such endpoint: %s" % parsed.path})
            return
        try:
            answer = route(self._view, query)
        except Exception as error:  # noqa: BLE001 - a bad request is not a crash
            self._send(500, {"kind": "error", "version": 1,
                              "error": str(error)})
            return
        self._send(200, answer)

    def _refuse(self):
        """Every method that is not a read is refused."""
        self._send(405, {"kind": "error", "version": 1,
                          "error": "this API is read-only"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _refuse

    # --------------------------------------------------------------- output
    def _send(self, status, answer):
        body = json.dumps(answer, ensure_ascii=False, allow_nan=False,
                          indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - the base class's name
        """Keep the server quiet.

        The platform's stdout carries the protocol's JSONL frames, and a
        request log would corrupt it.
        """


def make_server(directory, port, host="127.0.0.1"):
    """Build the server without starting it."""
    view = View(directory)
    handler = type("_BoundHandler", (_Handler,), {"_view": view})
    return ThreadingHTTPServer((host, int(port)), handler)


def serve(directory, port, host="127.0.0.1"):
    """Serve until interrupted."""
    server = make_server(directory, port, host)
    server.serve_forever()


def serve_in_background(directory, port, host="127.0.0.1"):
    """Start the server on a thread and return it."""
    server = make_server(directory, port, host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def start_from_environment():
    """Start the server if the environment asks for it.

    Returns the server, or ``None`` when the environment does not ask
    for one - which is the normal case on the evaluation machine.
    """
    raw = os.environ.get(PORT_ENV, "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        return None
    directory = os.environ.get(STATE_ENV) or os.environ.get("CONTEST_WORK") or "."
    return serve_in_background(directory, port)


def main(argv=None):
    """Run the server in the foreground."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=STATE_ENV,
                        help="the directory to read (the round's state)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    directory = os.environ.get(args.state) or args.state
    print("view: serving %s on http://%s:%d/api/overview"
          % (directory, args.host, args.port))
    serve(directory, args.port, args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
