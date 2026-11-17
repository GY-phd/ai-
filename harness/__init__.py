# -*- coding: utf-8 -*-
"""Industrial model self-evolution harness (drift -> diagnose -> experiment -> gate -> rollback).

The package ships code and configuration only: no model weights, no serving
process.  The deployed industrial model is a checkpoint produced by
``rulkit.train``; the decision layer is the competition-specified language model
(Qwen3.8-27B) reached through an OpenAI-compatible endpoint, with a
deterministic policy as the offline fallback.
"""
import sys

__version__ = "1.1.0"


def _harden_stdout_encoding() -> None:
    """Make stdout/stderr survive a non-UTF-8 console (GBK on Windows).

    SwanLab prints emoji from an ``atexit`` hook.  On a Windows console whose
    code page is GBK, that write raises ``UnicodeEncodeError`` *after* the run
    has already produced its artefacts - so the process dies with a traceback
    and the exit code is lost, even though the report was written.

    Reconfiguring the streams with ``errors="replace"`` keeps the run's exit
    status meaningful.  Failing to reconfigure is not fatal: a stream that does
    not support ``reconfigure`` is simply left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            # Already detached, or the stream refuses reconfiguration.
            pass


_harden_stdout_encoding()
