# -*- coding: utf-8 -*-
"""One canonical byte form for every text file, so digests never drift.

The manifest records a SHA-256 per shipped file.  A digest is only
meaningful if it is the same on every machine, and the line endings of a
checked-out file are *not* the same on every machine: Git converts LF to
CRLF in a Windows working tree when ``core.autocrlf`` is on, and a fresh
clone on Linux gets LF back.  Hashing the raw bytes would then make every
text file look tampered with on the machine that did not create it.

So the digest is always computed over the **canonical** form: LF line
endings, and binary payloads untouched.  That is exactly the form Git
stores in its object database, so the digest agrees with the blob the
repository actually carries.

    python -c "from text_canonical import canonical_bytes; ..."
"""
from __future__ import annotations

# A file is treated as binary when a NUL byte appears in its first block.
# That is the same heuristic Git uses to decide whether to apply the text
# conversion rules, so the two agree about what a "text file" is.
_SNIFF_BYTES = 8192


def is_binary(data: bytes) -> bool:
    """Return True when *data* looks like a binary payload."""
    return b"\x00" in data[:_SNIFF_BYTES]


def canonical_bytes(data: bytes) -> bytes:
    """Return *data* in the form the repository stores.

    Text files get their line endings normalised to LF; binary files are
    returned unchanged.
    """
    if is_binary(data):
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_digest(data: bytes) -> str:
    """The SHA-256 of the canonical form of *data*, hex-encoded."""
    import hashlib

    return hashlib.sha256(canonical_bytes(data)).hexdigest()
