# -*- coding: utf-8 -*-
"""The adapter registry: the one file that changes when a profile is added.

Adding a fourth dataset profile is now a three-step job:

1. write ``contest_baselines/adapters/profile_d.py`` with an
   ``ADAPTER`` instance;
2. add one line to ``_ADAPTERS`` below;
3. add the profile's data package to ``contest_baselines/offline_host.py``.

Nothing in ``train.py``, ``strategies.py`` or ``predict.py`` has to know
that the profile exists.

The registry also owns the **fallback** for a dataset key that has no
adapter.  The platform only ever sends ``A``, ``B`` or ``C``, so an
unknown key is a contract violation rather than a new profile - and the
fallback refuses to guess.  It raises instead, because silently
treating an unknown profile as a plain regression would train on the
wrong grouping and produce a candidate that scores nothing without
saying why.
"""
from __future__ import annotations

from .base import Adapter
from .profile_a import GasAdapter
from .profile_b import BearingAdapter
from .profile_c import CmapssAdapter

#: The registered adapters, keyed by dataset key.
_ADAPTERS = {
    GasAdapter.key: GasAdapter(),
    BearingAdapter.key: BearingAdapter(),
    CmapssAdapter.key: CmapssAdapter(),
}

#: The dataset keys the platform is allowed to send.
KNOWN_KEYS = tuple(sorted(_ADAPTERS))


def get(dataset_key):
    """The adapter for a dataset key.

    Raises ``KeyError`` for an unknown key: the platform's contract
    says the key is one of ``A``, ``B`` or ``C``, so an unknown key means
    the contract was violated and guessing would only hide it.
    """
    try:
        return _ADAPTERS[dataset_key]
    except KeyError:
        raise KeyError("no adapter for dataset key %r (known: %s)"
                       % (dataset_key, ", ".join(KNOWN_KEYS))) from None


def register(adapter):
    """Register an adapter at runtime.

    This is the plugin seam: a caller that ships its own adapter can
    register it and the loop will use it, without the loop knowing
    anything about it.
    """
    if not isinstance(adapter, Adapter):
        raise TypeError("an adapter must be an Adapter instance")
    if not adapter.key:
        raise ValueError("an adapter needs a dataset key")
    _ADAPTERS[adapter.key] = adapter
    return adapter


def unregister(dataset_key):
    """Remove an adapter.  Mostly useful in tests."""
    return _ADAPTERS.pop(dataset_key, None)


__all__ = [
    "Adapter",
    "KNOWN_KEYS",
    "get",
    "register",
    "unregister",
]
