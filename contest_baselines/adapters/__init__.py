# -*- coding: utf-8 -*-
"""The adapter plugin package.

One plugin per dataset profile.  ``base`` holds the contract,
``hypotheses`` holds the shared bank, ``profile_a``/``profile_b``/
``profile_c`` hold the profiles, and ``registry`` maps a dataset key to
its adapter.

The point of the split is that the loop asks *questions* and the adapter
answers them:

===========================================  ===========================
question                                      answered by
===========================================  ===========================
what task is this profile?                    ``Adapter.task``
what are its labels?                          ``Adapter.classes``
what is a unit in this profile?               ``Adapter.group_of``
which label does an id carry?                 ``Adapter.label_of``
which hypotheses can it test?                 ``Adapter.hypotheses``
how long may a round run?                     ``Adapter.epoch_scale``
how wide should a capacity migration be?     ``Adapter.resize_hidden``
===========================================  ===========================

Before this package existed, those answers were ``if profile == "A"``
branches scattered through ``strategies.py`` and ``train.py``.
"""
from __future__ import annotations

from .base import Adapter
from .hypotheses import BANK, Hypothesis
from .profile_a import CLASSES_A, GasAdapter
from .profile_b import BearingAdapter
from .profile_c import CmapssAdapter
from .registry import KNOWN_KEYS, get, register, unregister

__all__ = [
    "Adapter",
    "BANK",
    "BearingAdapter",
    "CLASSES_A",
    "CmapssAdapter",
    "GasAdapter",
    "Hypothesis",
    "KNOWN_KEYS",
    "get",
    "register",
    "unregister",
]
