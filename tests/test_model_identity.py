# -*- coding: utf-8 -*-
"""The model identity contract: one authority, one rule.

The competition allows exactly two base models and one route of
post-training derivatives.  Deciding whether an id names one of them
is a compliance question, and a compliance question with two answers
in the repository is a compliance question that will eventually give
the wrong one.

So these tests check three things:

* the **rule** behaves - short names, slashed names, separator
  variants, post-training derivatives, and refusals all land where the
  competition says they should;
* the **authority** is singular - no module outside ``harness/llm.py``
  may carry a second copy of the matching rule;
* the **early comparison implementation** reaches the same answer
  through the shared entry point rather than through its own copy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The early comparison implementation lives under ``evo/src``, so its
#: package root has to be on the path for it to be importable at all.
_EVO_SRC = ROOT / "evo" / "src"
if _EVO_SRC.is_dir() and str(_EVO_SRC) not in sys.path:
    sys.path.insert(0, str(_EVO_SRC))

from harness.llm import (  # noqa: E402
    MODEL_ALIASES,
    is_allowed_model,
    normalize_model_id,
    resolve_model_id,
)

#: The bases the competition permits.
ALLOWED = ["Qwen3.8-27B", "Qwen3.5-4B"]


# --------------------------------------------------------------- the rule
class TestShortNames:
    """The competition's short form maps onto the platform's spelling."""

    @pytest.mark.parametrize("short", sorted(MODEL_ALIASES))
    def test_short_form_maps_to_the_platform_spelling(self, short):
        assert resolve_model_id(short) == MODEL_ALIASES[short]

    def test_the_documented_example(self):
        assert resolve_model_id("Qwen3.8-27B") == "Qwen/Qwen3.8-27B"

    def test_the_mapping_is_case_insensitive(self):
        assert resolve_model_id("qwen3.8-27b") == "Qwen/Qwen3.8-27B"

    def test_an_already_slashed_id_is_left_alone(self):
        assert resolve_model_id("Qwen/Qwen3.8-27B") == "Qwen/Qwen3.8-27B"


class TestUnknownIds:
    """An unknown id is passed through, not guessed at."""

    def test_an_unknown_id_is_returned_unchanged(self):
        assert resolve_model_id("gpt-4o") == "gpt-4o"

    def test_an_unknown_id_does_not_raise(self):
        assert resolve_model_id("some-other-model") == "some-other-model"

    def test_an_empty_id_becomes_an_empty_string(self):
        assert resolve_model_id(None) == ""
        assert resolve_model_id("") == ""
        assert resolve_model_id("   ") == ""


class TestNormalisation:
    """The shared comparison form."""

    @pytest.mark.parametrize("spelling", [
        "Qwen3.8-27B", "Qwen3-8-27B", "qwen3_8_27b", "Qwen/Qwen3.8-27B",
        "qwen3.8-27b",
    ])
    def test_the_spellings_of_one_model_agree(self, spelling):
        assert normalize_model_id(spelling) == normalize_model_id("Qwen3.8-27B")

    def test_an_empty_id_normalises_to_an_empty_string(self):
        assert normalize_model_id(None) == ""


class TestAllowance:
    """The compliance judgement."""

    @pytest.mark.parametrize("ok", [
        "Qwen3.8-27B", "qwen3-8-27b", "Qwen3.5-4B", "qwen3-5-4b",
        "Qwen/Qwen3.8-27B",
    ])
    def test_a_permitted_base_is_allowed(self, ok):
        assert is_allowed_model(ok, ALLOWED)

    @pytest.mark.parametrize("bad", ["gpt-4o", "llama3-70b", "", None])
    def test_a_model_outside_the_allow_list_is_refused(self, bad):
        assert not is_allowed_model(bad, ALLOWED)

    def test_a_post_training_derivative_is_allowed_by_its_base(self):
        assert is_allowed_model("qwen3.8-27b-ft-20261115", ALLOWED)

    def test_a_post_training_derivative_in_short_form_is_allowed(self):
        assert is_allowed_model("qwen3.8-27b-ft-20261115", ALLOWED)

    def test_an_explicitly_declared_derivative_is_allowed(self):
        assert is_allowed_model("my-custom-finetuned-model", ALLOWED,
                                ["my-custom-finetuned-model"])

    def test_a_derivative_of_the_wrong_base_is_refused(self):
        assert not is_allowed_model("llama3-70b-finetuned", ALLOWED, [])

    def test_an_explicit_declaration_does_not_rewrite_the_id(self):
        """The judgement never changes the spelling."""
        assert is_allowed_model("my-custom-finetuned-model", ALLOWED,
                                ["my-custom-finetuned-model"])
        assert resolve_model_id("my-custom-finetuned-model") == "my-custom-finetuned-model"

    def test_an_empty_allow_list_refuses_everything(self):
        assert not is_allowed_model("Qwen3.8-27B", [])

    def test_an_empty_model_id_is_refused(self):
        assert not is_allowed_model("", ALLOWED)


# ----------------------------------------------------------- the authority
class TestSingleAuthority:
    """No second copy of the rule anywhere else in the repository."""

    #: The rule's fingerprint: a model id reduced to letters and digits,
    #: then compared for containment.  A file that matches this shape is
    #: carrying its own copy.
    #:
    #: The pattern is deliberately narrow.  A bare "strip the separators"
    #: idiom also appears in column-key normalisation and in text
    #: handling, and those are not compliance rules - flagging them
    #: would make this test cry wolf.  So the fingerprint asks for the
    #: idiom **and** a model identity nearby.
    _PATTERN = re.compile(
        r"""["']\.join\(\s*ch\s+for\s+ch\s+in\s+.*?isalnum\(\)""",
        re.DOTALL,
    )

    #: Words that put a normalisation idiom in a model-identity context.
    _CONTEXT = re.compile(
        r"""model[_\s-]?id|model[_\s-]?name|allow[_\s-]?list|whitelist""",
        re.IGNORECASE,
    )

    def _python_files(self):
        """The repository's **live** source files.

        The tree holds archival copies of earlier layouts (``AI2/``,
        ``_tmp_rm/``) and build outputs.  Those are snapshots, not code
        that runs, and a rule frozen inside a snapshot is not a second
        authority - it is a record of one.  So the scan covers the
        directories that are actually imported.

        The Python environments are skipped by **prefix**, not by exact
        name: the machine may carry several (``.venv``, ``.venv-py310``,
        ``.venv-py313``, ``.venv-py314``), and a third-party package
        that ships a deliberately non-UTF-8 source file - ``joblib``
        does, in its test suite - would otherwise be read as if it were
        this project's code.
        """
        skip = {"node_modules", ".git", "__pycache__", "dist",
                "build", ".pytest_tmp", "openspec", "AI2", "_tmp_rm",
                "artifacts", "runs"}
        for path in ROOT.rglob("*.py"):
            if any(part in skip for part in path.parts):
                continue
            if any(part.startswith(".venv") for part in path.parts):
                continue
            yield path

    def test_only_the_authority_carries_the_normalisation_rule(self):
        offenders = []
        authority = ROOT / "harness" / "llm.py"
        for path in self._python_files():
            if path == authority:
                continue
            source = path.read_text(encoding="utf-8")
            if self._PATTERN.search(source) and self._CONTEXT.search(source):
                offenders.append(path.relative_to(ROOT).as_posix())
        assert offenders == [], (
            "these files carry their own copy of the model-id normalisation "
            "rule: %s" % ", ".join(offenders))

    def test_the_early_implementation_does_not_fallback_to_a_local_copy(self):
        """A fallback copy is still a second copy."""
        source = (ROOT / "evo" / "src" / "evo" / "llm.py").read_text(encoding="utf-8")
        body = source.split("is_allowed_model = _load_model_identity()", 1)[-1]
        assert self._PATTERN.search(body) is None, (
            "evo/llm.py keeps a fallback copy of the rule instead of "
            "reaching the authority")

    def test_the_authority_is_the_module_the_spec_names(self):
        assert (ROOT / "harness" / "llm.py").is_file()


# ------------------------------------------------- the comparison implementation
class TestTheComparisonImplementation:
    """``evo`` reaches the same answer through the shared entry point."""

    def test_evo_uses_the_shared_rule(self):
        from evo.llm import is_allowed_model as evo_is_allowed

        assert evo_is_allowed is is_allowed_model

    def test_evo_agrees_on_the_allowed_models(self):
        from evo.llm import is_allowed_model as evo_is_allowed

        for model in ("Qwen3.8-27B", "qwen3-8-27b", "qwen3.8-27b-ft-20261115"):
            assert evo_is_allowed(model, ALLOWED) is is_allowed_model(model, ALLOWED)

    def test_evo_agrees_on_the_refused_models(self):
        from evo.llm import is_allowed_model as evo_is_allowed

        for model in ("gpt-4o", "llama3-70b-finetuned"):
            assert evo_is_allowed(model, ALLOWED, []) is is_allowed_model(
                model, ALLOWED, [])
