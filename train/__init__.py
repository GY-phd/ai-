"""Offline SFT tooling for the harness decision layer.

The online loop never imports this package.  Training data is built from
artefacts already in the repository (``evidence/``) plus counterfactual cards
labelled by the deterministic policy in ``harness/policy.py``; the runtime
prompt is rendered by the same ``harness.llm`` helpers, so training and
inference cannot drift apart.
"""
