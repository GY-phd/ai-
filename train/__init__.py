"""Offline tooling for the PyTRIO experiment-planner SFT.

Nothing in this package is imported by the online harness. The harness only
needs the trained model path and the usual OpenAI-compatible environment
variables; these scripts build data, train, and evaluate offline.
"""
