"""Verify the local PyTRiO setup using the persisted CLI profile.

Run:  python tools/check_pytrio_profile.py
Requires: `trio login` to have been executed at least once.
Uses only read-only endpoints, so it consumes no model quota.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "contest_harness_sdk"))


def main():
    import json
    import urllib.request

    import pytrio
    from pytrio.config import config as trio_config

    print("pytrio module :", pytrio.__file__)

    # 1) credential resolution: saved profile first, then the env var
    trio_config.load_login_config()
    key = trio_config.auth_state.api_key or ""
    base = trio_config.configuration.base_url or ""
    print("profile base_url :", base)
    print("profile api_key  :", (key[:12] + "...len=" + str(len(key))) if key else "<none>")

    if not key:
        key = os.environ.get("PYTRIO_API_KEY", "")
        base = base or os.environ.get("PYTRIO_BASE_URL", "https://pytrio.com")
        print("env PYTRIO_API_KEY:", (key[:12] + "...len=" + str(len(key))) if key else "<none>")

    if not key:
        print("RESULT: FAIL - no API key available")
        return 1

    base = base.rstrip("/")

    # 2) read-only API call: list models (consumes no token quota)
    request = urllib.request.Request(
        base + "/api/models?limit=100&offset=0",
        headers={"Authorization": "Bearer " + key, "X-API-Key": key},
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.load(response)
    print("available models :", [item["model_id"] for item in payload.get("items", [])])

    # 3) quota status: the credit-account endpoint is session-scoped, so it is
    #    only readable from the signed-in browser session. It is checked there.
    print("credit account   : check in the signed-in browser session (/billing)")

    print("\nRESULT: OK - credentials resolve and the key authenticates")
    print("NOTE: live inference still requires a non-zero PyTRiO workspace balance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
