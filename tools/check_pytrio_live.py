"""Live smoke check for the local PyTRiO setup. Not part of the shipped runtime."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "contest_harness_sdk"))

os.environ.setdefault("PYTRIO_BASE_URL", "https://pytrio.com")

from contest_harness_sdk.contest_harness.transport import model_transport  # noqa: E402


class FakeRound:
    """Minimal stand-in that routes to the real PyTRiO OpenAI endpoint."""

    model_name = os.environ.get("PYTRIO_MODEL", "Qwen/Qwen3.8-27B")
    max_output_tokens = 64
    call_ids = []

    def model(self, messages, **options):
        import json
        import urllib.request

        base = os.environ.get("HARNESS_LLM_BASE_URL", "https://pytrio.com/api/openai/v1")
        key = os.environ.get("HARNESS_LLM_API_KEY") or os.environ.get("PYTRIO_API_KEY", "")
        body = json.dumps({"model": self.model_name, "messages": messages, **options}).encode()
        request = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
        self.call_ids.append(payload.get("id"))
        return type("Receipt", (), {"response": payload})()


def main():
    rnd = FakeRound()
    with model_transport(rnd) as transport:
        print("base_url:", transport.base_url)
        print("model   :", rnd.model_name)
        print("placeholder key:", transport.api_key)
        print("--- live call through SDK transport -> PyTRiO ---")
        # The model emits a reasoning trace first, so the visible content
        # needs enough of the token budget left over after reasoning.
        reply = rnd.model(
            [{"role": "user", "content": "Reply with exactly: PYTRIO_OK"}],
            max_tokens=64,
            temperature=0,
        )
        message = reply.response["choices"][0]["message"]
        text = message.get("content") or ""
        print("content  :", repr(text))
        print("reasoning:", (message.get("reasoning") or "")[:80].replace("\n", " "))
        print("usage   :", reply.response.get("usage"))
        print("receipt id:", rnd.call_ids[-1])
        assert "PYTRIO_OK" in text.upper(), "unexpected reply"
        print("\nLIVE CHECK PASSED")


if __name__ == "__main__":
    main()
