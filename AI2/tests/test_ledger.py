# -*- coding: utf-8 -*-
"""Ledger hash chain: append, verify, detect tampering and broken links."""
import json
import tempfile
import unittest
from pathlib import Path

from harness.ledger import Ledger
from tools.verify_ledger import digest_of, load_ledger, verify_records


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.ledger = Ledger(self.workspace)

    def tearDown(self):
        self.tmp.cleanup()

    def _append_three(self):
        self.ledger.append({"cycle": 1, "stage": "decide", "action": "hpo", "decision": "promote"})
        self.ledger.append({"cycle": 2, "stage": "decide", "action": "do_nothing", "decision": "skip"})
        self.ledger.append({"cycle": 3, "stage": "decide", "action": "retrain_head", "decision": "reject"})
        return load_ledger(self.workspace / "ledger.jsonl")

    def test_chain_is_valid_after_appends(self):
        records = self._append_three()
        ok, message = verify_records(records)
        self.assertTrue(ok, message)
        self.assertIn("3 records", message)
        self.assertIn("1 promotions", message)

    def test_first_record_links_to_genesis(self):
        records = self._append_three()
        self.assertEqual(records[0]["prev_sha256"], "genesis")
        self.assertEqual(records[1]["prev_sha256"], records[0]["record_sha256"])

    def test_edited_payload_is_detected(self):
        records = self._append_three()
        records[1]["decision"] = "promote"     # rewrite history
        ok, message = verify_records(records)
        self.assertFalse(ok)
        self.assertIn("TAMPERED", message)

    def test_broken_link_is_detected(self):
        records = self._append_three()
        records.pop(1)                          # drop a record
        ok, message = verify_records(records)
        self.assertFalse(ok)
        self.assertIn("BROKEN LINK", message)

    def test_empty_ledger_is_not_ok(self):
        ok, message = verify_records([])
        self.assertFalse(ok)
        self.assertIn("empty", message)

    def test_digest_ignores_its_own_field(self):
        record = {"seq": 1, "prev_sha256": "genesis", "payload": 2}
        with_digest = {**record, "record_sha256": digest_of(record)}
        self.assertEqual(digest_of(record), digest_of(with_digest))

    def test_champion_round_trip(self):
        checkpoint = self.workspace / "model.npz"
        checkpoint.write_bytes(b"not-a-real-checkpoint")
        entry = {"checkpoint": str(checkpoint), "metrics": {"rmse": 487.47}}
        self.ledger.set_champion(entry)
        loaded = self.ledger.champion()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["metrics"]["rmse"], 487.47)

    def test_missing_champion_file_returns_none(self):
        self.ledger.set_champion({"checkpoint": str(self.workspace / "gone.npz"), "metrics": {}})
        self.assertIsNone(self.ledger.champion())


if __name__ == "__main__":
    unittest.main()
