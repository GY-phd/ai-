"""Deterministic gas recognition tools and transactional local version management."""

from contextlib import contextmanager
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from uuid import uuid4

from .vendor.baselines.model import Model
from .vendor.predict import Predictor, digest, design, MODES
from .vendor import feature_ops
from .vendor.decision import assess, DEFAULT_THRESHOLDS
from .data import read_inputs, labeled
import numpy as np

ASSETS = Path(__file__).parent / "vendor"
CLASSES = feature_ops.CLASSES
PUBLIC_BATCH_2 = "sha256:6be1370388bf67d017f7426a51e149a65baf9bc1186f5f41f6731ce5006ffbee"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return "sha256:" + hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def resources(cpu, wall):
    return {"cpu_seconds": process_time() - cpu, "wall_seconds": perf_counter() - wall,
            "llm_calls": 0, "llm_tokens": 0}


def metrics(y, scores):
    result = feature_ops.score(y, scores)
    result["macro_f1"] = float(np.mean([v["f1"] for v in result["per_class"].values() if v["support"] or v["predicted"]]))
    return result


def checked_scores(model, x):
    scores = model.predict(x)
    if scores.shape != (len(x), len(CLASSES)) or not np.isfinite(scores).all():
        raise ValueError("model produced invalid or nonfinite scores")
    return scores


class Fusion:
    def __init__(self, parent, state):
        self.parent, self.meta, self.state = parent, parent.meta, state

    def predict(self, x):
        s = self.state
        raw = feature_ops.transform(x, "kinetics")
        auxiliary = feature_ops.softmax(6 * (design(raw, s["mean"], s["scale"]) @ s["coef"]))
        return (self.parent.predict(x) + auxiliary) / 2


class Engine:
    def __init__(self, store, *, mode="real", policy=None):
        if mode not in ("real", "mock"):
            raise ValueError("mode must be real or mock")
        self.store = Path(store).resolve()
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / "models").mkdir(exist_ok=True)
        self.mode = mode
        self.manifest = ASSETS / "manifest.json"
        self.seed_digest = digest(self.manifest)
        provenance_path = Path(__file__).parent / "seed_provenance.json"
        self.provenance_digest = digest(provenance_path)
        seed = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.development_keys = frozenset(seed["development_keys"])
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        for key in ("incumbent", "fusion", "routing_state"):
            relative = manifest[key]
            path = (ASSETS / relative).resolve()
            if relative not in manifest["files"] or not path.is_relative_to(ASSETS.resolve()):
                raise ValueError("seed model reference must be a checked local asset")
        Predictor(self.manifest, "stable")
        selected_policy = {**DEFAULT_THRESHOLDS, **(policy or {})}
        assess(None, {}, thresholds=selected_policy)
        with self._db(write=True) as db:
            db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, spec TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS usage (key TEXT PRIMARY KEY, trained INTEGER NOT NULL DEFAULT 0, evaluated INTEGER NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, created_at TEXT NOT NULL, event TEXT NOT NULL)")
            if db.execute("SELECT 1 FROM state WHERE key='current'").fetchone() is None:
                self._put(db, "current", {"active_model_id": "stable", "revision": 0, "rollback_stack": [], "evaluation_id": None})
                self._put(db, "settings", {"mode": mode, "policy": selected_policy, "seed_digest": self.seed_digest,
                                           "provenance_digest": self.provenance_digest})
                for name in MODES:
                    spec = {"kind": "seed", "variant": name, "sha256": identity({"manifest": self.seed_digest, "variant": name}),
                            "parent_id": None, "training": {"sha256": seed["train_sha256"], "release_id": "official-public-batch1"}}
                    db.execute("INSERT INTO models VALUES (?,?)", (name, encode(spec)))
                self._mark(db, seed["training_keys"], "trained")
                self._mark(db, seed["development_keys"], "evaluated")
                self._event(db, {"action": "initialize", "active_model_id": "stable", "mode": mode})
            settings = self._get(db, "settings")
            if (settings["mode"] != mode or settings["seed_digest"] != self.seed_digest
                    or settings.get("provenance_digest") != self.provenance_digest):
                raise ValueError("store mode, seed bundle or provenance differs from its initialization")
            if policy is not None and settings["policy"] != selected_policy:
                raise ValueError("policy is fixed per store; use the existing policy or a new store")
            self.policy = settings["policy"]

    @contextmanager
    def _db(self, write=False):
        db = sqlite3.connect(self.store / "engine.sqlite3", timeout=30, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db, key):
        return json.loads(db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()[0])

    @staticmethod
    def _put(db, key, value):
        db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, encode(value)))

    @staticmethod
    def _event(db, record):
        db.execute("INSERT INTO events(created_at,event) VALUES (?,?)", (datetime.now(timezone.utc).isoformat(), encode(record)))

    @staticmethod
    def _mark(db, keys, column):
        db.executemany(f"INSERT INTO usage(key,{column}) VALUES (?,1) ON CONFLICT(key) DO UPDATE SET {column}=1", ((key,) for key in keys))

    @staticmethod
    def _known(db, keys, column):
        return any(db.execute(f"SELECT 1 FROM usage WHERE key=? AND {column}=1", (key,)).fetchone() for key in keys)

    @staticmethod
    def _spec(db, model_id):
        row = db.execute("SELECT spec FROM models WHERE id=?", (model_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown model_id {model_id}")
        return json.loads(row[0])

    def _load(self, db, model_id):
        spec = self._spec(db, model_id)
        if spec["kind"] == "seed":
            if digest(self.manifest) != self.seed_digest:
                raise ValueError("seed manifest changed")
            return Predictor(self.manifest, spec["variant"])
        path = (self.store / spec["checkpoint"]).resolve()
        if not path.is_relative_to(self.store) or digest(path) != spec["sha256"]:
            raise ValueError("candidate checkpoint digest mismatch")
        if spec["kind"] == "mlp":
            return Model.load(path)
        if spec["kind"] != "fusion":
            raise ValueError("unknown candidate kind")
        with np.load(path, allow_pickle=False) as arrays:
            state = {key: arrays[key] for key in ("mean", "scale", "coef")}
        return Fusion(self._load(db, spec["parent_id"]), state)

    def status(self):
        with self._db() as db:
            current = self._get(db, "current")
            models = [{"model_id": name, "kind": json.loads(spec)["kind"], "sha256": json.loads(spec)["sha256"]}
                      for name, spec in db.execute("SELECT id,spec FROM models ORDER BY rowid")]
        return {"active_model_id": current["active_model_id"], "revision": current["revision"], "mode": self.mode, "models": models}

    def predict(self, inputs, model_id=None):
        data = read_inputs(inputs)
        with self._db() as db:
            current = self._get(db, "current")
            name = model_id or current["active_model_id"]
            model = self._load(db, name)
            wall, cpu = perf_counter(), process_time()
            scores = checked_scores(model, data["x"])
            cost = resources(cpu, wall)
            cost["prediction_calls"] = 1
        return {"model_id": name, "revision": current["revision"], "classes": CLASSES,
                "labels": [CLASSES[i] for i in scores.argmax(axis=1)], "scores": scores.tolist(), "resources": cost}

    def diagnose(self, inputs, model_id=None):
        data = read_inputs(inputs)
        with self._db() as db:
            current = self._get(db, "current")
            name = model_id or current["active_model_id"]
            model = self._load(db, name)
            wall, cpu = perf_counter(), process_time()
            reference = model
            while hasattr(reference, "parent"):
                reference = reference.parent
            with np.errstate(over="ignore", invalid="ignore"):
                z = (data["x"] - reference.mean) / reference.scale
                scores = checked_scores(model, data["x"])
                shift = np.linalg.norm(z.mean(axis=0).reshape(16, 8), axis=1)
                total_shift = float(np.linalg.norm(shift))
            if not np.isfinite(shift).all() or not np.isfinite(total_shift):
                raise ValueError("input magnitude exceeds finite diagnostic range")
            rank = np.argsort(shift)[::-1][:3]
            value = {"model_id": name, "rows": len(z), "revision": current["revision"], "reference": "parent_training_scaler",
                     "mean_shift_l2": total_shift, "outside_3sd_fraction": float(np.mean(np.abs(z) > 3)),
                     "top_sensors": [{"sensor": int(i + 1), "mean_shift_l2": float(shift[i])} for i in rank],
                     "predicted_counts": dict(zip(CLASSES, np.bincount(scores.argmax(axis=1), minlength=6).tolist())),
                     "mean_entropy": float(np.mean(-np.sum(scores * np.log(np.maximum(scores, 1e-300)), axis=1))),
                     "labels_used": False, "interpretation": "descriptive_shift_may_include_class_mix",
                     "resources": resources(cpu, wall)}
            value["resources"]["prediction_calls"] = 1
        return value

    def propose(self, diagnosis, *, training_release=None):
        if not isinstance(diagnosis, Mapping) or (training_release is not None and not isinstance(training_release, Mapping)):
            raise ValueError("diagnosis and training_release must be JSON objects")
        with self._db() as db:
            current = self._get(db, "current")
            if diagnosis.get("model_id") != current["active_model_id"] or diagnosis.get("revision") != current["revision"]:
                raise ValueError("diagnosis is stale for the active model")
            spec = self._spec(db, current["active_model_id"])
        allowed = bool(training_release and training_release.get("purpose") == "train" and training_release.get("labels") == "released"
                       and training_release.get("release_id") and training_release.get("kind") == self.mode)
        candidates = [{"method": "no_update", "reason": "retain_current_version"}]
        if allowed:
            base = current["active_model_id"] if spec["kind"] == "mlp" or current["active_model_id"] == "stable" else "stable"
            candidates.extend([
                {"method": "mlp_continue", "parent_id": base, "parameters": {"epochs": 5, "learning_rate": .001}, "requires": "released_training_labels"},
                {"method": "fusion_refit", "parent_id": base, "parameters": {}, "requires": "released_training_labels_for_all_six_classes"},
            ])
        return {"active_model_id": current["active_model_id"], "revision": current["revision"], "candidates": candidates,
                "reason": "small_fixed_recipe_set" if allowed else "training_labels_unavailable", "agent_decides": True,
                "llm_calls": 0, "llm_tokens": 0}

    def _release(self, path, release, purpose):
        data = labeled(path, release, purpose, CLASSES)
        if data["provenance"]["kind"] != self.mode:
            raise ValueError("release kind must match store mode")
        return data

    def _development(self, data):
        return (data["sha256"] == PUBLIC_BATCH_2 or not self.development_keys.isdisjoint(data["keys"])
                or any((name or "").startswith("gas_b02_") for name in data["ids"]))

    def train(self, path, release, *, parent_id=None, method="mlp_continue", parameters=None):
        if method not in ("mlp_continue", "fusion_refit"):
            raise ValueError("supported training methods are mlp_continue and fusion_refit")
        parameters = dict(parameters or {})
        if method == "mlp_continue":
            if set(parameters) - {"epochs", "learning_rate"}:
                raise ValueError("unknown training parameter")
            epochs, lr = parameters.get("epochs", 5), parameters.get("learning_rate", .001)
            if type(epochs) is not int or not 1 <= epochs <= 10000 or isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr <= 1:
                raise ValueError("invalid epochs or learning rate")
            parameters = {"epochs": epochs, "learning_rate": lr}
        elif parameters:
            raise ValueError("fusion_refit uses the fixed first-round linear recipe")
        data = self._release(path, release, "train")
        if self._development(data):
            raise ValueError("public Batch 2 evaluation labels cannot be used for training")
        with self._db(write=True) as db:
            current = self._get(db, "current")
            parent_id = parent_id or current["active_model_id"]
            parent_spec = self._spec(db, parent_id)
            parent = self._load(db, parent_id)
            if method == "mlp_continue" and not (parent_spec["kind"] == "mlp" or parent_id == "stable"):
                raise ValueError("MLP continuation requires a MLP parent; select stable explicitly for another architecture")
            if method == "fusion_refit" and set(data["y"].tolist()) != set(range(6)):
                raise ValueError("fusion_refit requires training samples for all six classes")
            self._mark(db, data["keys"], "trained")
            name = "a_" + uuid4().hex[:16]
            wall, cpu = perf_counter(), process_time()
            provenance = {"sha256": data["sha256"], "release": data["provenance"], "rows": len(data["y"])}
            try:
                with TemporaryDirectory(prefix=".training-", dir=self.store) as temporary:
                    staged = Path(temporary) / "checkpoint.npz"
                    if method == "mlp_continue":
                        model = parent.parent if parent_id == "stable" else parent
                        model.meta["learning_rate"] = lr
                        initial_steps = model.meta["steps"]
                        losses = model.fit(data["x"], data["y"], epochs)
                        model.save(staged)
                        detail = {"initial_steps": initial_steps, "final_steps": model.meta["steps"], "last_loss": losses[-1]}
                    else:
                        values = feature_ops.transform(data["x"], "kinetics")
                        mean = np.median(values, axis=0)
                        low, high = np.percentile(values, (25, 75), axis=0)
                        spread = high - low
                        scale = np.maximum(spread, max(1e-6, .1 * float(np.median(spread))))
                        matrix = design(values, mean, scale)
                        weight = np.sqrt(len(data["y"]) / (6 * np.bincount(data["y"])[data["y"]]))
                        weighted = matrix * weight[:, None]
                        coef = np.linalg.solve(weighted.T @ weighted + np.eye(matrix.shape[1]), weighted.T @ (np.eye(6)[data["y"]] * weight[:, None]))
                        np.savez_compressed(staged, mean=mean, scale=scale, coef=coef)
                        detail = {"feature": "kinetics", "auxiliary_weight": .5, "fit": "balanced_ridge"}
                        if not all(np.isfinite(value).all() for value in (mean, scale, coef)):
                            raise ValueError("nonfinite fusion checkpoint")
                    cost = resources(cpu, wall)
                    sha = digest(staged)
                    destination = self.store / "models" / (name + ".npz")
                    os.replace(staged, destination)
                    spec = {"kind": "mlp" if method == "mlp_continue" else "fusion", "sha256": sha,
                            "checkpoint": destination.relative_to(self.store).as_posix(), "parent_id": parent_id,
                            "parent_sha256": parent_spec["sha256"], "training": provenance, "method": method,
                            "parameters": parameters, "detail": detail}
                    db.execute("INSERT INTO models VALUES (?,?)", (name, encode(spec)))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, OSError) as error:
                result = {"status": "failed", "active_model_id": current["active_model_id"], "error": str(error), "resources": resources(cpu, wall)}
                self._event(db, {"action": "train", "training": provenance, **result})
                return result
            result = {"status": "trained", "model_id": name, "parent_id": parent_id, "sha256": sha,
                      "active_model_id": current["active_model_id"], "training_rows": len(data["y"]),
                      "method": method, "resources": cost, "training_summary": detail}
            self._event(db, {"action": "train", "training": provenance, **result})
            return result

    def evaluate(self, model_id, path, release):
        data = self._release(path, release, "evaluate")
        with self._db(write=True) as db:
            if self._known(db, data["keys"], "trained"):
                raise ValueError("evaluation overlaps training exposure by ID or features")
            reused = self._known(db, data["keys"], "evaluated")
            current = self._get(db, "current")
            candidate, incumbent = self._load(db, model_id), self._load(db, current["active_model_id"])
            declared_independent = data["provenance"]["independent"]
            development = self._development(data)
            independent = declared_independent and not reused and not development
            wall, cpu = perf_counter(), process_time()
            old = metrics(data["y"], checked_scores(incumbent, data["x"]))
            new = metrics(data["y"], checked_scores(candidate, data["x"]))
            cost = resources(cpu, wall)
            cost["prediction_calls"] = 2
            review = assess(old, {model_id: new}, evidence="independent_released" if independent else "public_development", thresholds=self.policy)
            evaluation_id = "eval_" + uuid4().hex[:16]
            record = {"evaluation_id": evaluation_id, "model_id": model_id, "model_sha256": self._spec(db, model_id)["sha256"],
                      "incumbent_id": current["active_model_id"], "incumbent_sha256": self._spec(db, current["active_model_id"])["sha256"],
                      "revision": current["revision"], "dataset": {"sha256": data["sha256"], "release": data["provenance"], "rows": len(data["y"])},
                      "evidence": {"scope": self.mode, "independent_attested": declared_independent, "previously_evaluated": reused,
                                   "known_development_set": development, "independent_usable": independent,
                                   "generalization_verified": False},
                      "metrics": {"incumbent": old, "candidate": new}, "policy": review["safety"],
                      "eligible": review["decision"] == "promote", "reason": review["reason"], "resources": cost,
                      "agent_summary": {**review["agent_summary"], "evidence_scope": self.mode},
                      "metric_protocol": {"main": "macro_f1", "secondary": "accuracy", "macro_classes": "observed_true_or_predicted", "policy_is_official_score": False}}
            self._mark(db, data["keys"], "evaluated")
            db.execute("INSERT INTO evaluations VALUES (?,?)", (evaluation_id, encode({**record, "data_keys": data["keys"]})))
            self._event(db, {"action": "evaluate", "evaluation_id": evaluation_id, "model_id": model_id, "eligible": record["eligible"]})
            return record

    def activate(self, model_id, evaluation_id, *, expected_revision=None):
        with self._db(write=True) as db:
            current = self._get(db, "current")
            if expected_revision is not None and current["revision"] != expected_revision:
                raise ValueError("active revision changed")
            row = db.execute("SELECT record FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
            if row is None:
                raise ValueError("unknown evaluation_id")
            evidence = json.loads(row[0])
            if current["active_model_id"] == model_id and current["evaluation_id"] == evaluation_id:
                self._load(db, model_id)
                return {"status": "kept", "reason": "already_active", "active_model_id": model_id, "revision": current["revision"]}
            if evidence["model_id"] != model_id:
                raise ValueError("evaluation belongs to a different candidate")
            if evidence["revision"] != current["revision"] or evidence["incumbent_id"] != current["active_model_id"]:
                raise ValueError("evaluation is stale for the active version")
            if not evidence["eligible"]:
                return {"status": "kept", "reason": evidence["reason"], "active_model_id": current["active_model_id"], "revision": current["revision"]}
            if self._known(db, evidence["data_keys"], "trained"):
                raise ValueError("evaluation rows were subsequently released for training; obtain fresh evidence")
            if evidence["model_sha256"] != self._spec(db, model_id)["sha256"] or evidence["incumbent_sha256"] != self._spec(db, current["active_model_id"])["sha256"]:
                raise ValueError("evaluation model identities differ")
            self._load(db, model_id)
            self._load(db, current["active_model_id"])
            current["rollback_stack"].append(current["active_model_id"])
            current.update(active_model_id=model_id, revision=current["revision"] + 1, evaluation_id=evaluation_id)
            self._put(db, "current", current)
            result = {"status": "activated", "active_model_id": model_id, "revision": current["revision"], "evidence_scope": self.mode}
            self._event(db, {"action": "activate", "evaluation_id": evaluation_id, **result})
            return result

    def rollback(self, *, expected_revision=None):
        with self._db(write=True) as db:
            current = self._get(db, "current")
            if expected_revision is not None and current["revision"] != expected_revision:
                raise ValueError("active revision changed")
            if not current["rollback_stack"]:
                return {"status": "kept", "reason": "no_previous_activation", "active_model_id": current["active_model_id"], "revision": current["revision"]}
            previous = current["rollback_stack"][-1]
            self._load(db, previous)
            current["rollback_stack"].pop()
            current.update(active_model_id=previous, revision=current["revision"] + 1, evaluation_id=None)
            self._put(db, "current", current)
            result = {"status": "rolled_back", "active_model_id": previous, "revision": current["revision"]}
            self._event(db, {"action": "rollback", **result})
            return result
