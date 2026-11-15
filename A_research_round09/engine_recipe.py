"""Optional RecipeEngine extension of TaskAEngine_v1; default remains stable."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from types import SimpleNamespace
from uuid import uuid4

from recipe import ROOT, StructuredModel, anchor_update, np
sys.path.insert(0, str(ROOT / "TaskAEngine_v1"))
from task_a_engine import Engine
from task_a_engine.engine import encode, resources
from task_a_engine.vendor.predict import digest


class RecipePredictor:
    def __init__(self, model, mean, scale):
        self.model = model
        self.parent = SimpleNamespace(mean=mean, scale=scale)

    def predict(self, x):
        return self.model.predict(x)


class RecipeEngine(Engine):
    """Reuse Engine lifecycle with two explicitly requested, fixed training recipes."""
    def _load(self, db, model_id):
        spec = self._spec(db, model_id)
        if spec["kind"] != "replay_anchor":
            return super()._load(db, model_id)
        path = (self.store / spec["checkpoint"]).resolve()
        if not path.is_relative_to(self.store) or digest(path) != spec["sha256"]:
            raise ValueError("recipe checkpoint digest mismatch")
        model = StructuredModel.load(path)
        with np.load(path, allow_pickle=False) as state:
            mean, scale = state["input_mean"], state["input_scale"]
        if mean.shape != (128,) or scale.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("invalid recipe input reference")
        return RecipePredictor(model, mean, scale)

    def train(self, path=None, release=None, *, parent_id=None, method="mlp_continue", parameters=None,
              replay_path=None, replay_release=None):
        if method == "no_update":
            return {"status": "kept", "active_model_id": self.status()["active_model_id"], "reason": "no_update_requested"}
        if method not in ("structured_bootstrap", "C_replay_anchor"):
            return super().train(path, release, parent_id=parent_id, method=method, parameters=parameters)
        if parameters:
            raise ValueError("recipe uses the frozen feature mapping and lambda=1")
        data = self._release(path, release, "train")
        replay = self._release(replay_path, replay_release, "train") if method == "C_replay_anchor" else data
        if self._development(data) or self._development(replay):
            raise ValueError("development evaluation data cannot train this recipe")
        if method == "C_replay_anchor" and set(data["keys"]) & set(replay["keys"]):
            raise ValueError("calibration overlaps historical replay")
        with self._db(write=True) as db:
            current = self._get(db, "current")
            parent_id = parent_id or current["active_model_id"]
            parent_spec = self._spec(db, parent_id)
            if self._known(db, data["keys"], "evaluated") or self._known(db, replay["keys"], "evaluated"):
                raise ValueError("recipe training cannot reuse evaluation exposure")
            if method == "C_replay_anchor":
                if parent_spec["kind"] != "replay_anchor":
                    raise ValueError("update requires an explicitly selected structured parent")
                if replay["sha256"] != parent_spec["replay_sha256"]:
                    raise ValueError("replay differs from the parent's fixed source dataset")
                if set(data["keys"]) & set(parent_spec["calibration_keys"]):
                    raise ValueError("calibration was already used in this parent lineage")
                parent = self._load(db, parent_id)
            else:
                if set(data["y"].tolist()) != set(range(6)):
                    raise ValueError("initial structured head requires six source classes")
                self._load(db, parent_id)
            self._mark(db, data["keys"], "trained")
            self._mark(db, replay["keys"], "trained")
            cpu, wall = process_time(), perf_counter()
            name = "anchor_" + uuid4().hex[:16]
            destination = self.store / "models" / f"{name}.npz"
            try:
                if method == "structured_bootstrap":
                    model = StructuredModel.fit(data["x"], data["y"])
                    mean = data["x"].mean(axis=0)
                    scale = np.maximum(data["x"].std(axis=0), 1e-12)
                    calibration_keys = []
                else:
                    model = anchor_update(parent.model, data["x"], data["y"], replay["x"], replay["y"])
                    mean, scale = parent.parent.mean, parent.parent.scale
                    calibration_keys = sorted(set(parent_spec["calibration_keys"]) | set(data["keys"]))
                with TemporaryDirectory(prefix=".anchor-", dir=self.store) as temporary:
                    staged = Path(temporary) / "checkpoint.npz"
                    np.savez_compressed(staged, format="round08.structured.v1", kind=model.kind,
                        variant=model.variant, amplitude_scale=model.amplitude_scale, center=model.center,
                        scale=model.scale, coef=model.coef, input_mean=mean, input_scale=scale)
                    StructuredModel.load(staged)
                    sha = digest(staged)
                    os.replace(staged, destination)
                spec = {"kind": "replay_anchor", "sha256": sha, "checkpoint": destination.relative_to(self.store).as_posix(),
                        "parent_id": parent_id, "parent_sha256": parent_spec["sha256"], "method": method,
                        "replay_sha256": replay["sha256"], "calibration_keys": calibration_keys,
                        "training": {"sha256": data["sha256"], "release": data["provenance"], "rows": len(data["y"])},
                        "replay_release": replay["provenance"], "recipe_version": "fixed-round08-lambda1"}
                db.execute("INSERT INTO models VALUES (?,?)", (name, encode(spec)))
            except (ValueError, OSError, FloatingPointError, np.linalg.LinAlgError) as error:
                if destination.exists():
                    destination.unlink()
                result = {"status": "failed", "error": str(error), "active_model_id": current["active_model_id"],
                          "resources": resources(cpu, wall)}
                self._event(db, {"action": "recipe_train", **result})
                return result
            result = {"status": "trained", "model_id": name, "parent_id": parent_id, "sha256": sha,
                      "active_model_id": current["active_model_id"], "method": method,
                      "training_rows": len(data["y"]), "class_counts": np.bincount(data["y"], minlength=6).tolist(),
                      "missing_calibration_classes": np.flatnonzero(np.bincount(data["y"], minlength=6) == 0).tolist(),
                      "resources": resources(cpu, wall)}
            self._event(db, {"action": "recipe_train", **result})
            return result

    def evaluate(self, model_id, path, release):
        record = super().evaluate(model_id, path, release)
        with self._db(write=True) as db:
            if self._spec(db, model_id)["kind"] != "replay_anchor":
                return record
            counts = record["metrics"]["candidate"]["per_class"]
            checks = {"real_store": self.mode == "real", "independent_evidence": record["evidence"]["independent_usable"],
                      "six_classes_evaluated": all(row["support"] > 0 for row in counts.values())}
            record["recipe_enable_conditions"] = checks
            if not all(checks.values()):
                if record["eligible"]:
                    self._event(db, {"action": "recipe_evidence_restricted", "evaluation_id": record["evaluation_id"],
                                     "eligible": False, "conditions": checks})
                record.update(eligible=False, reason="recipe_independent_real_six_class_evaluation_required")
                record["agent_summary"].update(decision="keep", reason=record["reason"])
            stored = json.loads(db.execute("SELECT record FROM evaluations WHERE id=?", (record["evaluation_id"],)).fetchone()[0])
            stored.update(record)
            db.execute("UPDATE evaluations SET record=? WHERE id=?", (encode(stored), record["evaluation_id"]))
        return record

    def activate(self, model_id, evaluation_id, *, expected_revision=None):
        # Recheck extra conditions at the point of switching, including for an
        # evaluation recorded through an older Engine instance on the same DB.
        with self._db() as db:
            current = self._get(db, "current")
            if expected_revision is not None and current["revision"] != expected_revision:
                raise ValueError("active revision changed")
            spec = self._spec(db, model_id)
            row = db.execute("SELECT record FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
            if row is None:
                raise ValueError("unknown evaluation_id")
            record = json.loads(row[0])
            if record["model_id"] != model_id:
                raise ValueError("evaluation belongs to another model")
            if spec["kind"] == "replay_anchor":
                enough = all(m["support"] > 0 for m in record["metrics"]["candidate"]["per_class"].values())
                if self.mode != "real" or not record["evidence"]["independent_usable"] or not enough:
                    return {"status": "kept", "active_model_id": current["active_model_id"], "revision": current["revision"],
                            "reason": "recipe_independent_real_six_class_evaluation_required"}
        return super().activate(model_id, evaluation_id, expected_revision=expected_revision)

    def propose(self, diagnosis, *, training_release=None):
        result = super().propose(diagnosis, training_release=training_release)
        if len(result["candidates"]) > 1:
            parents = [m["model_id"] for m in self.status()["models"] if m["kind"] == "replay_anchor"]
            result["candidates"].append({"method": "C_replay_anchor" if parents else "structured_bootstrap",
                                         "available_parent_ids": parents,
                                         "requires": "released_training_labels_and_parent_bound_replay" if parents else "six_class_source_training"})
        return result
