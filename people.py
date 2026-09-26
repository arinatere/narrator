"""Named speaker identification: enroll a reference face once per person, then
match the current speaker's face against that roster using DeepFace embedding
distance -- so narration can say "Alex looks happy" instead of a fixed label.
"""

import json
import os

from deepface import DeepFace
from deepface.modules import verification

MODEL_NAME = "Facenet"
DISTANCE_METRIC = "cosine"
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "known_faces.json")


def _load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f)


def enroll(name, frame):
    """Compute and store a face embedding for `name` from `frame`, which must
    contain a clear view of their face. Overwrites any previous enrollment
    for that name; other enrolled people are left untouched."""
    reps = DeepFace.represent(
        frame, model_name=MODEL_NAME, enforce_detection=False, detector_backend="opencv"
    )
    largest = max(reps, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
    registry = _load_registry()
    registry[name] = largest["embedding"]
    _save_registry(registry)


def known_names():
    return list(_load_registry().keys())


class Identifier:
    """Loads the roster once at startup so identifying a live face crop only
    costs one embedding extraction, not a re-analysis of every reference photo."""

    def __init__(self):
        self.registry = _load_registry()
        self.threshold = verification.find_threshold(MODEL_NAME, DISTANCE_METRIC)

    def has_known_faces(self):
        return bool(self.registry)

    def identify(self, face_crop):
        """Returns the closest enrolled name for `face_crop`, or None if no
        enrolled person is a close enough match."""
        if not self.registry or face_crop.size == 0:
            return None
        try:
            reps = DeepFace.represent(
                face_crop, model_name=MODEL_NAME, enforce_detection=False, detector_backend="opencv"
            )
        except Exception:
            return None
        if not reps:
            return None
        embedding = reps[0]["embedding"]

        best_name, best_distance = None, None
        for name, known_embedding in self.registry.items():
            distance = verification.find_distance(embedding, known_embedding, DISTANCE_METRIC)
            if distance <= self.threshold and (best_distance is None or distance < best_distance):
                best_name, best_distance = name, distance
        return best_name
