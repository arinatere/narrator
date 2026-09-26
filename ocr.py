"""Reads on-screen text via macOS's Vision framework (on-device OCR, no
external binary or model download) so narration can use whatever name a call
app already displays under a participant's video tile -- instead of requiring
manual enrollment."""

import re

import cv2
import Quartz
import Vision
from Foundation import NSData

# Call-app chrome that sometimes lands near a tile and would otherwise be
# mistaken for a participant's name.
_UI_NOISE = {
    "mute", "unmute", "stop video", "start video", "chat", "participants",
    "leave", "share screen", "record", "reactions", "more",
}

_YOU_SUFFIX = re.compile(r"\(\s*you\s*\)", re.IGNORECASE)


def _cgimage_from_bgr(frame):
    ok, buf = cv2.imencode(".png", frame)
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    return Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)


def recognize_text_blocks(frame, min_confidence=0.3):
    """Returns (text, confidence, (x, y, w, h)) for each text block Vision
    finds, in frame pixel coordinates with a top-left origin."""
    height, width = frame.shape[:2]
    cgimage = _cgimage_from_bgr(frame)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    success, _error = handler.performRequests_error_([request], None)
    if not success:
        return []

    blocks = []
    for obs in request.results():
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        text = candidate.string().strip()
        confidence = candidate.confidence()
        if not text or confidence < min_confidence:
            continue
        if text.lower() in _UI_NOISE:
            continue
        box = obs.boundingBox()  # normalized, Vision's origin is bottom-left
        x = box.origin.x * width
        w = box.size.width * width
        y = (1.0 - box.origin.y - box.size.height) * height  # flip to top-left origin
        h = box.size.height * height
        blocks.append((text, confidence, (x, y, w, h)))
    return blocks


def find_name_near_face(frame, face_region, max_vertical_gap=None, horizontal_margin=40):
    """Finds the text block that most plausibly labels `face_region`. Call apps
    overlay a participant's display name at the bottom-LEFT of their tile --
    not centered under wherever their face happens to be within it -- so this
    looks below the face and anywhere from the frame's left edge out to just
    past the face's right edge, picking the closest such match vertically.
    Returns (name, is_self) where `is_self` is True if the label contains a
    "(you)" marker, or (None, False) if nothing nearby looks like a name.

    `max_vertical_gap` defaults to 40% of the frame height: the face-detector's
    box size (and so its distance to the tile's bottom-anchored label) varies a
    lot frame to frame -- a fixed pixel cap missed real labels in testing -- so
    this scales with the capture instead of being a fixed pixel count."""
    fx, fy, fw, fh = face_region["x"], face_region["y"], face_region["w"], face_region["h"]
    if max_vertical_gap is None:
        max_vertical_gap = frame.shape[0] * 0.4
    face_bottom = fy + fh
    right_bound = fx + fw + horizontal_margin

    best_text, best_gap = None, None
    for text, _confidence, (x, y, w, h) in recognize_text_blocks(frame):
        text_center_x = x + w / 2
        if text_center_x > right_bound:
            continue
        gap = y - face_bottom
        if -20 <= gap <= max_vertical_gap:  # small overlap with the tile is fine
            if best_gap is None or gap < best_gap:
                best_text, best_gap = text, gap

    if best_text is None:
        return None, False

    is_self = bool(_YOU_SUFFIX.search(best_text))
    name = _clean_name(_YOU_SUFFIX.sub("", best_text))
    return name, is_self


def _clean_name(text):
    """A mute/status icon beside the name gets OCR'd as noise attached to it,
    inconsistently -- observed as "%", "will", "ll", "all %", "/", etc. Rather
    than pattern-match each variant, strip a leading glued-on symbol, then take
    from the first capitalized word onward (real display names start with a
    capital letter; the icon noise never does). Returns None if no such word
    is found."""
    text = re.sub(r"^[^A-Za-z0-9]+", "", text).strip(" -,")
    tokens = text.split()
    for i, token in enumerate(tokens):
        if token[:1].isupper():
            return " ".join(tokens[i:])
    return None
