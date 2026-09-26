"""Non-speech audio cues (earcons) that convey emotion category and confidence
almost instantly -- the "glance" equivalent for a blind/low-vision listener, who
can't verify a shaky read the way a sighted person double-checks with their eyes.

A single background worker plays, in order: an optional "alert" cue (for a
freshly-concerning emotion), the emotion earcon, then the spoken sentence --
so the fast non-verbal cue always lands before the slower narration.
"""

import math
import os
import queue
import struct
import subprocess
import tempfile
import threading
import wave

SAMPLE_RATE = 44100
CONCERNING_EMOTIONS = {"angry", "sad", "fear", "disgust"}

# (frequency Hz, duration seconds) notes played in sequence per emotion.
# Positive/neutral emotions use clean sine tones; concerning ones are rendered
# with a rougher timbre (see `harsh` in _tone_samples) so they're distinguishable
# by *quality*, not just pitch.
EMOTION_MOTIFS = {
    "happy":    [(880, 0.08), (1175, 0.12)],
    "surprise": [(1000, 0.05), (1400, 0.06)],
    "neutral":  [(500, 0.06)],
    "sad":      [(440, 0.15), (300, 0.20)],
    "angry":    [(220, 0.18)],
    "fear":     [(650, 0.05), (500, 0.05), (650, 0.05)],
    "disgust":  [(260, 0.15)],
}

# Two quick high ticks with a gap -- layered before the emotion earcon when the
# subject has just shifted INTO a concerning emotion, so that transition stands
# out from routine happy/neutral drift.
ALERT_MOTIF = [(1600, 0.04), (0, 0.03), (1600, 0.04)]


def _tone_samples(freq, duration, volume, harsh=False):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        if freq == 0:
            value = 0.0
        else:
            t = i / SAMPLE_RATE
            value = math.sin(2 * math.pi * freq * t)
            if harsh:
                # Blend in a square-wave harmonic for a rougher, more "alerting" timbre.
                value = 0.6 * value + 0.4 * (1.0 if value >= 0 else -1.0)
        fade = min(1.0, i / 200.0, (n - i) / 200.0)  # avoid clicks at edges
        samples.append(value * volume * fade)
    return samples


def _write_wav(path, motif, volume, harsh=False):
    all_samples = []
    for freq, duration in motif:
        all_samples.extend(_tone_samples(freq, duration, volume, harsh=harsh))
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        frames = b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in all_samples
        )
        f.writeframes(frames)


def _make_tone_file(motif, volume, harsh=False):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    _write_wav(path, motif, volume=volume, harsh=harsh)
    return path


_jobs = queue.Queue()


def _play_tone(path):
    try:
        subprocess.run(["afplay", path], check=False)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _worker():
    while True:
        job = _jobs.get()
        try:
            if job["alert_path"]:
                _play_tone(job["alert_path"])
            if job["tone_path"]:
                _play_tone(job["tone_path"])
            if job["phrase"]:
                subprocess.run(["say", job["phrase"]], check=False)
        except Exception as e:
            print(f"[AUDIO ERROR]: {e}")


threading.Thread(target=_worker, daemon=True).start()


def announce(emotion, confidence_pct, phrase=None, escalate=False):
    """Queue the earcon for `emotion` (plus an alert cue first if `escalate`),
    followed by an optional spoken `phrase`. `confidence_pct` (0-100) scales
    loudness so an uncertain read doesn't sound as confident as a strong one."""
    motif = EMOTION_MOTIFS.get(emotion)
    volume = 0.25 + 0.65 * max(0.0, min(1.0, confidence_pct / 100.0))
    harsh = emotion in CONCERNING_EMOTIONS

    alert_path = _make_tone_file(ALERT_MOTIF, volume=0.6) if escalate else None
    tone_path = _make_tone_file(motif, volume=volume, harsh=harsh) if motif else None

    _jobs.put({"alert_path": alert_path, "tone_path": tone_path, "phrase": phrase})
