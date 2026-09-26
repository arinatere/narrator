# Emotion Narrator

Watches your webcam, detects the dominant facial emotion with DeepFace, and
narrates emotion changes out loud using macOS's `say` command.

## Setup

```bash
cd emotion-narrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First run of DeepFace will download the emotion model (~6 MB) to
`~/.deepface/weights/`.

## Run

```bash
source venv/bin/activate
python emotion_narrator.py
```

- A window opens showing the webcam feed with a box + label around the
  detected face.
- Press `q` in that window to quit.
- macOS will ask for Camera permission on first run (System Settings >
  Privacy & Security > Camera).

## Notes

- `CAMERA_INDEX` at the top of `emotion_narrator.py` selects which webcam to
  use. If you have an iPhone signed into the same Apple ID nearby, macOS may
  register it as a Continuity Camera at index `0`, which OpenCV can open but
  which returns black frames until actively engaged. Index `1` is usually the
  real built-in camera in that case — if the video feed is black, try
  changing this value.

- Narration uses `say`, so this only speaks on macOS as-is. On other
  platforms, swap `subprocess.run(["say", text])` in `emotion_narrator.py`
  for another TTS engine (e.g. `pyttsx3`).
- `FRAME_SKIP`, `HOLD_SECONDS`, and `COOLDOWN_SECONDS` at the top of the
  script control detection frequency and how "sticky" an emotion must be
  before it's announced.
