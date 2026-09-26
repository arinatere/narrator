# Emotion Narrator

Turns a person's facial expression -- a cue sighted people read at a glance --
into audio a blind/low-vision user can pick up without looking at anything.

Built for **Hearing Hues**: sighted people read a conversation partner's
expression effortlessly and constantly (in a call, a classroom, a meeting).
A blind/low-vision person is cut off from that signal entirely. This watches
a face -- your own webcam, or a remote participant's video during a Zoom/Teams
call -- and relays what it sees through sound:

- An **instant earcon** the moment an expression change is confirmed: a short
  distinct tone per emotion (bright ascending chime for happy, a rougher low
  tone for angry/sad/fear/disgust, a soft click for neutral). This is the
  "glance" equivalent -- faster than a sentence.
- **Confidence-scaled loudness**: a shaky read plays quieter than a strong
  one, since a blind user can't sanity-check an uncertain guess the way a
  sighted person double-checks with their eyes.
- An **alert cue** layered on top when the subject shifts *into* a concerning
  emotion (angry/sad/fear/disgust) from a non-concerning one, so that moment
  stands out from routine happy/neutral drift -- similar in spirit to
  noticing when an AI agent needs your attention.
- A **spoken sentence** ("She looks happy") after the earcon, for full detail.

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

### Your own webcam
```bash
source venv/bin/activate
python emotion_narrator.py
```
- macOS will ask for Camera permission on first run (System Settings >
  Privacy & Security > Camera).
- A window shows the feed with a box + label around the detected face.
  Press `q` in that window to quit.

### A Zoom / Teams / FaceTime call (narrate the *other* person's expression)
```bash
python emotion_narrator.py --source window --app zoom.us
python emotion_narrator.py --source window --app Teams
python emotion_narrator.py --source window --app FaceTime
```
`--app` matches a substring of the app's window owner name -- whichever
window from that app is largest on screen is captured (usually the main call
window). This uses macOS screen capture, not any call app's API, so it works
with any video call app without API keys or app review.
- macOS will ask for **Screen Recording** permission on first run (System
  Settings > Privacy & Security > Screen Recording) -- grant it to your
  terminal (or the app running Python), then restart the script.

### Headless (audio only, no window)
Add `--headless` to either mode above. No video window opens at all; quit
with Ctrl+C. This is the intended mode for a blind/low-vision user, since it
has no dependency on seeing anything on screen.
```bash
python emotion_narrator.py --headless
python emotion_narrator.py --source window --app zoom.us --headless
```

## Notes

- `--camera-index` (default `1`) selects which webcam device to use in
  `webcam` mode. If you have an iPhone signed into the same Apple ID nearby,
  macOS may register it as a Continuity Camera at index `0`, which OpenCV can
  open but which returns black frames until actively engaged. If the feed is
  black, try `--camera-index 0` or check `system_profiler SPCameraDataType`
  for your actual device order.
- Narration uses macOS's `say` and window capture uses Quartz/`mss`, so this
  project is macOS-only as-is. Porting `webcam` mode + speech to another OS
  just means swapping `say` for another TTS engine (e.g. `pyttsx3`).
- Tuning constants (`FRAME_SKIP`, `HOLD_SECONDS`, `COOLDOWN_SECONDS`,
  `MIN_FACE_CONFIDENCE`) are at the top of `emotion_narrator.py`.
- Earcon waveforms and the concerning-emotion set live in `audio_cues.py`;
  capture logic lives in `capture.py`.
