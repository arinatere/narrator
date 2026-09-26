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
- A **spoken sentence** ("Alex looks happy") after the earcon, for full detail
  -- naming the speaker using whatever name the call app already displays on
  their tile (see Named speakers below), or a fallback label otherwise.
- **Speaker focus** in multi-person calls: the largest detected face is treated
  as the active speaker and everyone else is ignored, so a busy call doesn't
  turn into a wall of narration.
- **Stays silent when you're the speaker**, if the call app labels your own
  tile with a "(you)" marker (see Named speakers).

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

### Named speakers
In `--source window` mode, narration reads the participant name Zoom/Teams
already display under the active speaker's tile -- via on-device OCR (macOS's
Vision framework, no cloud calls, no extra model download) -- so no setup is
needed for this to say real names. If the call app marks your own tile "Name
(You)", that tile is recognized as you and stays silent instead of narrating
your own expression back to you.

If a name label can't be read (covered, too small, an app that doesn't show
one), it falls back to face enrollment: point the source at someone's face
once and register them by name --
```bash
python emotion_narrator.py --source window --app zoom.us --enroll Alex
```
This grabs a reference frame, stores a face embedding for "Alex" locally in
`known_faces.json` (never committed -- it's gitignored, since it's personal
biometric data), and exits. On future runs, a detected speaker matching an
enrolled face is called by that name. If neither an on-screen label nor an
enrollment matches, it falls back to `--name <label>` (default: "The
speaker").
```bash
python emotion_narrator.py --source window --app zoom.us --headless
python emotion_narrator.py --source window --app zoom.us --headless --name "Someone new"
```

### Headless (audio only, no window)
Add `--headless` to either mode above. No video window opens at all; quit
with Ctrl+C. This is the intended mode for a blind/low-vision user, since it
has no dependency on seeing anything on screen.
```bash
python emotion_narrator.py --headless
python emotion_narrator.py --source window --app zoom.us --headless
```

### Interactive mode (keypress-controlled, spoken confirmations)
```bash
python emotion_narrator.py --source window --app zoom.us --interactive
```
Launches and waits rather than narrating immediately -- entirely keyboard and
voice driven, no video window, no mouse needed:
- On launch, it speaks: *"Narrator is ready. Press space to start. Press q to
  stop."*
- **Space** starts narration in the background and it speaks: *"Accessibility
  descriptions enabled."*
- **Q** stops it, speaks a confirmation, and exits.

Keys are read only while this terminal window has focus (standard terminal
input, via Python's `termios`/`tty` -- no extra dependency, no special OS
permission, unlike a true system-wide hotkey).

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
  capture logic lives in `capture.py`; on-screen name-label reading lives in
  `ocr.py` (macOS Vision framework); face enrollment/identification (the
  fallback when no label is found) lives in `people.py` (embeddings via
  DeepFace's Facenet model, matched by cosine distance). None of this needs
  extra model downloads or a system OCR binary like Tesseract.
- **Speaker focus assumes Speaker View** (whoever's talking auto-enlarges to
  the main tile), which is the default or a one-click switch in Zoom/Teams. In
  Gallery View, all tiles are equal-sized, so the "largest face" heuristic has
  no signal to go on and may pick the wrong person.
- **Name-label reading is a proximity heuristic**: it picks whichever text
  Vision finds just below (and not too far right of) the speaker's face, since
  call apps anchor the name to a tile's bottom-left corner regardless of where
  the face sits within it. Works well for one tile filling most of the frame;
  in a dense Gallery View it can occasionally grab a neighboring tile's label.
- **Self-filtering depends on the call app rendering a "(you)" marker.** Some
  apps/clients don't (e.g. Zoom's web client shows your name with no such
  marker), in which case it narrates you like anyone else -- there is no
  visual cue to key off in that case.
