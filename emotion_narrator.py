"""Turns a person's facial expression -- a cue sighted people read at a glance --
into audio a blind/low-vision user can pick up without looking at anything.

Source is either a local webcam or a live capture of a call window (Zoom, Teams,
FaceTime, ...), so it can also narrate a remote participant's expression during
a call. In a multi-person call, the largest detected face is treated as the
active speaker (Speaker View auto-enlarges whoever's talking) and everyone else
is ignored -- see README.md for the Gallery View caveat and setup/permissions.
"""

import argparse
import select
import subprocess
import sys
import termios
import threading
import time
import tty

import cv2
from deepface import DeepFace

import audio_cues
import ocr
import people
from capture import WebcamCapture, WindowCapture

FRAME_SKIP = 5               # only run detection every 5th frame
HOLD_SECONDS = 1.5           # how long an expression must hold before it's a "real" change
COOLDOWN_SECONDS = 3         # minimum gap between announcements
MIN_FACE_CONFIDENCE = 0.5    # detector's face_confidence is a 0-1 fraction, not a percentage


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["webcam", "window"], default="webcam",
                    help="Frame source: local webcam, or a captured on-screen call window.")
    p.add_argument("--camera-index", type=int, default=1,
                    help="Webcam device index (macOS often puts an inactive Continuity "
                         "Camera at 0; the real built-in camera is usually 1).")
    p.add_argument("--app", type=str, default=None,
                    help="Substring of the target app's window owner name for "
                         "--source window, e.g. 'zoom.us', 'Teams', 'FaceTime'.")
    p.add_argument("--headless", action="store_true",
                    help="Don't open a video window -- audio only. Quit with Ctrl+C.")
    p.add_argument("--name", type=str, default="The speaker",
                    help="Fallback label used in narration when the current "
                         "speaker isn't a recognized, enrolled face (default: "
                         "'The speaker'). See --enroll to register real names.")
    p.add_argument("--enroll", type=str, default=None, metavar="NAME",
                    help="Enroll a person instead of running detection: grabs a "
                         "reference frame from --source (point it at their face "
                         "first) and stores it under NAME for future narration.")
    p.add_argument("--interactive", action="store_true",
                    help="Wait for a keypress instead of starting immediately: Space "
                         "starts narration in the background, Q stops and exits. "
                         "Runs headless (no video window) with spoken confirmations "
                         "at each step. Keys are only read while this terminal has "
                         "focus -- no special OS permission needed.")
    args = p.parse_args()
    if args.source == "window" and not args.app:
        p.error("--source window requires --app <substring>, e.g. --app zoom.us")
    return args


def make_capture(args):
    if args.source == "webcam":
        return WebcamCapture(index=args.camera_index)
    return WindowCapture(args.app)


def crop_region(frame, region, pad_ratio=0.2):
    """Crop `region` (a DeepFace 'region' dict) out of `frame` with a bit of
    padding, for a cleaner face-identification embedding than a tight box."""
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    y0, y1 = max(0, y - pad_y), min(frame.shape[0], y + h + pad_y)
    x0, x1 = max(0, x - pad_x), min(frame.shape[1], x + w + pad_x)
    return frame[y0:y1, x0:x1]


def run_enroll(args):
    cap = make_capture(args)
    try:
        frame = None
        for _ in range(10):  # let the source warm up (webcam exposure, window focus)
            ret, f = cap.read()
            if ret:
                frame = f
            time.sleep(0.1)
    finally:
        cap.release()

    if frame is None:
        print("Could not capture a frame to enroll from.")
        return
    people.enroll(args.enroll, frame)
    print(f"Enrolled '{args.enroll}'. Known speakers: {', '.join(people.known_names())}")


def run_narrator(args, stop_event=None):
    """Runs the detection loop until `stop_event` is set (or, with no
    stop_event, until Ctrl+C / 'q' in the video window). `stop_event` lets
    run_interactive() below stop this cooperatively from a keypress instead."""
    if stop_event is None:
        stop_event = threading.Event()

    cap = make_capture(args)
    identifier = people.Identifier()

    frame_count = 0
    current_emotion = None
    current_confidence = 0.0
    current_subject = args.name
    last_box = None

    pending_emotion = None      # the emotion currently being observed
    pending_since = time.time() # when we started observing it
    last_announced = None       # the last emotion we actually spoke aloud
    last_speak_time = 0

    print("(First run may take ~30 seconds while DeepFace downloads its model)")
    if args.source == "window":
        print("Reading on-screen name labels for identification (falls back to "
              "enrolled faces, then --name, if no label is found nearby).")
    if identifier.has_known_faces():
        print(f"Known speakers: {', '.join(people.known_names())}")
    elif args.source != "window":
        print("No speakers enrolled yet -- run with --enroll NAME to add one. "
              f"Unrecognized speakers will be called '{args.name}'.")
    if args.headless:
        print("Running headless -- audio only. Press Ctrl+C to quit.")
    else:
        print("Press 'q' in the video window to quit.")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the source.")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                try:
                    results = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True,
                    )
                    if not isinstance(results, list):
                        results = [results]

                    # analyze() returns one entry per detected face. In a multi-person
                    # call the active speaker's tile is usually the largest one on
                    # screen (Speaker View auto-enlarges whoever's talking), so treat
                    # the largest detected face as the speaker and ignore the rest.
                    speaker = max(results, key=lambda r: r['region']['w'] * r['region']['h'])

                    face_confidence = speaker.get('face_confidence', 1.0)
                    dominant = speaker['dominant_emotion']
                    if face_confidence < MIN_FACE_CONFIDENCE:
                        # Not confident a face was even found -- don't relay a guess
                        # the listener has no way to visually double-check.
                        current_emotion = None
                    else:
                        ocr_name, is_self = (None, False)
                        if args.source == "window":
                            # Call apps overlay each participant's display name on
                            # their tile -- read it directly instead of requiring
                            # manual enrollment. A "(you)" label means this tile is
                            # the local user, who doesn't need their own expression
                            # narrated back to them.
                            ocr_name, is_self = ocr.find_name_near_face(frame, speaker['region'])

                        if is_self:
                            current_emotion = None
                        else:
                            current_emotion = dominant
                            current_confidence = speaker['emotion'][dominant]
                            if ocr_name:
                                current_subject = ocr_name
                            elif identifier.has_known_faces():
                                matched_name = identifier.identify(crop_region(frame, speaker['region']))
                                current_subject = matched_name if matched_name else args.name
                            else:
                                current_subject = args.name
                    last_box = speaker['region']
                except Exception as e:
                    current_emotion = None
                    print(f"[DEBUG] Detection error: {e}")

            if not args.headless:
                if current_emotion and last_box and last_box['w'] > 0:
                    (x, y, w, h) = last_box['x'], last_box['y'], last_box['w'], last_box['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, current_emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Emotion Narrator (press q to quit)", frame)

            if current_emotion != pending_emotion:
                # A new candidate emotion appeared - start timing how long it holds
                pending_emotion = current_emotion
                pending_since = time.time()
            elif (pending_emotion
                  and pending_emotion != last_announced
                  and (time.time() - pending_since) > HOLD_SECONDS
                  and (time.time() - last_speak_time) > COOLDOWN_SECONDS):
                escalate = (pending_emotion in audio_cues.CONCERNING_EMOTIONS
                            and last_announced not in audio_cues.CONCERNING_EMOTIONS)
                audio_cues.announce(
                    pending_emotion,
                    current_confidence,
                    phrase=f"{current_subject} looks {pending_emotion}",
                    escalate=escalate,
                )
                last_announced = pending_emotion
                last_speak_time = time.time()

            if not args.headless:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


def _read_key(timeout=0.2):
    """Returns one character read from stdin if available within `timeout`
    seconds, else None. Only sees keys while this terminal is focused --
    unlike a global hotkey, this needs no special OS permission."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def run_interactive(args):
    """Space starts narration in the background; Q stops it and exits. Runs
    headless regardless of --headless, since a video window has no place in
    a keypress-driven, audio-only workflow."""
    args.headless = True
    subprocess.run(["say", "Narrator is ready. Press space to start. Press q to stop."])
    print("Narrator is ready. Press Space to start. Press Q to stop.")

    stop_event = threading.Event()
    thread = None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key()
            if key is None:
                continue
            if key == " ":
                if thread and thread.is_alive():
                    continue
                subprocess.run(["say", "Accessibility descriptions enabled."])
                stop_event = threading.Event()
                thread = threading.Thread(target=run_narrator, args=(args, stop_event), daemon=True)
                thread.start()
            elif key.lower() == "q":
                if thread and thread.is_alive():
                    stop_event.set()
                    thread.join(timeout=2)
                subprocess.run(["say", "Narrator stopped."])
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    args = parse_args()
    if args.enroll:
        run_enroll(args)
    elif args.interactive:
        run_interactive(args)
    else:
        run_narrator(args)


if __name__ == "__main__":
    main()
