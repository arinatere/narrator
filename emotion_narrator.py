"""Turns a person's facial expression -- a cue sighted people read at a glance --
into audio a blind/low-vision user can pick up without looking at anything.

Source is either a local webcam or a live capture of a call window (Zoom, Teams,
FaceTime, ...), so it can also narrate a remote participant's expression during
a call. See README.md for setup and permissions.
"""

import argparse
import time

import cv2
from deepface import DeepFace

import audio_cues
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
    args = p.parse_args()
    if args.source == "window" and not args.app:
        p.error("--source window requires --app <substring>, e.g. --app zoom.us")
    return args


def make_capture(args):
    if args.source == "webcam":
        return WebcamCapture(index=args.camera_index)
    return WindowCapture(args.app)


def main():
    args = parse_args()
    cap = make_capture(args)

    frame_count = 0
    current_emotion = None
    current_confidence = 0.0
    last_box = None

    pending_emotion = None      # the emotion currently being observed
    pending_since = time.time() # when we started observing it
    last_announced = None       # the last emotion we actually spoke aloud
    last_speak_time = 0

    print("(First run may take ~30 seconds while DeepFace downloads its model)")
    if args.headless:
        print("Running headless -- audio only. Press Ctrl+C to quit.")
    else:
        print("Press 'q' in the video window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the source.")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True,
                    )
                    if isinstance(result, list):
                        result = result[0]

                    face_confidence = result.get('face_confidence', 1.0)
                    dominant = result['dominant_emotion']
                    if face_confidence < MIN_FACE_CONFIDENCE:
                        # Not confident a face was even found -- don't relay a guess
                        # the listener has no way to visually double-check.
                        current_emotion = None
                    else:
                        current_emotion = dominant
                        current_confidence = result['emotion'][dominant]
                    last_box = result['region']
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
                    phrase=f"She looks {pending_emotion}",
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


if __name__ == "__main__":
    main()
