import cv2
from deepface import DeepFace
import time
import threading
import queue
import subprocess

speech_queue = queue.Queue()

def _speech_worker():
    while True:
        text = speech_queue.get()
        try:
            print(f"[Narrating]: {text}")
            subprocess.run(["say", text])
        except Exception as e:
            print(f"[TTS ERROR]: {e}")

threading.Thread(target=_speech_worker, daemon=True).start()

def speak_async(text):
    speech_queue.put(text)

CAMERA_INDEX = 1  # 0 is often an inactive Continuity Camera (iPhone) on macOS; 1 is the built-in camera
cap = cv2.VideoCapture(CAMERA_INDEX)

FRAME_SKIP = 5   # only run detection every 5th frame
frame_count = 0
current_emotion = None
last_box = None

pending_emotion = None      # the emotion currently being observed
pending_since = time.time() # when we started observing it
last_announced = None       # the last emotion we actually spoke aloud
last_speak_time = 0
HOLD_SECONDS = 1.5          # how long an expression must hold before it's a "real" change
COOLDOWN_SECONDS = 3

print("Press 'q' in the video window to quit.")
print("(First run may take ~30 seconds while DeepFace downloads its model)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            if isinstance(result, list):
                result = result[0]

            current_emotion = result['dominant_emotion']
            last_box = result['region']
        except Exception as e:
            current_emotion = None
            print(f"[DEBUG] Detection error: {e}")

    # Draw the most recent known box/label every frame (even skipped ones)
    # so the video doesn't look choppy
    if current_emotion and last_box and last_box['w'] > 0:
        (x, y, w, h) = last_box['x'], last_box['y'], last_box['w'], last_box['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, current_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if current_emotion != pending_emotion:
        # A new candidate emotion appeared - start timing how long it holds
        pending_emotion = current_emotion
        pending_since = time.time()
    elif (pending_emotion
          and pending_emotion != last_announced
          and (time.time() - pending_since) > HOLD_SECONDS
          and (time.time() - last_speak_time) > COOLDOWN_SECONDS):
        speak_async(f"She looks {pending_emotion}")
        last_announced = pending_emotion
        last_speak_time = time.time()

    cv2.imshow("Emotion Narrator (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
