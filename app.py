"""Menu-bar app wrapper around emotion_narrator: a small icon plus global
hotkeys, so a blind/low-vision user can start/stop narration during a call
without needing to see or click anything. All state changes are spoken.

    Option+Space  ->  start narrating in the background
    Option+Q      ->  stop

Run with: python app.py [--source window --app zoom.us | --source webcam]
"""

import argparse
import subprocess
import threading

import rumps
from pynput import keyboard

import emotion_narrator as narrator

START_HOTKEY = "<alt>+<space>"
STOP_HOTKEY = "<alt>+q"

IDLE_TITLE = "\U0001F442"       # ear
ACTIVE_TITLE = "\U0001F442\U0001F7E2"  # ear + green dot


def speak(text):
    subprocess.run(["say", text])


class NarratorApp(rumps.App):
    def __init__(self, source, app_name, camera_index, name):
        super().__init__("Narrator", title=IDLE_TITLE, quit_button="Quit")
        self.menu = ["Start", "Stop"]
        self.menu["Stop"].set_callback(None)  # disabled until narration is running

        self.args = argparse.Namespace(
            source=source, app=app_name, camera_index=camera_index,
            headless=True, name=name, enroll=None,
        )
        self.stop_event = threading.Event()
        self.thread = None

        self.hotkeys = keyboard.GlobalHotKeys({
            START_HOTKEY: self.start_narrating,
            STOP_HOTKEY: self.stop_narrating,
        })
        self.hotkeys.daemon = True
        self.hotkeys.start()

        speak("Narrator is ready. Press option space to start. Press option q to stop.")

    def start_narrating(self):
        if self.thread and self.thread.is_alive():
            return
        speak("Accessibility descriptions enabled.")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=narrator.run_narrator, args=(self.args, self.stop_event), daemon=True
        )
        self.thread.start()
        self.title = ACTIVE_TITLE
        self.menu["Start"].set_callback(None)
        self.menu["Stop"].set_callback(self.stop_clicked)

    def stop_narrating(self):
        if not (self.thread and self.thread.is_alive()):
            return
        self.stop_event.set()
        speak("Accessibility descriptions stopped.")
        self.title = IDLE_TITLE
        self.menu["Start"].set_callback(self.start_clicked)
        self.menu["Stop"].set_callback(None)

    @rumps.clicked("Start")
    def start_clicked(self, _sender):
        self.start_narrating()

    @rumps.clicked("Stop")
    def stop_clicked(self, _sender):
        self.stop_narrating()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["webcam", "window"], default="window")
    p.add_argument("--app", type=str, default="zoom.us",
                    help="Substring of the call app's window owner name (only used "
                         "with --source window). Default: zoom.us")
    p.add_argument("--camera-index", type=int, default=1)
    p.add_argument("--name", type=str, default="The speaker")
    args = p.parse_args()

    NarratorApp(args.source, args.app, args.camera_index, args.name).run()


if __name__ == "__main__":
    main()
