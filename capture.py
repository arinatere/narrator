"""Frame sources: a local webcam, or an on-screen call window (Zoom, Teams,
FaceTime, anything) captured live via macOS screen capture. Window capture lets
the analyzer read a call partner's face without any app-specific API."""

import numpy as np
import cv2
import mss

try:
    import Quartz
    _HAS_QUARTZ = True
except ImportError:
    _HAS_QUARTZ = False


class WebcamCapture:
    def __init__(self, index=1):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {index}.")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class WindowCapture:
    """Captures whichever on-screen window belongs to an app whose name matches
    `app_substring` (e.g. 'zoom.us', 'Teams', 'FaceTime'). Requires the Screen
    Recording permission (System Settings > Privacy & Security > Screen Recording)
    to be granted to the terminal/Python running this."""

    def __init__(self, app_substring):
        if not _HAS_QUARTZ:
            raise RuntimeError("Window capture requires pyobjc-framework-Quartz (macOS only).")
        self.app_substring = app_substring.lower()
        self.sct = mss.mss()
        if self._find_window_bounds() is None:
            raise RuntimeError(
                f"No on-screen window found matching '{app_substring}'. "
                "Make sure the call is open and visible, and that Screen Recording "
                "permission is granted to your terminal/Python."
            )

    def _find_window_bounds(self):
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        candidates = []
        for w in window_list:
            owner = (w.get('kCGWindowOwnerName', '') or '')
            if self.app_substring in owner.lower():
                bounds = w.get('kCGWindowBounds')
                if bounds and bounds['Width'] > 100 and bounds['Height'] > 100:
                    candidates.append(bounds)
        if not candidates:
            return None
        # Assume the largest matching window is the main call window, not a
        # toolbar/notification panel from the same app.
        candidates.sort(key=lambda b: b['Width'] * b['Height'], reverse=True)
        return candidates[0]

    def read(self):
        bounds = self._find_window_bounds()
        if bounds is None:
            return False, None
        monitor = {
            "left": int(bounds['X']),
            "top": int(bounds['Y']),
            "width": int(bounds['Width']),
            "height": int(bounds['Height']),
        }
        img = self.sct.grab(monitor)
        frame = np.array(img)[:, :, :3]  # BGRA -> BGR
        return True, frame

    def release(self):
        self.sct.close()
