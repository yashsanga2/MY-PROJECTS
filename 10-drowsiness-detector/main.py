"""
DRIVER DROWSINESS DETECTOR
Problem: Drowsy driving causes a large share of night road accidents in India.
By the time a tired driver realizes they're nodding off, it's already dangerous.

Solution: A webcam watches the driver's eyes in real time:
  - Eyes closed > 1.5 seconds → audio ALARM + red warning
  - Tracks EAR (Eye Aspect Ratio) per frame — the standard research approach
    (Soukupová & Čech, 2016)

Model/tech: dlib's 68-point face landmark predictor + Eye Aspect Ratio (EAR):
    EAR = (|p2-p6| + |p3-p5|) / (2·|p1-p4|)
Eyes open → EAR ≈ 0.3 ; eyes closed → EAR ≈ 0.1
It's a beautiful example of classic CV math beating heavy deep nets for a
specific task — great story for interviews.

  python main.py
"""

import time

EAR_THRESHOLD = 0.25      # below this → eye considered closed
CLOSED_SECONDS = 1.5      # closed longer than this → alarm

# dlib 68-point layout: left eye 36-41, right eye 42-47
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def eye_aspect_ratio(eye_points) -> float:
    """eye_points: 6 (x, y) tuples → EAR value."""
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical = (p2[1] - p6[1]) + (p3[1] - p5[1])
    horizontal = abs(p1[0] - p4[0])
    if horizontal == 0:
        return 0.0
    return vertical / (2 * horizontal)


def alarm():
    """Cross-platform beep."""
    import sys
    try:
        if sys.platform.startswith("win"):
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 300)
        else:
            import os
            for _ in range(3):
                os.system("afplay /System/Library/Sounds/Glass.aiff 2>/dev/null"
                          if sys.platform == "darwin"
                          else "play -q /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || paplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || printf '\\a'")
    except Exception:
        print("\a\a\a  ⚠️  WAKE UP!  ⚠️")


def run():
    import cv2
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    closed_since = None
    alarms_today = 0

    print(" Watching... press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            left = [(shape.part(i).x, shape.part(i).y) for i in LEFT_EYE]
            right = [(shape.part(i).x, shape.part(i).y) for i in RIGHT_EYE]

            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2

            # draw eye outlines
            for pts in (left, right):
                pts_arr = cv2.convexHull(__import__("numpy").array(pts, dtype="int32"))
                cv2.drawContours(frame, [pts_arr], -1, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                closed_since = closed_since or time.time()
                closed_for = time.time() - closed_since
                color, msg = (0, 0, 255), f"EYES CLOSED {closed_for:.1f}s"
                if closed_for > CLOSED_SECONDS:
                    alarm()
                    alarms_today += 1
                    msg += "  ⚠️ WAKE UP!"
            else:
                closed_since = None
                color, msg = (0, 255, 0), f"EAR {ear:.2f} — alert"

            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"alarms: {alarms_today}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Drowsiness Detector (q=quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session summary: {alarms_today} drowsiness alarm(s). Drive safe!")


if __name__ == "__main__":
    run()