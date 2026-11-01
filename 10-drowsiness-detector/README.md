# 👁️ Driver Drowsiness Detector

**Problem:** Drowsy driving causes a huge share of highway accidents in India — truckers driving overnight, students driving home after late nights. By the time you realize you're nodding off, it's too late.

**Solution:** Webcam watches the driver's eyes in real time. Eyes closed longer than 1.5 seconds → **loud alarm + red on-screen warning**. Logs how many alarms happened per trip.

## 🧠 The CV inside
- **dlib 68-point face landmark detector** finds both eyes every frame
- **Eye Aspect Ratio (EAR)** — the standard research method (Soukupová & Čech, 2016):

$$EAR = \frac{|p_2-p_6| + |p_3-p_5|}{2\,|p_1-p_4|}$$

- Open eyes → EAR ≈ 0.3; closed → EAR ≈ 0.1. Threshold 0.25, calibrated per person
- Classic CV math doing a safety-critical job at 30 fps on a laptop CPU — no heavy GPU needed

## ▶️ Run
```bash
pip install -r requirements.txt

# Download the landmark model (dlib's standard 68-point predictor):
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#   extract into this folder, then:

python main.py
```
Sit in front of your webcam, close your eyes for 2 seconds, and hear the alarm. 😄

## 🌍 Portfolio talking points
- Real research method (EAR paper), cited in README
- Safety-critical design: threshold + time-window to avoid false alarms from blinking
- Session logging → data for a "fatigue score" over long drives

## 🚀 Extension ideas
- Yawn detection (mouth landmark 48-68, same EAR trick for the jaw)
- Head-tilt detection (nodding off = head drops)
- Alert escalation: alarm → vibration → auto music
