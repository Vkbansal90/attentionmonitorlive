# Attention Monitor Live

Real-time student attention monitoring with computer vision. A webcam (and optionally a microphone) is analysed frame by frame to score how attentive a person is. The system detects closed eyes, yawns, looking away, slouching, fidgeting, phone use and noise, then turns them into a live score, alerts and a session report.

**Live demo:** https://vkbansal90.github.io/attentionmonitorlive/

The demo runs entirely in the visitor's browser. Video and audio never leave the device. If you have no camera, click **View a sample report** to see the analytics on simulated data.

## Features

| | |
|---|---|
| **Eyes and drowsiness** | Eye aspect ratio (EAR) with a calibrated threshold, blink rate per minute, PERCLOS (percentage of time eyes are closed), alert on sustained eye closure |
| **Yawning** | Mouth aspect ratio combined with the jaw-open blendshape; yawns are counted per session |
| **Gaze / head pose** | 3D yaw, pitch and roll from the face transformation matrix, measured against a neutral baseline from the first 2 seconds |
| **Posture** | Neck angle and shoulder tilt (as in the Python version), plus slouch detection from nose-to-shoulder height |
| **Hands** | Hand-on-face distance and fidgeting (rapid fingertip movement) |
| **Phone detection** | EfficientDet-Lite object detector restricted to the COCO `cell phone` class |
| **Multiple people** | Flags when more than one face is in frame |
| **Audio** | RMS noise level from the microphone (optional) |
| **Alerts** | Banner and sound when the score stays under a user-set threshold for 5 s, naming the cause |
| **Session report** | Focus grade, average and peak score, time focused, longest focus streak, timeline with event markers, state breakdown, event log, CSV / JSON export |
| **History** | Past sessions are kept in the browser (web) or in SQLite with CSV export (desktop) |

## Scoring

```
total = 0.6 · face + 0.2 · posture + 0.2 · audio      (microphone off: 0.75 · face + 0.25 · posture)
```

Each component starts at 1.0 and is multiplied by a penalty for every active cue, for example ×0.7 for eyes closed, ×0.55 while yawning (then ×0.85 for 30 s of fatigue), ×0.75 for looking away for more than 1.5 s and ×0.8 for slouching. The face score is smoothed over the last 30 frames. A phone in view halves the total. A score of 0.70 or more counts as *focused*, 0.40 to 0.70 as *partially attentive*, and anything lower as *distracted*.

## Project layout

```
web/                 Browser version (hosted on GitHub Pages)
  index.html         UI
  app.js             MediaPipe Tasks pipeline, scoring, alerts, reports
  style.css
src/                 Desktop version (Python)
  app.py             Streamlit dashboard
  face_analyzer.py   dlib 68-point landmarks: EAR, MAR, head tilt, hand distance
  posture_detector.py MediaPipe Pose and Hands: neck and shoulder angles, fidgeting
  audio_analyzer.py  sounddevice and librosa: noise and yawn audio cues
  utils.py           SQLite logging and statistics
  setup_models.py    Downloads the dlib shape predictor if missing
data/attention_logs.db  SQLite log of desktop sessions
models/              dlib shape_predictor_68_face_landmarks.dat
```

## Run the web version locally

```bash
python -m http.server 8000 --directory web
```

Then open http://localhost:8000. Browsers allow camera access on `localhost` and over HTTPS.

## Run the desktop version

Requires Python 3.10.

```bash
python -m venv venv
venv\Scripts\activate          # Windows (use source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
cd src
streamlit run app.py
```

`setup_models.py` downloads the dlib landmark model on first run if `models/` is empty.

## Tech stack

- **Web:** MediaPipe Tasks Vision (Face Landmarker, Pose Landmarker, Hand Landmarker, Object Detector) on WebGL/WASM, Web Audio API, Canvas, vanilla JavaScript, GitHub Pages
- **Desktop:** Python, OpenCV, dlib, MediaPipe, librosa, sounddevice, Streamlit, SQLite, pandas
