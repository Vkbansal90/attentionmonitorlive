import {
  FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker, ObjectDetector, DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

// ---------------------------------------------------------------------------
// Configuration: thresholds ported from src/face_analyzer.py, posture_detector.py
// and audio_analyzer.py, plus the new browser-only signals.
// ---------------------------------------------------------------------------
const CFG = {
  EAR_THRESH: 0.2,          // eye aspect ratio below this = eyes closed
  MAR_THRESH: 0.5,          // mouth aspect ratio above this = mouth wide open
  ROLL_THRESH: 30,          // head tilt (degrees), same as HEAD_POSE_THRESH
  YAW_THRESH: 30,           // looking left/right (degrees from calibrated baseline)
  PITCH_THRESH: 25,         // looking up/down (degrees from calibrated baseline)
  LOOK_AWAY_GRACE_MS: 1500, // short glances (e.g. at notes) are not penalised
  YAWN_FATIGUE_MS: 30000,   // a yawn keeps lowering the face score for 30 s
  HAND_DIST_THRESH: 0.23,   // 150 px at 640 px width, as a fraction of frame width
  NECK_THRESH: 20,
  SHOULDER_THRESH: 15,
  SLOUCH_RATIO: 0.8,        // nose-to-shoulder height drops below 80% of baseline
  FIDGET_MOVE: 0.047,       // 30 px per frame at 640 px width
  FIDGET_HISTORY: 10,
  AUDIO_CALIB_MS: 2000,     // room's quiet level is measured over the first 2 s with the mic on
  SPEECH_MIN_DB: 5,         // speech must be this many dB above the room's quiet level
  NOISE_MIN_DB: 10,         // non-speech sound this loud above the quiet level = noisy room
  CALIB_MS: 2000,           // neutral head/posture baseline is measured over the first 2 s
  CALIB_MIN_FRAMES: 5,
  SMOOTH_HISTORY: 30,
  WEIGHTS: { face: 0.6, posture: 0.2, audio: 0.2 },
  WEIGHTS_NO_MIC: { face: 0.75, posture: 0.25, audio: 0 },
  PHONE_PENALTY: 0.5,
  ALERT_AFTER_MS: 5000,
  ALERT_COOLDOWN_MS: 15000,
  SAMPLE_MS: 250,
};

const MP_VERSION = '0.10.14';
const MODELS = {
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  phone: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
};

// Face mesh indices equivalent to dlib's 6-point eye and mouth landmarks.
const L_EYE = [33, 160, 158, 133, 153, 144];
const R_EYE = [362, 385, 387, 263, 373, 380];

// Signals shown in the side panel. `minMs` is how long a condition must last
// before it is logged as a distraction event.
const SIGNALS = {
  noFace:     { label: 'Away from camera', minMs: 2000 },
  lookAway:   { label: 'Looking away', minMs: 2000 },
  eyesClosed: { label: 'Eyes closed', minMs: 1500, event: 'Drowsy (eyes closed)' },
  yawn:       { label: 'Yawning', minMs: 1000 },
  headTilt:   { label: 'Head tilted', minMs: 3000 },
  handFace:   { label: 'Hand on face', minMs: 2500 },
  slouch:     { label: 'Slouching', minMs: 4000 },
  fidget:     { label: 'Fidgeting', minMs: 1500 },
  phone:      { label: 'Phone in view', minMs: 1000 },
  multiFace:  { label: 'Multiple people', minMs: 1500 },
  talking:    { label: 'Talking', minMs: 2000 },
  noise:      { label: 'Noisy room', minMs: 2500 },
};
const DISTRACTION_KEYS = Object.keys(SIGNALS);

const $ = id => document.getElementById(id);
const els = {
  video: $('video'), overlay: $('overlay'), placeholder: $('placeholder'), loadStatus: $('loadStatus'),
  start: $('startBtn'), stop: $('stopBtn'), sample: $('sampleBtn'),
  mic: $('micToggle'), phone: $('phoneToggle'), sound: $('soundToggle'), showOverlay: $('overlayToggle'),
  thr: $('thr'), thrOut: $('thrOut'), alert: $('alertBanner'), alertReason: $('alertReason'),
  stateChip: $('stateChip'), calibChip: $('calibChip'), timer: $('timer'), fps: $('fps'),
  total: $('totalScore'), stateLabel: $('stateLabel'), gauge: $('gaugeArc'),
  faceBar: $('faceBar'), postureBar: $('postureBar'), audioBar: $('audioBar'),
  faceVal: $('faceVal'), postureVal: $('postureVal'), audioVal: $('audioVal'), weightNote: $('weightNote'),
  audioMeterRow: $('audioMeterRow'), audioMeter: $('audioMeter'), audioFloor: $('audioFloor'), audioState: $('audioState'),
  signals: $('signals'), trend: $('trend'),
  statBlink: $('statBlink'), statPerclos: $('statPerclos'), statYawns: $('statYawns'),
  statDistractions: $('statDistractions'), statFocus: $('statFocus'), statStreak: $('statStreak'),
  history: $('historyTable'), clearHistory: $('clearHistory'),
  report: $('report'),
};

let tasks = null;        // loaded MediaPipe tasks
let S = null;            // current session state
let stream = null;
let audio = null;        // microphone analysis state, see enableMic()
let audioCtx = null;     // shared Web Audio context (mic analysis and alert beep)
let tickLoaded = false;

// Posts a message every 2048 samples (~20 per second). Unlike timers and
// animation frames, the audio thread is not throttled in background tabs,
// so the microphone keeps being analysed while the student is in another tab.
const TICK_WORKLET = `registerProcessor('tick', class extends AudioWorkletProcessor {
  n = 0;
  process() {
    this.n += 128;
    if (this.n >= 2048) { this.n = 0; this.port.postMessage(0); }
    return true;
  }
});`;
let lastVideoTime = -1;
let lastReport = null;

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
async function createTask(Cls, options) {
  try {
    return await Cls.createFromOptions(tasks.fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } });
  } catch (err) {
    console.warn(`${Cls.name}: GPU delegate failed, using CPU`, err);
    return Cls.createFromOptions(tasks.fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
  }
}

async function loadModels() {
  if (tasks?.face) return;
  setStatus('Loading vision models (about 15 MB, first time only)…');
  tasks = { fileset: await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`) };
  const [face, pose, hand] = await Promise.all([
    createTask(FaceLandmarker, {
      baseOptions: { modelAssetPath: MODELS.face }, runningMode: 'VIDEO', numFaces: 2,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    }),
    createTask(PoseLandmarker, { baseOptions: { modelAssetPath: MODELS.pose }, runningMode: 'VIDEO', numPoses: 1 }),
    createTask(HandLandmarker, { baseOptions: { modelAssetPath: MODELS.hand }, runningMode: 'VIDEO', numHands: 2 }),
  ]);
  Object.assign(tasks, { face, pose, hand });
  // The first inference compiles GPU shaders and can take several seconds;
  // do it now so it doesn't stall the start of the session.
  setStatus('Warming up models…');
  const warm = Object.assign(document.createElement('canvas'), { width: 64, height: 64 });
  const t = performance.now();
  face.detectForVideo(warm, t);
  pose.detectForVideo(warm, t);
  hand.detectForVideo(warm, t);
  loadPhoneDetector();
}

async function loadPhoneDetector() {
  if (tasks.phone || tasks.phoneLoading) return;
  tasks.phoneLoading = true;
  try {
    tasks.phone = await createTask(ObjectDetector, {
      baseOptions: { modelAssetPath: MODELS.phone }, runningMode: 'VIDEO',
      scoreThreshold: 0.35, maxResults: 2, categoryAllowlist: ['cell phone'],
    });
  } catch (err) {
    console.warn('Phone detector unavailable', err);
  }
  tasks.phoneLoading = false;
}

function setStatus(text, isError = false) {
  els.loadStatus.textContent = text;
  els.loadStatus.classList.toggle('error', isError);
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
function newSession() {
  return {
    startedAt: Date.now(), t0: performance.now(), lastT: performance.now(), lastSample: 0, frames: 0,
    samples: [],                 // { t, total, face, posture, audio }
    events: [],                  // { t, key, label }
    active: {},                  // key -> onset time (ms) while the condition holds
    logged: {},                  // key -> true once the current episode is logged
    current: {},                 // key -> boolean, raw per-frame state
    faceHist: [], stateMs: { focused: 0, partial: 0, distracted: 0 },
    calib: { done: false, n: 0, yaw: 0, pitch: 0, ear: 0, neck: [], slouch: [] },
    eye: { closed: false, since: 0, blinks: [], perclos: [] },
    yawns: 0, fatigueUntil: 0, lookSince: null, blinkTotal: 0, handPrev: null, handMoves: [],
    phoneSeenAt: -1e9, phoneBoxes: [], audioMs: { talking: 0, noisy: 0, total: 0 },
    lowSince: null, lastAlert: -1e9, streakStart: null, bestStreak: 0,
    last: null, fpsFrames: 0, fpsT: performance.now(),
  };
}

async function start() {
  primeAudio();
  els.start.disabled = true;
  els.sample.disabled = true;
  try {
    await loadModels();
    setStatus('Waiting for camera permission…');
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
    els.video.srcObject = stream;
    // play() can stay pending in throttled tabs, so wait for decoded frames instead.
    els.video.play().catch(err => console.warn('video.play()', err));
    if (els.video.readyState < 2) await new Promise(r => els.video.addEventListener('loadeddata', r, { once: true }));
    if (els.mic.checked) await enableMic();
  } catch (err) {
    console.error(err);
    const msg = err?.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow camera access in your browser and try again, or view the sample report.'
      : err?.name === 'NotFoundError'
        ? 'No camera was found on this device. You can still view the sample report.'
        : `Could not start: ${err?.message || err}`;
    setStatus(msg, true);
    els.start.disabled = false;
    els.sample.disabled = false;
    stopStreams();
    return;
  }
  S = newSession();
  lastVideoTime = -1;
  setStatus('');
  els.placeholder.classList.add('hidden');
  els.stop.disabled = false;
  els.calibChip.classList.remove('hidden');
  schedule();
}

function stop() {
  if (!S) return;
  const session = S;
  S = null;
  stopStreams();
  els.stop.disabled = true;
  els.start.disabled = false;
  els.sample.disabled = false;
  els.start.textContent = 'Start a new session';
  els.placeholder.classList.remove('hidden');
  els.alert.classList.add('hidden');
  els.calibChip.classList.add('hidden');
  els.fps.textContent = '';
  setChip(null);
  const ctx = els.overlay.getContext('2d');
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  if (session.samples.length < 4) {
    setStatus('Session was too short to produce a report.');
    return;
  }
  const report = buildReport(session);
  saveHistory(report);
  renderHistory();
  showReport(report);
}

function stopStreams() {
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  els.video.srcObject = null;
  disableMic();
}

// Browsers only let audio start from a user gesture, so the shared context is
// created inside the click handler, before any await.
function primeAudio() {
  try {
    audioCtx ??= new AudioContext();
    if (audioCtx.state === 'suspended') audioCtx.resume();
  } catch { /* Web Audio unavailable */ }
}

async function enableMic() {
  if (audio) return;
  try {
    // Ask for the raw signal: the browser's noise suppression and auto-gain
    // would otherwise remove most of the sound we want to measure.
    const micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    // If the mic is unplugged or revoked, turn the toggle off instead of
    // silently reporting a quiet room.
    micStream.getAudioTracks()[0].addEventListener('ended', () => {
      disableMic();
      els.mic.checked = false;
      updateAudioMeter();
    });
    primeAudio();
    await audioCtx.resume();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.5;
    const source = audioCtx.createMediaStreamSource(micStream);
    source.connect(analyser);
    let tick = null;
    try {
      if (!tickLoaded) {
        await audioCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([TICK_WORKLET], { type: 'text/javascript' })));
        tickLoaded = true;
      }
      tick = new AudioWorkletNode(audioCtx, 'tick');
      const mute = audioCtx.createGain();
      mute.gain.value = 0;
      source.connect(tick).connect(mute).connect(audioCtx.destination);
      tick.port.onmessage = () => {
        if (!S) return;
        analyzeAudio(performance.now());
        updateAudioMeter();
      };
    } catch (err) {
      console.warn('AudioWorklet unavailable; analysing audio once per video frame', err);
      tick = null;
    }
    audio = {
      analyser, source, tick, stream: micStream, score: 1,
      time: new Float32Array(analyser.fftSize), freq: new Float32Array(analyser.frequencyBinCount),
      startedAt: performance.now(), calib: [], floor: null, level: -100, above: 0,
      voiced: [], talking: false, noisy: false, state: 'calibrating',
    };
  } catch (err) {
    console.warn('Microphone unavailable', err);
    els.mic.checked = false;
  }
  updateWeightNote();
}

function disableMic() {
  if (!audio) return;
  audio.source.disconnect();
  audio.tick?.disconnect();
  audio.stream.getTracks().forEach(t => t.stop());
  audio = null;
  updateWeightNote();
}

function updateWeightNote() {
  els.weightNote.textContent = audio
    ? 'Weights: face 60%, posture 20%, audio 20%.'
    : 'Microphone off: face 75%, posture 25%.';
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------
// requestAnimationFrame stops in background tabs, so a timer keeps the
// analysis running (at a lower rate) while the student is in another tab.
function schedule() {
  let done = false;
  const run = () => { if (!done) { done = true; loop(); } };
  requestAnimationFrame(run);
  setTimeout(run, 200);
}

function loop() {
  if (!S) return;
  const v = els.video;
  if (v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    processFrame(performance.now());
  }
  schedule();
}

function processFrame(now) {
  const v = els.video;
  const w = v.videoWidth, h = v.videoHeight;
  if (!w || !h) return;
  S.frames++;
  const dt = Math.min(now - S.lastT, 250);
  S.lastT = now;

  const faceRes = tasks.face.detectForVideo(v, now);
  const handRes = tasks.hand.detectForVideo(v, now);
  // Pose changes slowly, so run it on every other frame to save compute.
  if (S.frames % 2 === 0 || !S.poseRes) S.poseRes = tasks.pose.detectForVideo(v, now);
  if (els.phone.checked && tasks.phone && S.frames % 8 === 0) {
    const det = tasks.phone.detectForVideo(v, now);
    if (det.detections.length) {
      S.phoneSeenAt = now;
      S.phoneBoxes = det.detections.map(d => d.boundingBox);
    }
  }
  const phone = els.phone.checked && now - S.phoneSeenAt < 1200;
  if (!phone) S.phoneBoxes = [];

  const c = S.calib;
  if (!c.done && now - S.t0 >= CFG.CALIB_MS && c.n >= CFG.CALIB_MIN_FRAMES) c.done = true;
  const face = analyzeFace(faceRes, handRes, w, h, now);
  const posture = analyzePosture(S.poseRes, handRes, w, h);
  const audioScore = !audio ? 1 : audio.tick ? audio.score : analyzeAudio(now);
  const calibrating = !c.done;

  const W = audio ? CFG.WEIGHTS : CFG.WEIGHTS_NO_MIC;
  let total = W.face * face.score + W.posture * posture.score + W.audio * audioScore;
  if (phone) total *= CFG.PHONE_PENALTY;
  total = clamp01(total);

  const flags = {
    noFace: !face.present,
    lookAway: face.lookAway, eyesClosed: face.eyesClosed, yawn: face.yawn,
    headTilt: face.headTilt, handFace: face.handNear, multiFace: face.count > 1,
    slouch: posture.slouch, fidget: posture.fidget, phone,
    talking: !!audio && audio.talking,
    noise: !!audio && audio.noisy,
  };
  if (!calibrating) trackEvents(flags, now);
  S.current = flags;

  const state = stateOf(total);
  S.stateMs[state] += dt;
  if (state === 'focused') {
    S.streakStart ??= now;
    S.bestStreak = Math.max(S.bestStreak, now - S.streakStart);
  } else {
    S.streakStart = null;
  }

  const t = now - S.t0;
  if (t - S.lastSample >= CFG.SAMPLE_MS) {
    S.lastSample = t;
    S.samples.push({ t: t / 1000, total, face: face.score, posture: posture.score, audio: audioScore });
  }
  S.last = { total, face: face.score, posture: posture.score, audio: audioScore, state, faceInfo: face, postureInfo: posture, calibrating };

  checkAlert(total, now, flags);
  draw(faceRes, S.poseRes, handRes, w, h);

  S.fpsFrames++;
  if (now - S.fpsT > 1000) {
    els.fps.textContent = `${Math.round(S.fpsFrames * 1000 / (now - S.fpsT))} fps`;
    S.fpsFrames = 0;
    S.fpsT = now;
  }
  if (S.frames % 4 === 0) updatePanel(now);
}

// ---------------------------------------------------------------------------
// Face: eyes, blinks, PERCLOS, yawns, head pose, hand-to-face
// ---------------------------------------------------------------------------
function analyzeFace(res, handRes, w, h, now) {
  const count = res.faceLandmarks.length;
  if (!count) {
    pushHist(S.faceHist, 0);
    S.eye.perclos.push([now, false]);
    return { present: false, count, score: 0 };
  }
  const lm = res.faceLandmarks[0];
  const P = i => [lm[i].x * w, lm[i].y * h];
  const ear = (eyeRatio(L_EYE.map(P)) + eyeRatio(R_EYE.map(P))) / 2;
  const mar = (dist(P(13), P(14)) + dist(P(81), P(178)) + dist(P(311), P(402))) / (3 * dist(P(78), P(308)));
  const bs = {};
  for (const c of res.faceBlendshapes[0]?.categories ?? []) bs[c.categoryName] = c.score;
  const blink = ((bs.eyeBlinkLeft ?? 0) + (bs.eyeBlinkRight ?? 0)) / 2;
  const { yaw, pitch, roll } = matrixToEuler(res.facialTransformationMatrixes[0].data);

  // Calibrate a neutral head position and open-eye ratio over the first frames.
  const c = S.calib;
  if (!c.done) {
    c.yaw += yaw;
    c.pitch += pitch;
    c.ear += ear;
    c.n++;
  }
  const relYaw = c.done ? yaw - c.yaw / c.n : 0;
  const relPitch = c.done ? pitch - c.pitch / c.n : 0;
  const earThresh = c.done ? Math.min(CFG.EAR_THRESH, (c.ear / c.n) * 0.65) : CFG.EAR_THRESH;

  const eyesClosed = blink > 0.5 || ear < earThresh;
  const yawn = (bs.jawOpen ?? 0) > 0.45 && mar > 0.45;
  const headTilt = Math.abs(roll) > CFG.ROLL_THRESH;
  const lookAway = Math.abs(relYaw) > CFG.YAW_THRESH || Math.abs(relPitch) > CFG.PITCH_THRESH;
  S.lookSince = lookAway ? (S.lookSince ?? now) : null;
  const lookAwaySustained = lookAway && now - S.lookSince >= CFG.LOOK_AWAY_GRACE_MS;

  trackBlink(eyesClosed, now);

  // Hand-to-face distance, as a fraction of frame width.
  let cx = 0, cy = 0;
  for (const p of lm) { cx += p.x; cy += p.y; }
  cx /= lm.length; cy /= lm.length;
  let handDist = Infinity;
  for (const hand of handRes.landmarks) {
    for (const i of [0, 4, 8, 12]) handDist = Math.min(handDist, Math.hypot((hand[i].x - cx) * w, (hand[i].y - cy) * h) / w);
  }
  const handNear = handDist < CFG.HAND_DIST_THRESH;

  // Same multiplicative penalties as FaceAnalyzer._calculate_attention_score,
  // plus gaze and sustained eye closure.
  let score = 1;
  if (eyesClosed) score *= 0.7;
  if (now - S.eye.since > 1500 && S.eye.closed) score *= 0.5;
  if (yawn) score *= 0.55;
  if (now < S.fatigueUntil) score *= 0.85;   // lingering fatigue after a yawn
  if (headTilt) score *= 0.9;
  if (lookAwaySustained) score *= 0.75;
  if (handNear) score *= Math.max(0.5, handDist / CFG.HAND_DIST_THRESH);
  if (S.faceHist.length) score = 0.7 * score + 0.3 * mean(S.faceHist);
  score = clamp01(score);
  pushHist(S.faceHist, score);

  return { present: true, count, score, ear, mar, yaw: relYaw, pitch: relPitch, roll, eyesClosed, yawn, headTilt, lookAway, handNear, handDist };
}

function trackBlink(closed, now) {
  const e = S.eye;
  if (closed && !e.closed) e.since = now;
  if (!closed && e.closed) {
    const d = now - e.since;
    if (d > 60 && d < 500) { e.blinks.push(now); S.blinkTotal++; }
  }
  e.closed = closed;
  e.perclos.push([now, closed]);
  while (e.perclos.length && now - e.perclos[0][0] > 60000) e.perclos.shift();
  while (e.blinks.length && now - e.blinks[0] > 60000) e.blinks.shift();
}

// ---------------------------------------------------------------------------
// Posture: neck angle, shoulder tilt, slouch, fidgeting
// ---------------------------------------------------------------------------
function analyzePosture(res, handRes, w, h) {
  let score = 1;
  let slouch = false, fidget = false, neck = null, shoulder = null;

  // Fidgeting: rapid index-fingertip movement, as in PostureDetector.detect_hand_movement.
  const hand = handRes.landmarks[0];
  if (hand) {
    const cur = [hand[8].x, hand[8].y * h / w];
    if (S.handPrev) {
      S.handMoves.push(dist(cur, S.handPrev));
      if (S.handMoves.length > CFG.FIDGET_HISTORY) S.handMoves.shift();
    }
    S.handPrev = cur;
    fidget = S.handMoves.filter(m => m > CFG.FIDGET_MOVE).length > CFG.FIDGET_HISTORY / 2;
  } else {
    S.handPrev = null;
    S.handMoves = [];
  }
  if (fidget) score *= 0.8;

  const lm = res?.landmarks?.[0];
  if (lm && lm[11].visibility > 0.5 && lm[12].visibility > 0.5) {
    const px = p => [p.x * w, p.y * h];
    const nose = px(lm[0]), ls = px(lm[11]), rs = px(lm[12]);
    const mid = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
    neck = angle([nose[0], 0], nose, mid);
    shoulder = Math.abs(Math.atan2(rs[1] - ls[1], Math.abs(rs[0] - ls[0])) * 180 / Math.PI);
    const ratio = (mid[1] - nose[1]) / Math.max(1, dist(ls, rs));

    const c = S.calib;
    if (!c.done) {
      c.neck.push(neck);
      c.slouch.push(ratio);
    } else if (c.neck.length) {
      const idealNeck = mean(c.neck);
      const dev = Math.abs(neck - idealNeck);
      if (dev > CFG.NECK_THRESH) score *= Math.max(0.5, 1 - (dev - CFG.NECK_THRESH) / (90 - CFG.NECK_THRESH));
      if (shoulder > CFG.SHOULDER_THRESH) score *= Math.max(0.7, 1 - (shoulder - CFG.SHOULDER_THRESH) / (45 - CFG.SHOULDER_THRESH));
      slouch = ratio < mean(c.slouch) * CFG.SLOUCH_RATIO;
      if (slouch) score *= 0.8;
    }
  }
  return { score: clamp01(score), slouch, fidget, neck, shoulder };
}

// ---------------------------------------------------------------------------
// Audio: loudness relative to the room's calibrated quiet level, and talking
// vs. noise from the spectrum (extends AudioAnalyzer.calculate_noise_level).
// ---------------------------------------------------------------------------
function analyzeAudio(now) {
  if (!audio) return 1;
  const a = audio;
  a.analyser.getFloatTimeDomainData(a.time);
  let sum = 0;
  for (const x of a.time) sum += x * x;
  const db = 10 * Math.log10(sum / a.time.length + 1e-12);  // dBFS
  a.level = 0.7 * a.level + 0.3 * db;

  if (now - a.startedAt < CFG.AUDIO_CALIB_MS) {
    a.calib.push(db);
    a.state = 'calibrating';
    return (a.score = 1);
  }
  if (a.floor == null) a.floor = percentile(a.calib, 0.3) ?? db;
  // Follow the room down quickly. Rise only slowly (3 dB/min) and only while
  // the room is near-quiet, so talking or noise never becomes the new normal.
  const dt = Math.min(now - (a.lastT ?? now), 250) / 1000;
  a.lastT = now;
  if (db < a.floor) a.floor = 0.9 * a.floor + 0.1 * db;
  else if (!a.talking && db - a.floor < CFG.SPEECH_MIN_DB) a.floor += 0.05 * dt;
  a.above = a.level - a.floor;

  // Speech puts most of its energy in the 300-3400 Hz voice band and has a
  // harmonic (peaky) spectrum; fans and traffic are broadband and flat.
  a.analyser.getFloatFrequencyData(a.freq);
  const hz = audioCtx.sampleRate / a.analyser.fftSize;
  let voice = 0, total = 0, logSum = 0, n = 0;
  const hi = Math.min(a.freq.length, Math.floor(8000 / hz));
  for (let i = Math.ceil(80 / hz); i < hi; i++) {
    const p = 10 ** (a.freq[i] / 10);
    total += p;
    if (i * hz >= 300 && i * hz <= 3400) { voice += p; logSum += Math.log(p + 1e-20); n++; }
  }
  const voiceRatio = total ? voice / total : 0;
  const flatness = n && voice ? Math.exp(logSum / n) / (voice / n) : 1;
  // Use the instantaneous level here: syllable peaks matter, not the smoothed average.
  const voiced = db - a.floor > CFG.SPEECH_MIN_DB && voiceRatio > 0.55 && flatness < 0.3;

  // Speech has pauses, so call it talking when enough of the last 1.5 s was voiced.
  a.voiced.push([now, voiced]);
  while (now - a.voiced[0][0] > 1500) a.voiced.shift();
  a.talking = a.voiced.filter(v => v[1]).length / a.voiced.length > 0.35;
  a.noisy = !a.talking && a.above > CFG.NOISE_MIN_DB;
  a.state = a.talking ? 'talking' : a.noisy ? 'noisy' : 'quiet';
  if (S) {
    S.audioMs.total += dt * 1000;
    if (a.talking) S.audioMs.talking += dt * 1000;
    if (a.noisy) S.audioMs.noisy += dt * 1000;
  }

  a.score = a.talking ? 0.4
    : a.noisy ? 1 - Math.min(0.5, 0.2 + (a.above - CFG.NOISE_MIN_DB) / 40)
    : 1;
  return a.score;
}

// ---------------------------------------------------------------------------
// Events and alerts
// ---------------------------------------------------------------------------
function trackEvents(flags, now) {
  for (const key of DISTRACTION_KEYS) {
    if (flags[key]) {
      S.active[key] ??= now;
      if (!S.logged[key] && now - S.active[key] >= SIGNALS[key].minMs) {
        S.logged[key] = true;
        S.events.push({ t: (S.active[key] - S.t0) / 1000, key, label: SIGNALS[key].event ?? SIGNALS[key].label });
        if (key === 'yawn') { S.yawns++; S.fatigueUntil = now + CFG.YAWN_FATIGUE_MS; }
      }
    } else {
      delete S.active[key];
      S.logged[key] = false;
    }
  }
}

function checkAlert(total, now, flags) {
  const thr = Number(els.thr.value);
  if (total >= thr || !S.calib.done) {
    S.lowSince = null;
    if (now - S.lastAlert > 3000) els.alert.classList.add('hidden');
    return;
  }
  S.lowSince ??= now;
  if (now - S.lowSince < CFG.ALERT_AFTER_MS || now - S.lastAlert < CFG.ALERT_COOLDOWN_MS) return;
  S.lastAlert = now;
  const causes = DISTRACTION_KEYS.filter(k => flags[k]).map(k => SIGNALS[k].label);
  els.alertReason.textContent = causes.length ? causes.slice(0, 2).join(' · ') : 'Low overall attention';
  els.alert.classList.remove('hidden');
  if (els.sound.checked) beep();
}

function beep() {
  try {
    primeAudio();
    const ctx = audioCtx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } catch { /* audio not available */ }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
function draw(faceRes, poseRes, handRes, w, h) {
  const cv = els.overlay;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (!els.showOverlay.checked) return;
  const du = new DrawingUtils(ctx);
  const f = S.last?.faceInfo;

  for (const lm of faceRes.faceLandmarks) {
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: 'rgba(56,189,248,0.12)', lineWidth: 0.5 });
    const eyeColor = f?.eyesClosed ? '#ef4444' : '#facc15';
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: eyeColor, lineWidth: 1.5 });
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: eyeColor, lineWidth: 1.5 });
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: f?.yawn ? '#ef4444' : '#f472b6', lineWidth: 1.5 });
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: f?.lookAway ? '#ef4444' : 'rgba(56,189,248,0.6)', lineWidth: 1.5 });
  }
  const pose = poseRes?.landmarks?.[0];
  if (pose) {
    const upper = PoseLandmarker.POSE_CONNECTIONS.filter(c => c.start >= 11 && c.start <= 16 && c.end <= 16);
    du.drawConnectors(pose, upper, { color: S.last?.postureInfo?.slouch ? '#ef4444' : '#22c55e', lineWidth: 3 });
  }
  for (const hand of handRes.landmarks) {
    du.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: f?.handNear ? '#ef4444' : '#a78bfa', lineWidth: 2 });
  }
  for (const b of S.phoneBoxes) {
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(b.originX, b.originY, b.width, b.height);
  }
}

// ---------------------------------------------------------------------------
// Side panel
// ---------------------------------------------------------------------------
function buildSignals() {
  els.signals.innerHTML = DISTRACTION_KEYS.map(k =>
    `<div class="sig" id="sig-${k}"><span class="dot"></span>${SIGNALS[k].label}</div>`).join('');
}

function updatePanel(now) {
  const L = S.last;
  if (!L) return;
  els.calibChip.classList.toggle('hidden', !L.calibrating);
  setScore(L.total, L.state);
  setBar(els.faceBar, els.faceVal, L.face);
  setBar(els.postureBar, els.postureVal, L.posture);
  setBar(els.audioBar, els.audioVal, audio ? L.audio : null);
  updateAudioMeter();
  for (const k of DISTRACTION_KEYS) {
    const el = document.getElementById(`sig-${k}`);
    el.classList.toggle('on', !!S.current[k]);
    el.classList.toggle('ok', !S.current[k]);
  }
  const secs = (now - S.t0) / 1000;
  els.timer.textContent = fmtClock(secs);
  els.statBlink.textContent = Math.round(S.eye.blinks.length * 60 / Math.min(60, Math.max(secs, 10)));
  const pc = S.eye.perclos;
  els.statPerclos.textContent = pc.length ? `${Math.round(100 * pc.filter(p => p[1]).length / pc.length)}%` : '–';
  els.statYawns.textContent = S.yawns;
  els.statDistractions.textContent = S.events.length;
  const totalMs = S.stateMs.focused + S.stateMs.partial + S.stateMs.distracted;
  els.statFocus.textContent = totalMs ? `${Math.round(100 * S.stateMs.focused / totalMs)}%` : '–';
  els.statStreak.textContent = fmtDuration(S.bestStreak / 1000);
  drawChart(els.trend, S.samples, { window: 120, thr: Number(els.thr.value), now: secs });
}

function setScore(total, state) {
  els.total.textContent = total == null ? '–' : total.toFixed(2);
  els.stateLabel.textContent = total == null ? 'Not running' : STATE_LABEL[state];
  els.gauge.style.strokeDashoffset = String(314.16 * (1 - (total ?? 0)));
  els.gauge.style.stroke = total == null ? 'var(--accent)' : STATE_COLOR[state];
  setChip(state);
}

function setChip(state) {
  els.stateChip.className = `chip ${state ?? ''}`;
  els.stateChip.textContent = state ? STATE_LABEL[state] : 'Idle';
}

function setBar(bar, val, v) {
  bar.style.width = `${Math.round((v ?? 0) * 100)}%`;
  bar.style.background = v == null ? 'var(--surface-2)' : STATE_COLOR[stateOf(v)];
  val.textContent = v == null ? 'off' : v.toFixed(2);
}

const AUDIO_STATE_LABEL = { calibrating: 'Measuring room…', quiet: 'Quiet', talking: 'Talking', noisy: 'Noisy room' };
const dbToPct = db => clamp01((db + 90) / 80) * 100;  // -90 to -10 dBFS

function updateAudioMeter() {
  els.audioMeterRow.classList.toggle('off', !audio);
  if (!audio) {
    els.audioMeter.style.width = '0%';
    els.audioFloor.style.display = 'none';
    els.audioState.textContent = 'Mic off: turn on Microphone to detect talking and noise';
    return;
  }
  els.audioMeter.style.width = `${dbToPct(audio.level)}%`;
  els.audioMeter.className = audio.state;
  els.audioFloor.style.display = audio.floor == null ? 'none' : '';
  if (audio.floor != null) els.audioFloor.style.left = `${dbToPct(audio.floor)}%`;
  els.audioState.textContent = audio.state === 'calibrating'
    ? AUDIO_STATE_LABEL.calibrating
    : `${AUDIO_STATE_LABEL[audio.state]} · ${Math.max(0, Math.round(audio.above))} dB above room`;
}

const STATE_LABEL = { focused: 'Focused', partial: 'Partially attentive', distracted: 'Distracted' };
const STATE_COLOR = { focused: '#22c55e', partial: '#f59e0b', distracted: '#ef4444' };
const stateOf = v => (v >= 0.7 ? 'focused' : v >= 0.4 ? 'partial' : 'distracted');

// ---------------------------------------------------------------------------
// Charts (plain canvas, no dependencies)
// ---------------------------------------------------------------------------
function drawChart(canvas, samples, { window: win = null, thr = 0.5, now = null, events = [] } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) { canvas.width = cw * dpr; canvas.height = ch * dpr; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const pad = { l: 34, r: 10, t: 10, b: 22 };
  const pw = cw - pad.l - pad.r, ph = ch - pad.t - pad.b;
  const tEnd = now ?? (samples.at(-1)?.t ?? 1);
  const tStart = win ? Math.max(0, tEnd - win) : 0;
  const span = Math.max(win ?? tEnd - tStart, 1);
  const X = t => pad.l + ((t - tStart) / span) * pw;
  const Y = v => pad.t + (1 - v) * ph;

  // Grid and axes
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#8b97b3';
  ctx.strokeStyle = '#24304f';
  ctx.lineWidth = 1;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(cw - pad.r, Y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(2), 2, Y(v) + 4);
  }
  const step = niceStep(span);
  for (let t = Math.ceil(tStart / step) * step; t <= tStart + span; t += step) {
    ctx.fillText(fmtClock(t), X(t) - 14, ch - 5);
  }

  // Threshold
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = 'rgba(239,68,68,0.7)';
  ctx.beginPath(); ctx.moveTo(pad.l, Y(thr)); ctx.lineTo(cw - pad.r, Y(thr)); ctx.stroke();
  ctx.setLineDash([]);

  // Event markers
  ctx.strokeStyle = 'rgba(192,132,252,0.7)';
  for (const e of events) {
    if (e.t < tStart) continue;
    ctx.beginPath(); ctx.moveTo(X(e.t), pad.t); ctx.lineTo(X(e.t), pad.t + ph); ctx.stroke();
  }

  const pts = samples.filter(s => s.t >= tStart);
  if (pts.length < 2) return;
  // Area fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
  grad.addColorStop(0, 'rgba(56,189,248,0.28)');
  grad.addColorStop(1, 'rgba(56,189,248,0)');
  ctx.beginPath();
  ctx.moveTo(X(pts[0].t), Y(0));
  for (const p of pts) ctx.lineTo(X(p.t), Y(p.total));
  ctx.lineTo(X(pts.at(-1).t), Y(0));
  ctx.fillStyle = grad;
  ctx.fill();
  // Line coloured by state
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  for (let i = 1; i < pts.length; i++) {
    ctx.strokeStyle = STATE_COLOR[stateOf(pts[i].total)];
    ctx.beginPath(); ctx.moveTo(X(pts[i - 1].t), Y(pts[i - 1].total)); ctx.lineTo(X(pts[i].t), Y(pts[i].total)); ctx.stroke();
  }
}

function niceStep(span) {
  for (const s of [10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]) if (span / s <= 6) return s;
  return 7200;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
function buildReport(session, sample = false) {
  const scores = session.samples.map(s => s.total);
  const duration = session.samples.at(-1).t;
  const totalMs = session.stateMs.focused + session.stateMs.partial + session.stateMs.distracted || 1;
  const counts = {};
  for (const e of session.events) counts[e.label] = (counts[e.label] ?? 0) + 1;
  const avg = mean(scores);
  const focusPct = session.stateMs.focused / totalMs;
  return {
    sample,
    startedAt: session.startedAt,
    duration,
    avg,
    max: Math.max(...scores),
    min: Math.min(...scores),
    focusPct,
    statePct: {
      focused: session.stateMs.focused / totalMs,
      partial: session.stateMs.partial / totalMs,
      distracted: session.stateMs.distracted / totalMs,
    },
    bestStreak: session.bestStreak / 1000,
    audioPct: session.audioMs.total ? {
      talking: session.audioMs.talking / session.audioMs.total,
      noisy: session.audioMs.noisy / session.audioMs.total,
    } : null,
    blinkRate: duration > 0 ? session.blinkTotal * 60 / duration : 0,
    yawns: session.yawns,
    events: session.events,
    counts,
    grade: grade(avg, focusPct),
    samples: session.samples,
  };
}

function grade(avg, focusPct) {
  const s = 0.5 * avg + 0.5 * focusPct;
  return s >= 0.85 ? 'A' : s >= 0.7 ? 'B' : s >= 0.55 ? 'C' : s >= 0.4 ? 'D' : 'E';
}

function showReport(r) {
  lastReport = r;
  $('reportMeta').textContent = `${new Date(r.startedAt).toLocaleString()} · ${fmtDuration(r.duration)}`;
  $('sampleNote').classList.toggle('hidden', !r.sample);
  const tiles = [
    ['Focus grade', r.grade, 'grade'],
    ['Average attention', r.avg.toFixed(2)],
    ['Time focused', `${Math.round(r.focusPct * 100)}%`],
    ['Longest focus streak', fmtDuration(r.bestStreak)],
    ['Distraction events', r.events.length],
    ['Yawns', r.yawns],
    ['Blinks / min', Math.round(r.blinkRate)],
    ['Peak / lowest', `${r.max.toFixed(2)} / ${r.min.toFixed(2)}`],
  ];
  if (r.audioPct) {
    tiles.push(['Time talking', `${Math.round(r.audioPct.talking * 100)}%`]);
    tiles.push(['Time in noise', `${Math.round(r.audioPct.noisy * 100)}%`]);
  }
  $('reportTiles').innerHTML = tiles.map(([k, v, cls]) => `<div class="tile ${cls ?? ''}"><b>${v}</b><span>${k}</span></div>`).join('');
  $('stateBar').innerHTML = ['focused', 'partial', 'distracted'].map(s => {
    const pct = Math.round(r.statePct[s] * 100);
    return `<div style="width:${r.statePct[s] * 100}%;background:${STATE_COLOR[s]}" title="${STATE_LABEL[s]} ${pct}%">${pct >= 8 ? `${STATE_LABEL[s]} ${pct}%` : ''}</div>`;
  }).join('');
  const entries = Object.entries(r.counts).sort((a, b) => b[1] - a[1]);
  $('eventCounts').innerHTML = entries.length
    ? entries.map(([k, n]) => `<li><span>${k}</span><b>${n}</b></li>`).join('')
    : '<li><span>No distractions logged</span><b>🎯</b></li>';
  $('eventLog').innerHTML = r.events.length
    ? r.events.map(e => `<li><time>${fmtClock(e.t)}</time>${e.label}</li>`).join('')
    : '<li class="muted">Nothing to report. Nice focus.</li>';
  if (!els.report.open) els.report.showModal();
  requestAnimationFrame(() => drawChart($('reportChart'), r.samples, { thr: Number(els.thr.value), events: r.events }));
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function reportFileStem(r) {
  return `attention-session-${new Date(r.startedAt).toISOString().slice(0, 16).replace(/[:T]/g, '-')}`;
}

function exportCsv() {
  const r = lastReport;
  const rows = ['time_s,total,face,posture,audio,state'];
  for (const s of r.samples) rows.push([s.t.toFixed(2), s.total.toFixed(3), s.face.toFixed(3), s.posture.toFixed(3), s.audio.toFixed(3), stateOf(s.total)].join(','));
  rows.push('', 'event_time_s,event');
  for (const e of r.events) rows.push(`${e.t.toFixed(1)},${e.label}`);
  download(`${reportFileStem(r)}.csv`, rows.join('\n'), 'text/csv');
}

function exportJson() {
  download(`${reportFileStem(lastReport)}.json`, JSON.stringify(lastReport, null, 2), 'application/json');
}

// ---------------------------------------------------------------------------
// Session history (localStorage, per browser)
// ---------------------------------------------------------------------------
const HISTORY_KEY = 'attention-monitor-live.history';

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) ?? []; } catch { return []; }
}

function saveHistory(r) {
  // Keep the full sample list only for recent sessions so storage stays small.
  const list = [r, ...loadHistory()].slice(0, 20).map((x, i) => (i < 5 ? x : { ...x, samples: [] }));
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch { /* storage unavailable */ }
}

function renderHistory() {
  const list = loadHistory();
  if (!list.length) {
    els.history.innerHTML = '<tr><td class="muted">No sessions yet. Finished sessions appear here.</td></tr>';
    return;
  }
  els.history.innerHTML = `<thead><tr><th>Date</th><th>Duration</th><th>Grade</th><th>Avg score</th><th>Focused</th><th>Distractions</th><th>Top distraction</th></tr></thead><tbody>${
    list.map((r, i) => {
      const top = Object.entries(r.counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '–';
      return `<tr class="${r.samples.length ? 'clickable' : ''}" data-i="${i}"><td>${new Date(r.startedAt).toLocaleString()}</td><td class="num">${fmtDuration(r.duration)}</td><td>${r.grade}</td><td class="num">${r.avg.toFixed(2)}</td><td class="num">${Math.round(r.focusPct * 100)}%</td><td class="num">${r.events.length}</td><td>${top}</td></tr>`;
    }).join('')}</tbody>`;
}

// ---------------------------------------------------------------------------
// Sample report (simulated 12-minute session, for visitors without a camera)
// ---------------------------------------------------------------------------
function sampleSession() {
  const s = newSession();
  s.startedAt = Date.now() - 12 * 60 * 1000;
  const episodes = [
    [95, 12, 'lookAway'], [180, 6, 'handFace'], [260, 25, 'phone'], [340, 4, 'yawn'], [400, 9, 'slouch'],
    [470, 5, 'eyesClosed'], [520, 4, 'yawn'], [585, 30, 'noFace'], [650, 8, 'lookAway'], [690, 6, 'noise'], [300, 7, 'talking'],
  ];
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  let prev = 0.85;
  for (let t = 0; t <= 720; t += 1) {
    const ep = episodes.find(([st, d]) => t >= st && t < st + d);
    const fatigue = t > 420 ? 0.08 : 0;
    let target = 0.86 - fatigue + (rnd() - 0.5) * 0.12;
    if (ep) target = { phone: 0.3, noFace: 0, lookAway: 0.38, eyesClosed: 0.35, yawn: 0.55, slouch: 0.62, handFace: 0.5, noise: 0.6, talking: 0.55 }[ep[2]];
    prev = 0.6 * prev + 0.4 * target;
    const total = clamp01(prev);
    const face = clamp01(total + (rnd() - 0.5) * 0.08);
    const posture = clamp01(ep?.[2] === 'slouch' ? 0.6 : 0.9 + (rnd() - 0.5) * 0.1);
    s.samples.push({ t, total, face, posture, audio: ep?.[2] === 'noise' ? 0.6 : ep?.[2] === 'talking' ? 0.4 : 1 });
    s.audioMs.total += 1000;
    if (ep?.[2] === 'talking') s.audioMs.talking += 1000;
    if (ep?.[2] === 'noise') s.audioMs.noisy += 1000;
    s.stateMs[stateOf(total)] += 1000;
  }
  for (const [st, , key] of episodes) {
    s.events.push({ t: st + 1, key, label: SIGNALS[key].event ?? SIGNALS[key].label });
  }
  s.yawns = 2;
  s.blinkTotal = 12 * 16;
  s.bestStreak = 0;
  let run = 0;
  for (const p of s.samples) { run = p.total >= 0.7 ? run + 1000 : 0; s.bestStreak = Math.max(s.bestStreak, run); }
  return s;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function eyeRatio(p) {
  const C = dist(p[0], p[3]);
  return C > 0 ? (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * C) : 1;
}
function matrixToEuler(m) {
  // Column-major 4x4 transformation matrix from MediaPipe.
  const r00 = m[0], r10 = m[1], r20 = m[2], r21 = m[6], r22 = m[10];
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(r21, r22) * deg,
    yaw: Math.atan2(-r20, Math.hypot(r21, r22)) * deg,
    roll: Math.atan2(r10, r00) * deg,
  };
}
function angle(a, b, c) {
  let deg = Math.abs((Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(a[1] - b[1], a[0] - b[0])) * 180 / Math.PI);
  return deg > 180 ? 360 - deg : deg;
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function percentile(arr, q) {
  if (!arr.length) return null;
  const s = [...arr].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}
const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const clamp01 = v => Math.max(0, Math.min(1, v));
function pushHist(arr, v) { arr.push(v); if (arr.length > CFG.SMOOTH_HISTORY) arr.shift(); }
function fmtClock(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}
function fmtDuration(sec) {
  sec = Math.round(sec);
  if (sec < 60) return `${sec}s`;
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${m}m` : `${m}m ${s}s`;
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
buildSignals();
renderHistory();
updateWeightNote();
setBar(els.faceBar, els.faceVal, null);
setBar(els.postureBar, els.postureVal, null);
setBar(els.audioBar, els.audioVal, null);
updateAudioMeter();
drawChart(els.trend, [], { window: 120, now: 120 });

els.start.addEventListener('click', start);
els.stop.addEventListener('click', stop);
els.sample.addEventListener('click', () => showReport(buildReport(sampleSession(), true)));
els.mic.addEventListener('change', async () => {
  if (els.mic.checked) primeAudio();
  if (!S) return;
  await (els.mic.checked ? enableMic() : disableMic());
  updateAudioMeter();
});
els.phone.addEventListener('change', () => { if (els.phone.checked && tasks?.fileset) loadPhoneDetector(); });
els.thr.addEventListener('input', () => { els.thrOut.textContent = Number(els.thr.value).toFixed(2); });
els.clearHistory.addEventListener('click', () => {
  try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
  renderHistory();
});
els.history.addEventListener('click', e => {
  const row = e.target.closest('tr[data-i]');
  const r = row && loadHistory()[Number(row.dataset.i)];
  if (r?.samples.length) showReport(r);
});
$('closeReport').addEventListener('click', () => els.report.close());
$('closeReport2').addEventListener('click', () => els.report.close());
$('dlCsv').addEventListener('click', exportCsv);
$('dlJson').addEventListener('click', exportJson);
window.addEventListener('resize', () => { if (!S) drawChart(els.trend, [], { window: 120, now: 120 }); });
document.addEventListener('visibilitychange', () => {
  // Browsers pause rAF in background tabs; keep the session clock honest.
  if (S && !document.hidden) S.lastT = performance.now();
});
