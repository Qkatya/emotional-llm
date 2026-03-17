import { io } from 'socket.io-client';

const PROXY_WS_URL = import.meta.env.VITE_PROXY_WS_URL ?? 'ws://localhost:3080';
const SOCKET_URL = import.meta.env.VITE_SOCKET_URL ?? 'http://localhost:8181';
const SEND_SAMPLE_RATE = 24000;
const RECV_SAMPLE_RATE = 24000;
const CHUNK_MS = 100;
const TARGET_CHUNK_SAMPLES = Math.floor((SEND_SAMPLE_RATE * CHUNK_MS) / 1000);
const BLENDSHAPE_SEND_INTERVAL_MS = 1000;
const TURN_COMPLETE_DELAY_MS = 1500;
/** Silence duration (ms) after last user audio chunk before sending audioStreamEnd (commit). */
const END_OF_SPEECH_SILENCE_MS = 800;
/** Only send blendshape updates to the server when smile values change by at least this much (vs last sent). */
const DRASTIC_THRESHOLD = 0.25;

const VIDEO_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const WEBCAM_TIMEOUT_MS = 15000;

const ARKIT_52_ORDER = [
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
  'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight',
  'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'jawForward', 'jawLeft',
  'jawRight', 'jawOpen', 'noseSneerLeft', 'noseSneerRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
];
const CHOSEN_BLENDSHAPES = ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen', 'eyeBlinkRight', 'mouthRight'];
const HIGHLIGHT_BLENDSHAPES = new Set(CHOSEN_BLENDSHAPES);

const FAKE_SMILE_BLENDSHAPES = Object.fromEntries(ARKIT_52_ORDER.map((name) => [name, 0]));
FAKE_SMILE_BLENDSHAPES.mouthSmileLeft = 0.9;
FAKE_SMILE_BLENDSHAPES.mouthSmileRight = 0.9;
FAKE_SMILE_BLENDSHAPES.jawOpen = 0.2;

const FAKE_NEUTRAL_BLENDSHAPES = Object.fromEntries(ARKIT_52_ORDER.map((name) => [name, 0]));

const connectBtn = document.getElementById('connect-btn');
const disconnectBtn = document.getElementById('disconnect-btn');
const connectionStatus = document.getElementById('connection-status');
const inputTranscriptEl = document.getElementById('input-transcript');
const outputTranscriptEl = document.getElementById('output-transcript');
const inputAudioStatus = document.getElementById('input-audio-status');
const outputAudioStatus = document.getElementById('output-audio-status');
const outputErrorEl = document.getElementById('output-error');
const video = document.getElementById('video');
const videoPlaceholder = document.getElementById('video-placeholder');
const blendshapesCameraEl = document.getElementById('blendshapes-camera');
const blendshapesWebSocketEl = document.getElementById('blendshapes-websocket');
const chosenBarsEl = document.getElementById('chosen-bars');
const sourceCameraRadio = document.querySelector('input[name="blendshape-source"][value="camera"]');
const sourceWebSocketRadio = document.querySelector('input[name="blendshape-source"][value="websocket"]');
const simulateSmilingCheckbox = document.getElementById('simulate-smiling');
const simulateNotSmilingCheckbox = document.getElementById('simulate-not-smiling');
const micLevelFill = document.getElementById('mic-level-fill');
const micLevelStatus = document.getElementById('mic-level-status');
const debugSmileValueEl = document.getElementById('debug-smile-value');
const convLogEl = document.getElementById('conversation-log');
const traceLogEl = document.getElementById('trace-log');

let socket = null;
let connectionStartTs = 0;
let alwaysOnStream = null;
let alwaysOnContext = null;
let alwaysOnSource = null;
let micLevel = 0;
let audioBuffer = [];
let faceLandmarker = null;
let lastVideoTime = -1;
let latestBlendshapes = null;
let latestWebSocketBlendshapes = null;
let blendshapeSource = 'camera';
let qSocket = null;
let blendshapeSendTimer = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let playbackContext = null;
let nextPlayTime = 0;
let activeSources = [];
let setupReceived = false;
let isPlayingAudio = false;
let modelTurnComplete = false;
let lastPlaybackEndedAt = 0;
let turnCompleteDelayTimer = null;
/** Peak smile values during the current reaction window (from playback end until we send REACTION WINDOW). */
let reactionWindowPeakSmileL = 0;
let reactionWindowPeakSmileR = 0;
/** Last mouth smile values we sent to the server; used to only send on drastic change. */
let lastSentMouthSmileLeft = null;
let lastSentMouthSmileRight = null;
let lastOutputTranscriptText = '';
/** Kept after turnComplete so we can still detect echo when input transcript arrives after the turn ended */
let lastOutputForEchoDetection = '';
let currentTurnHasOutput = false;
let currentTurnOutputText = '';
let currentReplyEntry = null;
let pendingReplyTimers = [];
let lastInputTranscriptText = '';
let currentInputEntry = null;
/** Timer: send audioStreamEnd (commit) after user stops speaking (silence). */
let endOfSpeechTimer = null;

function setStatus(text, connected = false) {
  connectionStatus.textContent = text;
  connectionStatus.classList.toggle('connected', connected);
}

function showError(msg) {
  outputErrorEl.textContent = msg;
  outputErrorEl.classList.remove('hidden');
}

function clearError() {
  outputErrorEl.classList.add('hidden');
  outputErrorEl.textContent = '';
}

function formatTime(date = new Date()) {
  return date.toTimeString().slice(0, 8);
}

function formatTs(ts) {
  if (ts == null || ts === undefined) return '—';
  const d = new Date(ts);
  return d.toTimeString().slice(0, 12);
}

/** Append one row to the message trace log. Order is chronological (by when client adds the entry). */
function appendTraceEntry({ direction, type, summary, clientTs, serverTs }) {
  if (!traceLogEl) return;
  const entry = document.createElement('div');
  entry.className = `trace-entry trace-entry--${direction.replace(/\s+/g, '-').toLowerCase().replace(/[()]/g, '')}`;
  const clientStr = formatTs(clientTs);
  const serverStr = formatTs(serverTs);

  const isOutputTranscription = type === 'outputTranscription';
  const isInputTranscription = type === 'inputTranscription';
  const isModelTurn = type === 'modelTurn';
  const isTurnComplete = type === 'turnComplete';
  entry.appendChild(document.createTextNode(`[${direction}] ${type} `));
  if (isOutputTranscription && summary) {
    const span = document.createElement('span');
    span.className = 'trace-transcription';
    span.textContent = summary;
    entry.appendChild(span);
    entry.appendChild(document.createTextNode(' '));
  } else if (isInputTranscription && summary) {
    const span = document.createElement('span');
    span.className = 'trace-input-transcription';
    span.textContent = summary;
    entry.appendChild(span);
    entry.appendChild(document.createTextNode(' '));
  } else if (isModelTurn && summary) {
    const span = document.createElement('span');
    span.className = 'trace-model-turn';
    span.textContent = summary;
    entry.appendChild(span);
    entry.appendChild(document.createTextNode(' '));
  } else if (isTurnComplete && summary) {
    const span = document.createElement('span');
    span.className = 'trace-turn-complete';
    span.textContent = summary;
    entry.appendChild(span);
    entry.appendChild(document.createTextNode(' '));
  } else if (summary) {
    entry.appendChild(document.createTextNode(summary + ' '));
  }
  if (clientTs != null) entry.appendChild(document.createTextNode(`client=${clientStr} `));
  if (serverTs != null) entry.appendChild(document.createTextNode(`server=${serverStr}`));

  traceLogEl.appendChild(entry);
  traceLogEl.scrollTop = traceLogEl.scrollHeight;
}


function logSentText(text, turnComplete = null) {
  if (!convLogEl) return;
  const isReaction = text.startsWith('REACTION WINDOW');
  const smileLMatch = text.match(/mouthSmileLeft:\s*([\d.]+)/);
  const smileRMatch = text.match(/mouthSmileRight:\s*([\d.]+)/);
  const jawMatch = text.match(/jawOpen:\s*([\d.]+)/);
  const smileL = smileLMatch ? smileLMatch[1] : '?';
  const smileR = smileRMatch ? smileRMatch[1] : '?';
  const jaw = jawMatch ? jawMatch[1] : '?';

  const label = isReaction ? 'REACTION WINDOW' : 'blendshapes';
  const tcLabel = turnComplete != null ? ` [turnComplete=${turnComplete}]` : '';
  const summary = `${label}  smL=${smileL} smR=${smileR} jaw=${jaw}${tcLabel}`;

  const entry = document.createElement('div');
  entry.className = 'log-entry log-entry--user' + (isReaction ? ' log-entry--reaction' : '');

  const ts = document.createElement('span');
  ts.className = 'log-ts';
  ts.textContent = formatTime();

  const src = document.createElement('span');
  src.className = 'log-src log-src--user';
  src.textContent = 'USER';

  const sumSpan = document.createElement('span');
  sumSpan.className = 'log-summary';
  sumSpan.textContent = summary;

  const btn = document.createElement('button');
  btn.className = 'log-expand-btn';
  btn.textContent = 'show full';

  const full = document.createElement('div');
  full.className = 'log-full-text';
  full.textContent = text;

  btn.addEventListener('click', () => {
    const open = full.classList.toggle('open');
    btn.textContent = open ? 'hide' : 'show full';
  });

  entry.append(ts, src, sumSpan, btn, full);
  convLogEl.appendChild(entry);
  convLogEl.scrollTop = convLogEl.scrollHeight;
}

function appendReplyText(text) {
  if (!convLogEl) return;
  if (!currentReplyEntry) {
    currentReplyEntry = document.createElement('div');
    currentReplyEntry.className = 'log-entry log-entry--openai';
    const ts = document.createElement('span');
    ts.className = 'log-ts';
    ts.textContent = formatTime();
    const src = document.createElement('span');
    src.className = 'log-src log-src--openai';
    src.textContent = 'OPENAI';
    const span = document.createElement('span');
    span.className = 'log-reply-text log-reply-text--active';
    currentReplyEntry.appendChild(ts);
    currentReplyEntry.appendChild(src);
    currentReplyEntry.appendChild(span);
    convLogEl.appendChild(currentReplyEntry);
  }
  const span = currentReplyEntry.querySelector('.log-reply-text');
  if (span) span.textContent += text;
  convLogEl.scrollTop = convLogEl.scrollHeight;
}

function scheduleReplyText(deltaText) {
  if (!playbackContext) {
    appendReplyText(deltaText);
    return;
  }
  const delayMs = Math.max(0, (nextPlayTime - playbackContext.currentTime) * 1000);
  const timerId = window.setTimeout(() => {
    const idx = pendingReplyTimers.indexOf(timerId);
    if (idx !== -1) pendingReplyTimers.splice(idx, 1);
    appendReplyText(deltaText);
  }, delayMs);
  pendingReplyTimers.push(timerId);
}

function finalizeReplyEntry() {
  if (currentReplyEntry) {
    const span = currentReplyEntry.querySelector('.log-reply-text');
    if (span) span.classList.remove('log-reply-text--active');
    currentReplyEntry = null;
  }
  lastOutputForEchoDetection = lastOutputTranscriptText;
  lastOutputTranscriptText = '';
}

function flushPendingReplyText() {
  for (const t of pendingReplyTimers) window.clearTimeout(t);
  pendingReplyTimers = [];
  finalizeReplyEntry();
}

function logTurnComplete(who) {
  if (!convLogEl) return;
  const entry = document.createElement('div');
  entry.className = 'log-entry log-entry--turncomplete';
  const ts = document.createElement('span');
  ts.className = 'log-ts';
  ts.textContent = formatTime();
  const label = document.createElement('span');
  label.className = 'log-summary';
  label.textContent = `[turnComplete — ${who}]`;
  entry.append(ts, label);
  convLogEl.appendChild(entry);
  convLogEl.scrollTop = convLogEl.scrollHeight;
}

function appendInputTranscript(deltaText) {
  if (!convLogEl) return;
  if (!currentInputEntry) {
    currentInputEntry = document.createElement('div');
    currentInputEntry.className = 'log-entry log-entry--user log-entry--speech';
    const ts = document.createElement('span');
    ts.className = 'log-ts';
    ts.textContent = formatTime();
    const src = document.createElement('span');
    src.className = 'log-src log-src--user';
    src.textContent = 'USER';
    const span = document.createElement('span');
    span.className = 'log-input-text log-reply-text--active';
    currentInputEntry.appendChild(ts);
    currentInputEntry.appendChild(src);
    currentInputEntry.appendChild(span);
    convLogEl.appendChild(currentInputEntry);
  }
  const span = currentInputEntry.querySelector('.log-input-text');
  if (span) span.textContent += deltaText;
  convLogEl.scrollTop = convLogEl.scrollHeight;
}

function finalizeInputEntry() {
  if (currentInputEntry) {
    const span = currentInputEntry.querySelector('.log-input-text');
    if (span) span.classList.remove('log-reply-text--active');
    currentInputEntry = null;
  }
  lastInputTranscriptText = '';
}

function floatTo16(float32Array) {
  const len = float32Array.length;
  const buf = new ArrayBuffer(len * 2);
  const view = new DataView(buf);
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function resampleToSendRate(float32Samples, fromSampleRate) {
  const ratio = fromSampleRate / SEND_SAMPLE_RATE;
  const outLen = Math.floor(float32Samples.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIndex = i * ratio;
    const idx = Math.floor(srcIndex);
    const frac = srcIndex - idx;
    if (idx + 1 < float32Samples.length) {
      out[i] = float32Samples[idx] * (1 - frac) + float32Samples[idx + 1] * frac;
    } else {
      out[i] = float32Samples[idx];
    }
  }
  return out;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function computeRMS(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function startAlwaysOnMic() {
  navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }).then((stream) => {
    alwaysOnStream = stream;
    mediaStream = stream;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    alwaysOnContext = ctx;
    audioContext = ctx;
    const src = ctx.createMediaStreamSource(stream);
    alwaysOnSource = src;
    sourceNode = src;

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.6;
    src.connect(analyser);

    const bufferSize = 4096;
    const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
    src.connect(processor);
    processor.connect(ctx.destination);

    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      micLevel = computeRMS(input);

      if (socket && socket.readyState === WebSocket.OPEN && setupReceived) {
        // Always send mic audio so the API can detect user speech and interrupt when interrupt_response: true
        audioBuffer.push(new Float32Array(input));
        const totalSamples = audioBuffer.reduce((acc, b) => acc + b.length, 0);
        const needed = (ctx.sampleRate / SEND_SAMPLE_RATE) * TARGET_CHUNK_SAMPLES;
        if (totalSamples >= needed) {
          const combined = new Float32Array(totalSamples);
          let offset = 0;
          for (const b of audioBuffer) {
            combined.set(b, offset);
            offset += b.length;
          }
          audioBuffer = [];
          const resampled = resampleToSendRate(combined, ctx.sampleRate);
          const pcm = floatTo16(resampled);
          const base64 = arrayBufferToBase64(pcm);
          socket.send(JSON.stringify({ type: 'audio', data: base64 }));
          if (endOfSpeechTimer != null) clearTimeout(endOfSpeechTimer);
          endOfSpeechTimer = window.setTimeout(() => {
            endOfSpeechTimer = null;
            if (socket && socket.readyState === WebSocket.OPEN && setupReceived && !isPlayingAudio) {
              appendTraceEntry({ direction: 'Client sent', type: 'audioStreamEnd', summary: 'commit (end of speech — user transcript will follow)', clientTs: Date.now(), serverTs: null });
              socket.send(JSON.stringify({ audioStreamEnd: true }));
            }
          }, END_OF_SPEECH_SILENCE_MS);
        }
      }
    };

    const levelLoop = () => {
      if (micLevelFill) {
        const pct = Math.min(100, Math.round(micLevel * 800));
        micLevelFill.style.width = pct + '%';
      }
      requestAnimationFrame(levelLoop);
    };
    levelLoop();
    micLevelStatus.textContent = 'Mic active';
    if (inputAudioStatus) inputAudioStatus.textContent = 'Connect to see your transcript';
    return stream;
  }).catch((err) => {
    micLevelStatus.textContent = 'Mic denied or error';
    if (inputAudioStatus) inputAudioStatus.textContent = '—';
    console.error('Mic error:', err);
  });
}

function stopAlwaysOnMic() {
  if (alwaysOnStream) {
    alwaysOnStream.getTracks().forEach((t) => t.stop());
    alwaysOnStream = null;
    mediaStream = null;
  }
  if (alwaysOnSource && alwaysOnContext) {
    try { alwaysOnSource.disconnect(); } catch (_) {}
    alwaysOnSource = null;
  }
  alwaysOnContext = null;
  sourceNode = null;
  audioContext = null;
  micLevelStatus.textContent = 'Mic off';
  if (micLevelFill) micLevelFill.style.width = '0%';
}

function startMicrophoneForRealtime() {
  if (mediaStream && inputAudioStatus) inputAudioStatus.textContent = 'Recording…';
}

function stopAllPlayback() {
  for (const src of activeSources) {
    try { src.stop(); } catch (_) {}
  }
  activeSources = [];
  nextPlayTime = 0;
  isPlayingAudio = false;
  outputAudioStatus.textContent = '—';
}

function scheduleAudioChunk(base64Data) {
  const buffer = base64ToArrayBuffer(base64Data);
  const view = new Int16Array(buffer);
  if (view.length === 0) return;
  const float32 = new Float32Array(view.length);
  for (let i = 0; i < view.length; i++) float32[i] = view[i] / (view[i] < 0 ? 0x8000 : 0x7fff);

  if (!playbackContext) playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RECV_SAMPLE_RATE });
  const ctx = playbackContext;

  const doSchedule = () => {
    isPlayingAudio = true;
    const audioBuffer = ctx.createBuffer(1, float32.length, RECV_SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(float32);
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime, nextPlayTime);
    nextPlayTime = startAt + audioBuffer.duration;
    activeSources.push(source);
    source.onended = () => {
      const idx = activeSources.indexOf(source);
      if (idx !== -1) activeSources.splice(idx, 1);
      if (activeSources.length === 0) {
        outputAudioStatus.textContent = '—';
        isPlayingAudio = false;
        lastPlaybackEndedAt = Date.now();
        tryScheduleTurnCompleteDelay();
      }
    };
    outputAudioStatus.textContent = 'Playing…';
    source.start(startAt);
  };

  if (ctx.state === 'suspended') {
    ctx.resume().then(doSchedule);
  } else {
    doSchedule();
  }
}

function connect() {
  if (socket) return;
  clearError();
  setStatus('Connecting…');
  connectBtn.disabled = true;

  const wsUrl = PROXY_WS_URL.replace(/^http/, 'ws');
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    connectionStartTs = Date.now();
    setStatus('Connected', true);
    disconnectBtn.disabled = false;
    outputAudioStatus.textContent = '—';
    startBlendshapeSendInterval();
    startMicrophoneForRealtime();
  };

  socket.onmessage = (event) => {
    try {
      let raw = event.data;
      if (raw instanceof ArrayBuffer) {
        raw = new TextDecoder().decode(raw);
      } else if (raw instanceof Blob) {
        return;
      }
      const msg = JSON.parse(raw);

      // Trace: server received from client (separate message from server)
      if (msg.trace) {
        appendTraceEntry({
          direction: 'Server recv (client)',
          type: msg.trace.type || 'unknown',
          summary: msg.trace.summary || '',
          clientTs: Date.now(),
          serverTs: msg.trace.ts != null ? msg.trace.ts : null,
        });
        return;
      }

      // For every app message: log Server sent (serverTs) and Client recv (clientTs)
      const serverTs = msg.serverTs != null ? msg.serverTs : null;
      const clientTs = Date.now();
      let traceType = 'message';
      let traceSummary = '';
      if (msg.error) {
        traceType = 'error';
        traceSummary = String(msg.error).slice(0, 80);
      } else if (msg.setupComplete !== undefined || msg.setup_complete !== undefined) {
        traceType = 'setupComplete';
        traceSummary = 'ready';
      } else {
        const sc = msg.serverContent ?? msg.server_content;
        if (sc) {
          if (sc.interrupted) {
            traceType = 'interrupted';
            traceSummary = 'interrupted';
          } else if (sc.inputTranscription ?? sc.input_transcription) {
            traceType = 'inputTranscription';
            const t = sc.inputTranscription ?? sc.input_transcription;
            const userText = (t.text != null ? String(t.text) : '').trim();
            traceSummary = userText ? `User: ${userText.slice(0, 80)}` : 'User: (empty)';
          } else if (sc.outputTranscription ?? sc.output_transcription) {
            traceType = 'outputTranscription';
            const t = sc.outputTranscription ?? sc.output_transcription;
            traceSummary = (t.text != null ? String(t.text) : '').slice(0, 50);
          } else if (sc.turnComplete || sc.turn_complete) {
            traceType = 'turnComplete';
            traceSummary = 'turnComplete';
          } else if (sc.modelTurn ?? sc.model_turn) {
            traceType = 'modelTurn';
            traceSummary = 'audio delta';
          }
        }
      }
      appendTraceEntry({ direction: 'Server sent', type: traceType, summary: traceSummary, clientTs: null, serverTs });
      appendTraceEntry({ direction: 'Client recv', type: traceType, summary: traceSummary, clientTs, serverTs });

      if (msg.error) {
        showError(msg.error);
        if (msg.error.includes('OpenAI closed') || msg.error.includes('connection error')) {
          socket.close();
        }
        return;
      }
      if (msg.setupComplete !== undefined || msg.setup_complete !== undefined) {
        setupReceived = true;
        console.log('[OpenAI] setupComplete received');
        return;
      }
      // API may send serverContent (camelCase) or server_content (snake_case)
      const sc = msg.serverContent ?? msg.server_content;
      if (sc) {
        if (sc.interrupted) {
          stopAllPlayback();
          clearTurnCompleteState();
          flushPendingReplyText();
          finalizeInputEntry();
        }
        const inputTrans = sc.inputTranscription ?? sc.input_transcription;
        const outputTrans = sc.outputTranscription ?? sc.output_transcription;
        if (inputTrans && inputTrans.text != null) {
          const inputStr = String(inputTrans.text).trim();
          const lastOut = lastOutputTranscriptText || lastOutputForEchoDetection;
          const isLikelyEcho = inputStr.length > 0 && lastOut.length > 0 && (
            lastOut.includes(inputStr) ||
            lastOut.toLowerCase().includes(inputStr.toLowerCase())
          );
          const msSincePlayback = Date.now() - lastPlaybackEndedAt;
          const isDuringPlayback = isPlayingAudio || activeSources.length > 0 || msSincePlayback < 2000;
          const shouldSuppress = isLikelyEcho || isDuringPlayback;
          if (!shouldSuppress) {
            lastOutputForEchoDetection = '';
            lastInputTranscriptText += inputTrans.text;
            inputTranscriptEl.textContent = lastInputTranscriptText;
            appendInputTranscript(inputTrans.text);
          }
        }
        if (outputTrans && outputTrans.text != null) {
          currentTurnHasOutput = true;
          currentTurnOutputText += outputTrans.text;
          lastOutputTranscriptText += outputTrans.text;
          outputTranscriptEl.textContent = lastOutputTranscriptText;
          scheduleReplyText(outputTrans.text);
        }
        const turn = sc.modelTurn ?? sc.model_turn;
        const parts = turn?.parts ?? turn?.Parts ?? [];
        const hasAudio = parts.some((p) => (p.inlineData ?? p.inline_data)?.data);
        if (hasAudio) {
          clearTurnCompleteState();
        }
        for (const part of parts) {
          const inline = part.inlineData ?? part.inline_data;
          const b64 = inline?.data;
          if (b64) {
            scheduleAudioChunk(b64);
          }
        }
        if (sc.turnComplete || sc.turn_complete) {
          nextPlayTime = 0;
          const hadOutput = currentTurnHasOutput;
          const outputText = currentTurnOutputText.trim();
          const endsWithQuestion = outputText.endsWith('?');
          currentTurnHasOutput = false;
          currentTurnOutputText = '';
          modelTurnComplete = hadOutput && !endsWithQuestion;
          finalizeInputEntry();
          finalizeReplyEntry();
          logTurnComplete('OpenAI');
          // #region agent log
          fetch('http://127.0.0.1:7894/ingest/ee88f6f2-1fd1-4d28-8a7c-5834b810bd15',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'54c55a'},body:JSON.stringify({sessionId:'54c55a',location:'main.js:turnComplete',message:'turnComplete',data:{activeSources:activeSources.length,hadOutput,endsWithQuestion,outputSnippet:outputText.slice(-60)},timestamp:Date.now(),hypothesisId:'F2'})}).catch(()=>{});
          // #endregion
          if (hadOutput && !endsWithQuestion && turnCompleteDelayTimer == null) tryScheduleTurnCompleteDelay();
        }
      }
      const inputTransTop = msg.inputTranscription ?? msg.input_transcription;
      const outputTransTop = msg.outputTranscription ?? msg.output_transcription;
      if (inputTransTop && inputTransTop.text != null) {
        const inputStr = String(inputTransTop.text).trim();
        const lastOut = lastOutputTranscriptText || lastOutputForEchoDetection;
        const isLikelyEcho = inputStr.length > 0 && lastOut.length > 0 && (
          lastOut.includes(inputStr) ||
          lastOut.toLowerCase().includes(inputStr.toLowerCase())
        );
        const msSincePlayback = Date.now() - lastPlaybackEndedAt;
        const isDuringPlayback = isPlayingAudio || activeSources.length > 0 || msSincePlayback < 2000;
        const shouldSuppress = isLikelyEcho || isDuringPlayback;
        if (!shouldSuppress) {
          lastOutputForEchoDetection = '';
          lastInputTranscriptText += inputTransTop.text;
          inputTranscriptEl.textContent = lastInputTranscriptText;
          appendInputTranscript(inputTransTop.text);
        }
      }
      if (outputTransTop && outputTransTop.text != null) {
        currentTurnHasOutput = true;
        currentTurnOutputText += outputTransTop.text;
        lastOutputTranscriptText += outputTransTop.text;
        outputTranscriptEl.textContent = lastOutputTranscriptText;
        scheduleReplyText(outputTransTop.text);
      }
    } catch (_) {}
  };

  socket.onclose = (event) => {
    socket = null;
    audioBuffer = [];
    if (endOfSpeechTimer != null) { clearTimeout(endOfSpeechTimer); endOfSpeechTimer = null; }
    setupReceived = false;
    stopAllPlayback();
    clearTurnCompleteState();
    flushPendingReplyText();
    finalizeInputEntry();
    if (playbackContext) {
      playbackContext.close().catch(() => {});
      playbackContext = null;
    }
    stopBlendshapeSendInterval();
    setStatus('Disconnected');
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    inputAudioStatus.textContent = 'Connect to see your transcript';
    outputAudioStatus.textContent = '—';
    const reason = event.reason || (event.code !== 1000 ? `code ${event.code}` : '');
    if (reason) showError('Connection closed: ' + reason);
  };

  socket.onerror = () => {
    showError('WebSocket error');
  };
}

function disconnect() {
  if (socket) {
    if (endOfSpeechTimer != null) { clearTimeout(endOfSpeechTimer); endOfSpeechTimer = null; }
    appendTraceEntry({ direction: 'Client sent', type: 'audioStreamEnd', summary: 'commit (disconnect)', clientTs: Date.now(), serverTs: null });
    socket.send(JSON.stringify({ audioStreamEnd: true }));
    socket.close();
    socket = null;
  }
  audioBuffer = [];
  setupReceived = false;
  stopAllPlayback();
  clearTurnCompleteState();
  flushPendingReplyText();
  finalizeInputEntry();
  if (playbackContext) {
    playbackContext.close().catch(() => {});
    playbackContext = null;
  }
  stopBlendshapeSendInterval();
  setStatus('Disconnected');
  connectBtn.disabled = false;
  disconnectBtn.disabled = true;
  inputAudioStatus.textContent = '—';
  outputAudioStatus.textContent = '—';
}

function blendshapesToObject(faceBlendshapes) {
  if (!faceBlendshapes || !faceBlendshapes.length) return null;
  const first = faceBlendshapes[0];
  if (!first) return null;
  const obj = {};
  const categories = first.categories || first;
  if (Array.isArray(categories)) {
    for (const c of categories) {
      const name = c.categoryName ?? c.name;
      const score = c.score ?? c.value;
      if (name != null && typeof score === 'number') obj[name] = score;
    }
  } else if (categories && typeof categories === 'object') {
    for (const [name, value] of Object.entries(categories)) {
      if (typeof value === 'number') obj[name] = value;
    }
  }
  return Object.keys(obj).length ? obj : null;
}

function blendshapeArrayToObject(arr) {
  if (!arr || !Array.isArray(arr)) return null;
  if (typeof arr[0] === 'number') {
    const obj = {};
    const len = Math.min(arr.length, ARKIT_52_ORDER.length);
    for (let i = 0; i < len; i++) obj[ARKIT_52_ORDER[i]] = Number(arr[i]) || 0;
    return obj;
  }
  const obj = {};
  for (const item of arr) {
    if (item && typeof item === 'object') {
      const name = item.categoryName ?? item.name;
      const score = item.score ?? item.value;
      if (name != null && typeof score === 'number') obj[name] = score;
    }
  }
  return Object.keys(obj).length ? obj : null;
}

function normalizeBlendshapes(input) {
  if (!input) return null;
  if (Array.isArray(input)) return blendshapeArrayToObject(input);
  if (typeof input === 'object') {
    const obj = {};
    for (const [k, v] of Object.entries(input)) {
      if (typeof v === 'number') obj[k] = v;
    }
    return Object.keys(obj).length ? obj : null;
  }
  return null;
}

function formatBlendshapesForApi(obj) {
  if (!obj) return '';
  const lines = ARKIT_52_ORDER.map((name) => `${name}: ${(obj[name] ?? 0).toFixed(3)}`);
  return 'Current face blendshapes:\n' + lines.join('\n');
}

/** Format for turnComplete reaction window: label clearly so the model uses only this message for joke reaction. Optional peakSmile overrides (L,R) use peak values over the window instead of snapshot. Tagged as this turn only. */
function formatBlendshapesReactionWindow(obj, peakSmile = null) {
  if (!obj) return '';
  const smileL = peakSmile ? peakSmile.left : (obj.mouthSmileLeft ?? 0);
  const smileR = peakSmile ? peakSmile.right : (obj.mouthSmileRight ?? 0);
  const lines = ARKIT_52_ORDER.map((name) => {
    const v = (name === 'mouthSmileLeft' ? smileL : name === 'mouthSmileRight' ? smileR : (obj[name] ?? 0));
    return `${name}: ${Number(v).toFixed(3)}`;
  });
  return `REACTION WINDOW — this turn only (reaction to your last message). Use ONLY these values to judge if they smiled (above 0.5 = smiling): mouthSmileLeft: ${Number(smileL).toFixed(3)} mouthSmileRight: ${Number(smileR).toFixed(3)}\nCurrent face blendshapes:\n` + lines.join('\n');
}

function getBlendshapesForApi() {
  return blendshapeSource === 'camera' ? latestBlendshapes : latestWebSocketBlendshapes;
}

function renderBlendshapesIntoList(obj, listEl) {
  if (!listEl) return;
  if (!obj) {
    listEl.innerHTML = '<span class="muted">No data yet</span>';
    return;
  }
  const entries = Object.entries(obj).sort((a, b) => a[0].localeCompare(b[0]));
  listEl.innerHTML = entries
    .map(([name, value]) => {
      const highlight = HIGHLIGHT_BLENDSHAPES.has(name) ? ' row--highlight' : '';
      return `<div class="row${highlight}"><span class="name">${name}</span><span>${value.toFixed(3)}</span></div>`;
    })
    .join('');
}

function renderChosenBars() {
  if (!chosenBarsEl) return;
  const cameraObj = latestBlendshapes;
  const wsObj = latestWebSocketBlendshapes;
  const hasAny = cameraObj || wsObj;
  if (!hasAny) {
    chosenBarsEl.innerHTML = '<span class="muted">No data from camera or Q yet</span>';
    return;
  }
  chosenBarsEl.innerHTML = CHOSEN_BLENDSHAPES.map((name) => {
    const cameraVal = cameraObj?.[name] ?? 0;
    const wsVal = wsObj?.[name] ?? 0;
    const cameraPct = Math.round(cameraVal * 100);
    const wsPct = Math.round(wsVal * 100);
    return `
      <div class="chosen-bar-row">
        <span class="chosen-bar-label">${name}</span>
        <div class="chosen-bar-group">
          <div class="chosen-bar-item">
            <span class="chosen-bar-sourcelabel">Camera</span>
            <div class="chosen-bar-track chosen-bar-track--camera">
              <div class="chosen-bar-fill" style="width: ${cameraPct}%"></div>
            </div>
            <span class="chosen-bar-value">${cameraVal.toFixed(2)}</span>
          </div>
          <div class="chosen-bar-item">
            <span class="chosen-bar-sourcelabel">Q</span>
            <div class="chosen-bar-track chosen-bar-track--websocket">
              <div class="chosen-bar-fill" style="width: ${wsPct}%"></div>
            </div>
            <span class="chosen-bar-value">${wsVal.toFixed(2)}</span>
          </div>
        </div>
      </div>`;
  }).join('');
}

function startBlendshapeSendInterval() {
  if (blendshapeSendTimer) return;
  blendshapeSendTimer = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN && setupReceived) {
      const obj = simulateSmilingCheckbox?.checked
        ? FAKE_SMILE_BLENDSHAPES
        : simulateNotSmilingCheckbox?.checked
          ? FAKE_NEUTRAL_BLENDSHAPES
          : getBlendshapesForApi();
      if (obj) {
        const smileL = obj.mouthSmileLeft ?? 0;
        const smileR = obj.mouthSmileRight ?? 0;
        const isDrastic =
          lastSentMouthSmileLeft == null ||
          lastSentMouthSmileRight == null ||
          Math.abs(smileL - lastSentMouthSmileLeft) >= DRASTIC_THRESHOLD ||
          Math.abs(smileR - lastSentMouthSmileRight) >= DRASTIC_THRESHOLD;
        if (isDrastic) {
          lastSentMouthSmileLeft = smileL;
          lastSentMouthSmileRight = smileR;
          if (turnCompleteDelayTimer != null) {
            reactionWindowPeakSmileL = Math.max(reactionWindowPeakSmileL, smileL);
            reactionWindowPeakSmileR = Math.max(reactionWindowPeakSmileR, smileR);
            // const text = formatBlendshapesForApi(obj);
            // socket.send(JSON.stringify({ type: 'text', data: text, turnComplete: false }));
            // logSentText(text, false);
          } else {
            // const text = formatBlendshapesForApi(obj);
            // socket.send(JSON.stringify({ type: 'text', data: text, turnComplete: true }));
            // logSentText(text, true);
          }
          if (debugSmileValueEl) {
            debugSmileValueEl.textContent = smileR.toFixed(3);
          }
        }
      }
    }
  }, BLENDSHAPE_SEND_INTERVAL_MS);
}

function stopBlendshapeSendInterval() {
  if (blendshapeSendTimer) {
    clearInterval(blendshapeSendTimer);
    blendshapeSendTimer = null;
  }
  lastSentMouthSmileLeft = null;
  lastSentMouthSmileRight = null;
}

function clearTurnCompleteState() {
  if (turnCompleteDelayTimer != null) {
    clearTimeout(turnCompleteDelayTimer);
    turnCompleteDelayTimer = null;
  }
  if (endOfSpeechTimer != null) {
    clearTimeout(endOfSpeechTimer);
    endOfSpeechTimer = null;
  }
  modelTurnComplete = false;
  reactionWindowPeakSmileL = 0;
  reactionWindowPeakSmileR = 0;
}

function tryScheduleTurnCompleteDelay() {
  if (!modelTurnComplete || activeSources.length !== 0) return;
  if (!socket || socket.readyState !== WebSocket.OPEN || !setupReceived) return;
  if (turnCompleteDelayTimer != null) clearTimeout(turnCompleteDelayTimer);
  const obj0 = getBlendshapesForApi();
  reactionWindowPeakSmileL = obj0 ? (obj0.mouthSmileLeft ?? 0) : 0;
  reactionWindowPeakSmileR = obj0 ? (obj0.mouthSmileRight ?? 0) : 0;
  turnCompleteDelayTimer = window.setTimeout(() => {
    turnCompleteDelayTimer = null;
    modelTurnComplete = false;
    const obj = simulateSmilingCheckbox?.checked
      ? FAKE_SMILE_BLENDSHAPES
      : simulateNotSmilingCheckbox?.checked
        ? FAKE_NEUTRAL_BLENDSHAPES
        : getBlendshapesForApi();
    if (obj && socket && socket.readyState === WebSocket.OPEN) {
      const peakL = reactionWindowPeakSmileL;
      const peakR = reactionWindowPeakSmileR;
      reactionWindowPeakSmileL = 0;
      reactionWindowPeakSmileR = 0;
      const text = formatBlendshapesReactionWindow(obj, { left: peakL, right: peakR });
      lastSentMouthSmileLeft = peakL;
      lastSentMouthSmileRight = peakR;
      appendTraceEntry({ direction: 'Client sent', type: 'text', summary: 'REACTION WINDOW turnComplete=true', clientTs: Date.now(), serverTs: null });
      socket.send(JSON.stringify({ type: 'text', data: text, turnComplete: true }));
      logSentText(text, true);
    }
  }, TURN_COMPLETE_DELAY_MS);
}

async function loadMediaPipeVision() {
  const mod = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14');
  const vision = mod.default ?? mod;
  return {
    FaceLandmarker: vision.FaceLandmarker ?? mod.FaceLandmarker,
    FilesetResolver: vision.FilesetResolver ?? mod.FilesetResolver,
  };
}

async function initFaceLandmarker() {
  const { FaceLandmarker: FL, FilesetResolver: FS } = await loadMediaPipeVision();
  const vision = await FS.forVisionTasks(WASM_URL);
  faceLandmarker = await FL.createFromOptions(vision, {
    baseOptions: { modelAssetPath: VIDEO_MODEL_URL },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1,
  });
}

function getUserMediaWithTimeout(constraints, timeoutMs) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('Camera request timed out')), timeoutMs);
    navigator.mediaDevices.getUserMedia(constraints).then((s) => { clearTimeout(t); resolve(s); }, (e) => { clearTimeout(t); reject(e); });
  });
}

async function initWebcam() {
  const stream = await getUserMediaWithTimeout({ video: { width: 640, height: 480 } }, WEBCAM_TIMEOUT_MS);
  video.srcObject = stream;
  await video.play();
  videoPlaceholder.classList.add('hidden');
}

function detectFrame() {
  if (!faceLandmarker || video.readyState < 2) {
    requestAnimationFrame(detectFrame);
    return;
  }
  if (video.currentTime !== lastVideoTime) {
    const result = faceLandmarker.detectForVideo(video, performance.now());
    if (result.faceBlendshapes && result.faceBlendshapes.length) {
      latestBlendshapes = blendshapesToObject(result.faceBlendshapes);
      renderBlendshapesIntoList(latestBlendshapes, blendshapesCameraEl);
      renderChosenBars();
    }
    lastVideoTime = video.currentTime;
  }
  requestAnimationFrame(detectFrame);
}

function initQSocket() {
  qSocket = io(SOCKET_URL, { transports: ['websocket'], reconnection: true });
  qSocket.on('blendshapes_data', (data) => {
    let obj = null;
    if (data?.blendshapes) obj = normalizeBlendshapes(data.blendshapes);
    else if (data && typeof data === 'object') obj = normalizeBlendshapes(data);
    if (obj) {
      latestWebSocketBlendshapes = obj;
      renderBlendshapesIntoList(latestWebSocketBlendshapes, blendshapesWebSocketEl);
      renderChosenBars();
    }
  });
  qSocket.on('connect', () => {
    document.getElementById('websocket-status').textContent = 'Connected';
    document.getElementById('websocket-status').className = 'websocket-status websocket-status--connected';
  });
  qSocket.on('disconnect', () => {
    document.getElementById('websocket-status').textContent = 'Disconnected';
    document.getElementById('websocket-status').className = 'websocket-status websocket-status--disconnected';
  });
  qSocket.on('connect_error', (err) => {
    document.getElementById('websocket-status').textContent = 'Error';
    document.getElementById('websocket-status').className = 'websocket-status websocket-status--error';
  });
}

function setupSourceToggle() {
  if (sourceCameraRadio) sourceCameraRadio.addEventListener('change', () => { blendshapeSource = 'camera'; renderChosenBars(); });
  if (sourceWebSocketRadio) sourceWebSocketRadio.addEventListener('change', () => { blendshapeSource = 'websocket'; renderChosenBars(); });
  if (simulateSmilingCheckbox && simulateNotSmilingCheckbox) {
    simulateSmilingCheckbox.addEventListener('change', () => {
      if (simulateSmilingCheckbox.checked) simulateNotSmilingCheckbox.checked = false;
    });
    simulateNotSmilingCheckbox.addEventListener('change', () => {
      if (simulateNotSmilingCheckbox.checked) simulateSmilingCheckbox.checked = false;
    });
  }
}

connectBtn.addEventListener('click', connect);
disconnectBtn.addEventListener('click', disconnect);

async function main() {
  startAlwaysOnMic();
  renderBlendshapesIntoList(null, blendshapesCameraEl);
  blendshapesWebSocketEl.innerHTML = '<span class="muted">Connecting…</span>';
  renderChosenBars();
  setupSourceToggle();
  initQSocket();

  videoPlaceholder.textContent = 'Requesting camera…';
  try {
    await initWebcam();
    blendshapesCameraEl.innerHTML = '<span class="muted">Loading face model…</span>';
    await initFaceLandmarker();
    renderBlendshapesIntoList(null, blendshapesCameraEl);
    renderChosenBars();
    detectFrame();
  } catch (e) {
    console.error(e);
    videoPlaceholder.classList.remove('hidden');
    videoPlaceholder.textContent = e.message || 'Camera error';
  }
}

main();
