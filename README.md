# Emotional Chat – Realtime Conversation

Voice conversation with the **OpenAI Realtime API**, using:

- **Audio** from your microphone (24 kHz)
- **ARKit-52 facial blendshapes** from either:
  - **Webcam** (MediaPipe Face Landmarker), or
  - **WebSocket (Q)** – external blendshape stream

The model infers emotions from the blendshapes and addresses them in the conversation. Output is spoken audio plus live transcripts.

## Requirements

- **Node.js 18+** (for Vite and the backend)

## Setup

1. **Install dependencies**

   ```bash
   npm install
   cd server && npm install
   cd ../frontend && npm install
   ```

2. **API key**

   Create `server/.env` with:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

   Optional in `server/.env`:
   - `OPENAI_REALTIME_MODEL` – Realtime model (default: `gpt-realtime`)
   - `OPENAI_VOICE` – Voice for output (e.g. `marin`, `cedar`, `alloy`)

   Input and output transcription use the Realtime API’s built-in transcription.

## Run

From the project root:

```bash
npm run dev
```

This starts:

- **Backend** (WebSocket proxy to OpenAI Realtime) at `ws://localhost:3080`
- **Frontend** (Vite) – open the URL shown (e.g. http://localhost:5173)

Then:

1. Allow **microphone** when prompted – the **mic level bar** runs as soon as the page loads.
2. Allow **camera** for blendshapes (or use Q).
3. Choose blendshape source: **Camera (MediaPipe)** or **Q** (WebSocket).
4. Click **Connect** to start the Realtime session. Speak; the model responds with audio. **Input** and **output** transcripts come from the Realtime API’s transcription.

## Optional: Q (WebSocket) blendshapes

To feed blendshapes from an external app (e.g. a separate process or device), run a Socket.IO server that emits `blendshapes_data` with either:

- `{ blendshapes: [ 0, 0.1, ... ] }` – 52 numbers in ARKit-52 order (see plan), or  
- `{ blendshapes: { jawOpen: 0.2, mouthSmileLeft: 0.5, ... } }`

Set `VITE_SOCKET_URL` (e.g. in `frontend/.env`) to that server’s URL (default `http://localhost:8181`).

## Tech

- **Backend**: Node, `ws` – WebSocket proxy to OpenAI Realtime API.
- **Frontend**: Vite, MediaPipe, Socket.IO. **Always-on mic** level bar; when connected, mic stream is sent at 24 kHz. **Input** and **output** transcripts from the Realtime API; output audio is 24 kHz playback.
