import dotenv from 'dotenv';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: join(__dirname, '.env') });

// ANSI colors for terminal logs: blue = send to OpenAI, yellow = receive from OpenAI, green = turnComplete
const C = { blue: '\x1b[34m', yellow: '\x1b[33m', green: '\x1b[32m', reset: '\x1b[0m' };

const OPENAI_REALTIME_URL = 'wss://api.openai.com/v1/realtime';
const PORT = Number(process.env.PORT) || 3080;
const apiKey = process.env.OPENAI_API_KEY;
const model = process.env.OPENAI_REALTIME_MODEL || 'gpt-realtime';
const voice = process.env.OPENAI_VOICE || 'marin';

const SYSTEM_INSTRUCTION = `Role & Voice
You are a friendly, empathetic, and natural conversational partner. Your output is audio-only. Be concise, warm, and engaging. Talk about anything the user wants — their day, feelings, interests, questions, stories, or whatever comes up. Never use markdown, text formatting, or narration of your own actions. Follow the user's lead and match the tone of the conversation.

Processing Blendshapes
  You receive snapshots of facial blendshapes (e.g., mouthSmileLeft: 0.8).
  Acknowledge only when: The user asks ("Am I smiling?"), or there is a dramatic shift (e.g., a sudden jump from 0.1 to 0.7).
  Thresholds:
    Values >0.5 indicate a smile;
    values <0.3 indicate a neutral expression.

REACTION WINDOW — Emotional Feedback
  After your responses you will receive a message starting with "REACTION WINDOW" containing the user's facial expression data (smile values).
  How you handle it depends on whether you are currently in Joke Mode or not:

  During normal conversation:
    Use the smile data as emotional context only. For example, if the user is smiling, what you said resonated — you can acknowledge that naturally or simply continue the conversation. If they look neutral or unhappy, you might gently adjust your tone, ask if something is wrong, or just keep going. Do NOT tell a joke. Do NOT enter Joke Mode. Simply continue the conversation naturally.

  During Joke Mode (see below):
    Follow the Joke Mode REACTION WINDOW rules described in that section.

Joke Mode
  Joke Mode is a special sub-mode. You are NOT in Joke Mode by default.

  Entering Joke Mode:
    Only enter Joke Mode when the user explicitly asks for a joke, asks you to be funny, asks to be cheered up with humor, or clearly requests humor. Never enter Joke Mode on your own initiative.

  While in Joke Mode:
    Selection: Choose from One-Liners, Puns, or Knock-Knock jokes.
    Delivery: Deliver the entire joke in one single turn. For Knock-Knock jokes, perform both parts yourself (e.g., "Knock knock. Who's there? Cows go. Cows go who? No, cows go moo!").
    The Silent Wait (CRITICAL): After the punchline, stop speaking immediately. Do not ask "Did you like it?" or continue talking. Wait for the REACTION WINDOW.
    REACTION WINDOW in Joke Mode: When you receive a REACTION WINDOW while in Joke Mode, do BOTH in one turn:
      1) Brief acknowledgment of their reaction, then
      2) Immediately tell another full joke — do not stop after the acknowledgment.
      If Smile >0.5: Say a very brief line like "Glad you liked that one!" or "Nice!" then immediately deliver another joke of the SAME category.
      If Smile <0.3: Say a very brief line like "Tough crowd. Let's try a pun instead..." then immediately deliver a full joke from a DIFFERENT category. Do not stop after the pivot — the next joke must follow in the same turn.
      After that next joke, stop and wait again for the next REACTION WINDOW. Never end your turn with only an acknowledgment or only a pivot; always follow with a complete joke.
    REACTION WINDOW messages refer to the user's reaction to your last message only, not earlier history.

  Exiting Joke Mode:
    If the user says anything unrelated to jokes — changes the topic, asks a question, shares something personal, or starts a new conversation — immediately exit Joke Mode. Respond naturally to what they said. Do not tell another joke. From that point, treat any REACTION WINDOW as emotional context (normal conversation rules).

Constraints
  No Multi-turn Jokes: Never wait for the user to say "Who's there?".
  No Guessing: Base all physical feedback strictly on the provided blendshape data.
  Audio Only: Never output text descriptions like laughs or smiles.
  No Unsolicited Jokes: Never tell a joke unless the user has explicitly asked for one or you are currently in Joke Mode.
  In Joke Mode after REACTION WINDOW: Never end your turn with only an acknowledgment or pivot — always deliver a complete joke, then stop.

Do not respond to hebrew or russian speech, completely ignore it, do not respond, do not translate, do not acknowledge it in any way. Treat it as background noise. Do not respond to user reaction to your joke until you get REACTION WINDOW smile values after you finish the joke.`;

function connectToOpenAI(openaiApiKey) {
  const url = `${OPENAI_REALTIME_URL}?model=${encodeURIComponent(model)}`;
  return new WebSocket(url, {
    headers: {
      Authorization: `Bearer ${openaiApiKey}`,
    },
  });
}

function sendToClient(clientWs, obj) {
  if (clientWs.readyState === WebSocket.OPEN) {
    clientWs.send(JSON.stringify(obj));
  }
}

/** Send input_audio_buffer.commit then response.create so user audio is committed before generation. */
function sendCommitThenResponseCreate(openaiWs) {
  openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
  openaiWs.send(JSON.stringify({ type: 'response.create' }));
}

const httpServer = http.createServer((_req, res) => {
  res.writeHead(404);
  res.end();
});

const wss = new WebSocketServer({ server: httpServer });

wss.on('connection', (clientWs, req) => {
  const clientAddress = req.socket.remoteAddress;
  console.log(`[${new Date().toISOString()}] Client connected from ${clientAddress}`);

  if (!apiKey) {
    clientWs.send(JSON.stringify({ error: 'OPENAI_API_KEY not configured' }));
    clientWs.close();
    return;
  }

  const openaiWs = connectToOpenAI(apiKey);
  let setupComplete = false;
  const pendingClientMessages = [];
  const MAX_PENDING = 30;
  let blendshapeLogCount = 0;
  let lastOutputTranscriptText = '';
  let echoDetectionBuffer = '';
  const _DL = '/Users/katya.ivantsiv/development/emotional_chat/.cursor/debug-5debac.log';
  let _responseCount = 0;
  let responseInProgress = false;
  let pendingResponseCreate = false;
  /** REACTION WINDOW item deferred until response.done — do not send emotion until assistant turn is fully done. */
  let pendingReactionItem = null;

  openaiWs.on('open', () => {
    console.log('[OpenAI] WebSocket open, waiting for session.created');
  });

  openaiWs.on('message', (data) => {
    try {
      const payload = data.toString();
      const parsed = JSON.parse(payload);
      const eventType = parsed.type;
      // #region agent log
      try { fs.appendFileSync('/Users/katya.ivantsiv/development/emotional_chat/.cursor/debug-627095.log', JSON.stringify({sessionId:'627095',location:'server/index.js:event_received',message:'OpenAI event',data:{eventType},timestamp:Date.now(),hypothesisId:'H1_H2_H3'}) + '\n'); } catch (_) {}
      // #endregion

      if (eventType === 'session.created') {
        console.log('[OpenAI] session.created received, sending session.update');
        const sessionUpdate = {
          type: 'session.update',
          session: {
            type: 'realtime',
            instructions: SYSTEM_INSTRUCTION,
            output_modalities: ['audio'],
            audio: {
              input: {
                format: { type: 'audio/pcm', rate: 24000 },
                turn_detection: {
                  type: 'semantic_vad',
                  interrupt_response: true,
                },
              },
              output: {
                format: { type: 'audio/pcm', rate: 24000 },
                voice,
              },
            },
          },
        };
        openaiWs.send(JSON.stringify(sessionUpdate));
        // #region agent log
        try { fs.appendFileSync('/Users/katya.ivantsiv/development/emotional_chat/.cursor/debug-627095.log', JSON.stringify({sessionId:'627095',location:'server/index.js:session_update_sent',message:'session.update sent to OpenAI',data:{},timestamp:Date.now(),hypothesisId:'H1'}) + '\n'); } catch (_) {}
        // #endregion
        return;
      }

      if (eventType === 'session.updated') {
        console.log('[OpenAI] session.updated received, ready for input');
        // #region agent log
        try { fs.appendFileSync('/Users/katya.ivantsiv/development/emotional_chat/.cursor/debug-627095.log', JSON.stringify({sessionId:'627095',location:'server/index.js:session_updated',message:'session.updated received OK',data:{},timestamp:Date.now(),hypothesisId:'H1'}) + '\n'); } catch (_) {}
        // #endregion
        setupComplete = true;
        sendToClient(clientWs, { setupComplete: true });
        for (const msg of pendingClientMessages) {
          openaiWs.send(msg);
        }
        pendingClientMessages.length = 0;
        // Greeting: model says hello first
        openaiWs.send(JSON.stringify({
          type: 'conversation.item.create',
          item: {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: "Say a brief hello and that you're ready to chat." }],
          },
        }));
        openaiWs.send(JSON.stringify({ type: 'response.create' }));
        responseInProgress = true;
        // #region agent log
        try { fs.appendFileSync('/Users/katya.ivantsiv/development/emotional_chat/.cursor/debug-627095.log', JSON.stringify({sessionId:'627095',location:'server/index.js:greeting_sent',message:'greeting item + response.create sent (no commit)',data:{},timestamp:Date.now(),hypothesisId:'H1'}) + '\n'); } catch (_) {}
        // #endregion
        return;
      }

      if (eventType === 'error') {
        const errMsg = parsed.error?.message || JSON.stringify(parsed.error) || 'OpenAI error';
        console.error('[OpenAI] Error from API:', errMsg);
        // #region agent log
        try { fs.appendFileSync('/Users/katya.ivantsiv/development/emotional_chat/.cursor/debug-627095.log', JSON.stringify({sessionId:'627095',location:'server/index.js:api_error',message:'OpenAI API error',data:{errMsg},timestamp:Date.now(),hypothesisId:'H1'}) + '\n'); } catch (_) {}
        // #endregion
        sendToClient(clientWs, { error: errMsg });
        return;
      }

      if (eventType === 'response.cancelled') {
        responseInProgress = false;
        // #region agent log
        try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:response.cancelled',message:'response CANCELLED by API',data:{responseId:parsed.response_id||'',pendingResponseCreate},timestamp:Date.now(),hypothesisId:'H2'}) + '\n'); } catch (_) {}
        // #endregion
        sendToClient(clientWs, { serverContent: { interrupted: true } });
        if (pendingResponseCreate) {
          pendingResponseCreate = false;
          // Do not commit: buffer is empty after interrupt. Only start the deferred response.
          openaiWs.send(JSON.stringify({ type: 'response.create' }));
          responseInProgress = true;
        }
        return;
      }

      if (eventType === 'response.output_audio.delta' && parsed.delta) {
        responseInProgress = true;
        sendToClient(clientWs, {
          serverContent: {
            modelTurn: { parts: [{ inline_data: { data: parsed.delta } }] },
          },
        });
        return;
      }

      if (eventType === 'response.output_audio_transcript.delta' && parsed.delta) {
        lastOutputTranscriptText += typeof parsed.delta === 'string' ? parsed.delta : (parsed.delta?.text ?? '');
        console.log(`${C.yellow}[OpenAI reply] ts=${Date.now()} "${parsed.delta}"${C.reset}`);
        sendToClient(clientWs, {
          serverContent: { outputTranscription: { text: parsed.delta } },
        });
        return;
      }

      if (eventType === 'response.output_audio_transcript.done' && parsed.transcript) {
        const text = typeof parsed.transcript === 'string' ? parsed.transcript : parsed.transcript?.text ?? '';
        if (text) lastOutputTranscriptText = text;
        // #region agent log
        try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:transcript_done',message:'full transcript for response',data:{transcript:text.slice(0,200)},timestamp:Date.now(),hypothesisId:'H4'}) + '\n'); } catch (_) {}
        // #endregion
        if (text) {
          sendToClient(clientWs, {
            serverContent: { outputTranscription: { text } },
          });
        }
        return;
      }

      // Input transcript: conversation.item.input_audio_transcript (completed or done)
      if ((eventType === 'conversation.item.input_audio_transcript.completed' || eventType === 'conversation.item.input_audio_transcript.done') && (parsed.transcript != null || parsed.item?.input_audio_transcript != null)) {
        const raw = parsed.transcript ?? parsed.item?.input_audio_transcript;
        const text = typeof raw === 'string' ? raw : raw?.text ?? '';
        if (text) {
          const lastOutSnippet = (lastOutputTranscriptText || echoDetectionBuffer).slice(-200);
          const trimmed = text.trim().toLowerCase();
          const overlap = lastOutSnippet.length > 0 && trimmed.length > 0 && lastOutSnippet.toLowerCase().includes(trimmed);
          if (overlap) return;
          sendToClient(clientWs, {
            serverContent: { inputTranscription: { text } },
          });
          echoDetectionBuffer = '';
        }
        return;
      }

      if (eventType === 'response.done') {
        const status = parsed.response?.status;
        _responseCount++;
        responseInProgress = false;
        // #region agent log
        try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:response.done',message:'response.done received',data:{status,responseCount:_responseCount,transcriptSnippet:(lastOutputTranscriptText||'').slice(0,100),pendingResponseCreate},timestamp:Date.now(),hypothesisId:'H1_H2_H3'}) + '\n'); } catch (_) {}
        // #endregion
        console.log(`${C.green}[OpenAI] turnComplete ts=${Date.now()} (response.done)${C.reset}`);
        sendToClient(clientWs, { serverContent: { turnComplete: true } });
        echoDetectionBuffer = lastOutputTranscriptText;
        lastOutputTranscriptText = '';
        if (pendingReactionItem) {
          const itemToSend = pendingReactionItem;
          pendingReactionItem = null;
          openaiWs.send(itemToSend);
          sendCommitThenResponseCreate(openaiWs);
          responseInProgress = true;
        } else if (pendingResponseCreate) {
          pendingResponseCreate = false;
          // #region agent log
          try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:deferred_response_create',message:'firing deferred response.create',data:{},timestamp:Date.now(),hypothesisId:'H1'}) + '\n'); } catch (_) {}
          // #endregion
          sendCommitThenResponseCreate(openaiWs);
          responseInProgress = true;
        }
        return;
      }
    } catch (err) {
      const asText = data.toString('utf8');
      if (asText.trimStart().startsWith('{') && clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(asText);
      } else {
        console.warn('[OpenAI] Dropped non-JSON message (%d bytes)', data.length ?? asText.length);
      }
    }
  });

  openaiWs.on('error', (err) => {
    console.error('[OpenAI] WebSocket error:', err.message || err);
    sendToClient(clientWs, { error: 'OpenAI connection error: ' + String(err.message) });
  });

  openaiWs.on('close', (code, reason) => {
    const reasonStr = reason && reason.length ? reason.toString() : `code ${code}`;
    console.error('[OpenAI] WebSocket closed:', reasonStr);
    sendToClient(clientWs, { error: 'OpenAI closed: ' + reasonStr });
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.close(code, reason);
    }
  });

  clientWs.on('message', (data) => {
    if (openaiWs.readyState !== WebSocket.OPEN) return;
    try {
      const msg = JSON.parse(data.toString());
      let payload = null;

      if (msg.type === 'audio' && msg.data != null) {
        payload = JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: msg.data,
        });
      } else if (msg.type === 'text' && msg.data != null) {
        blendshapeLogCount += 1;
        const dataStr = String(msg.data);
        const smileLMatch = dataStr.match(/mouthSmileLeft:\s*([\d.]+)/);
        const smileRMatch = dataStr.match(/mouthSmileRight:\s*([\d.]+)/);
        const smileL = smileLMatch ? smileLMatch[1] : '?';
        const smileR = smileRMatch ? smileRMatch[1] : '?';
        const turnComplete = msg.turnComplete === true;
        const isReactionWindow = dataStr.startsWith('REACTION WINDOW');
        const color = turnComplete ? C.green : C.blue;
        console.log(`${color}[Blendshape] #${blendshapeLogCount} ts=${Date.now()} smileL=${smileL} smileR=${smileR} turnComplete=${turnComplete}${C.reset}`);

        const itemPayload = JSON.stringify({
          type: 'conversation.item.create',
          item: {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: msg.data }],
          },
        });
        if (setupComplete) {
          if (isReactionWindow && turnComplete) {
            // TURN 2: Send REACTION WINDOW only after assistant turn is fully done (response.done). Do not send during streaming.
            if (responseInProgress) {
              pendingReactionItem = itemPayload;
              console.log(`${C.blue}[Blendshape] REACTION WINDOW deferred until response.done (assistant still speaking)${C.reset}`);
              // #region agent log
              try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:reaction_deferred',message:'REACTION WINDOW deferred until response.done',data:{smileL,smileR},timestamp:Date.now(),hypothesisId:'H1_H3'}) + '\n'); } catch (_) {}
              // #endregion
            } else {
              openaiWs.send(itemPayload);
              sendCommitThenResponseCreate(openaiWs);
              responseInProgress = true;
            }
          } else {
            // Non-reaction text (e.g. blendshapes during turn); or reaction without turnComplete — send item, optionally response.create
            openaiWs.send(itemPayload);
            if (turnComplete) {
              if (responseInProgress) {
                pendingResponseCreate = true;
                // #region agent log
                try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:text_response_deferred',message:'response.create DEFERRED (response in progress)',data:{isReactionWindow:false,smileL,smileR},timestamp:Date.now(),hypothesisId:'H1_H3'}) + '\n'); } catch (_) {}
                // #endregion
              } else {
                // #region agent log
                try { fs.appendFileSync(_DL, JSON.stringify({sessionId:'5debac',location:'server:text_response_create',message:'response.create sent immediately',data:{isReactionWindow:false,smileL,smileR},timestamp:Date.now(),hypothesisId:'H1_H3'}) + '\n'); } catch (_) {}
                // #endregion
                sendCommitThenResponseCreate(openaiWs);
                responseInProgress = true;
              }
            }
          }
        } else {
          if (pendingClientMessages.length >= MAX_PENDING) pendingClientMessages.shift();
          pendingClientMessages.push(itemPayload);
          if (turnComplete) {
            pendingClientMessages.push(JSON.stringify({ type: 'input_audio_buffer.commit' }));
            pendingClientMessages.push(JSON.stringify({ type: 'response.create' }));
          }
        }
        return;
      } else if (msg.audioStreamEnd === true) {
        payload = JSON.stringify({ type: 'input_audio_buffer.commit' });
      }

      if (!payload) return;
      if (setupComplete) {
        openaiWs.send(payload);
      } else {
        if (pendingClientMessages.length >= MAX_PENDING) pendingClientMessages.shift();
        pendingClientMessages.push(payload);
      }
    } catch (err) {
      console.error('Parse client message error:', err);
    }
  });

  clientWs.on('close', () => {
    openaiWs.close();
  });

  clientWs.on('error', (err) => {
    console.error('Client WebSocket error:', err);
    openaiWs.close();
  });
});

httpServer.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT} (WebSocket for OpenAI Realtime)`);
  console.log('Set OPENAI_API_KEY in server/.env');
});
