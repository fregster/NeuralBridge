# Alexa Integration

This guide explains how to connect an Amazon Alexa Skill to NeuralBridge so that Alexa voice input is routed through NeuralBridge's priority-based AI backends.

## Architecture Overview

```
User speaks to Echo device
         │
         ▼
Amazon cloud (speech-to-text + NLU)
         │  transcribed text
         ▼
Alexa Skill webhook (separate project, self-hosted)
         │  POST /api/conversation/process
         ▼
Home Assistant Conversation API
         │  agent_id = conversation.neuralbridge
         ▼
NeuralBridge routes through configured agents
(local HA → Ollama → cloud LLMs, in priority order)
         │
         ▼
Response text returned to Alexa Skill → spoken by Alexa
```

> **Privacy note:** Alexa's speech-to-text always runs on Amazon's servers before the
> text reaches your Skill or Home Assistant. After that point NeuralBridge can route
> entirely locally (e.g. Ollama). If full local audio processing is required, use a
> local voice satellite connected directly to HA's Assist pipeline instead.

---

## Prerequisites

1. **NeuralBridge installed** with at least one agent configured.
2. **Home Assistant accessible via HTTPS** — the Alexa Skill must reach your HA instance
   over the internet using a valid TLS certificate. Common options:
   - [Nabu Casa / Home Assistant Cloud](https://www.nabucasa.com/) — easiest, provides
     `<your-id>.ui.nabu.casa`
   - DuckDNS + Let's Encrypt reverse proxy (nginx, Caddy, Traefik)
   - Any DNS hostname with a valid, publicly-trusted TLS certificate
3. **A separate Alexa Skill project** configured with the values from the steps below.

---

## Step 1 — Generate a Long-Lived Access Token

The Alexa Skill authenticates to HA using a long-lived access token.

1. In Home Assistant, navigate to your **Profile** (bottom-left avatar).
2. Scroll to **Long-Lived Access Tokens** and click **Create Token**.
3. Give it a descriptive name (e.g. `Alexa Skill`).
4. Copy the token — it is shown only once.
5. Store it securely in your Alexa Skill's configuration (environment variable or
   secrets manager). **Never commit it to source control.**

---

## Step 2 — Find Your NeuralBridge Entity ID

The entity ID is used as the `agent_id` in every API call.

- Default: `conversation.neuralbridge`
- To confirm: go to **Settings → Devices & Services → NeuralBridge** and check the
  entity listed, or run a quick test in **Developer Tools → Template**:

  ```
  {{ states.conversation | map(attribute='entity_id') | list }}
  ```

---

## Step 3 — Configure the Alexa Skill

Configure your Alexa Skill project with these three values:

| Setting | Value |
|---|---|
| `HA_URL` | Your HA base URL, e.g. `https://your-home.duckdns.org` |
| `HA_TOKEN` | The long-lived access token from Step 1 |
| `AGENT_ID` | `conversation.neuralbridge` (or your entity ID from Step 2) |

---

## HA Conversation API Reference

The Alexa Skill calls this endpoint for every user turn:

```
POST {HA_URL}/api/conversation/process
Authorization: Bearer {HA_TOKEN}
Content-Type: application/json

{
  "text": "Turn off the kitchen lights",
  "conversation_id": "amzn1.echo-api.session.abc123",
  "agent_id": "conversation.neuralbridge",
  "language": "en"
}
```

**Response:**

```json
{
  "response": {
    "speech": {
      "plain": {
        "speech": "OK, turning off the kitchen lights.",
        "extra_data": null
      }
    },
    "card": null,
    "language": "en",
    "response_type": "action_done",
    "data": {}
  },
  "conversation_id": "amzn1.echo-api.session.abc123"
}
```

The `speech.plain.speech` string is what the Alexa Skill should return as the spoken
response. The `conversation_id` in the response matches the one sent, confirming
NeuralBridge echoes it back correctly.

---

## Session Lifecycle (Multi-Turn Conversations)

NeuralBridge maintains per-session conversation history for Ollama agents using the
Alexa `sessionId` as the `conversation_id`. This enables multi-turn exchanges:

```
Turn 1  →  { "text": "Who wrote Dune?", "conversation_id": "amzn1.echo-api.session.XYZ" }
           ←  "Frank Herbert wrote Dune."

Turn 2  →  { "text": "What else did he write?", "conversation_id": "amzn1.echo-api.session.XYZ" }
           ←  "He also wrote the Destination Void series and The Dosadi Experiment."
           (NeuralBridge supplies the previous turn as context to Ollama)
```

### Session Cleanup

When the Alexa session ends, call the `clear_conversation` HA service to free memory
immediately instead of waiting for the 30-minute TTL:

```
POST {HA_URL}/api/services/neuralbridge/clear_conversation
Authorization: Bearer {HA_TOKEN}
Content-Type: application/json

{
  "conversation_id": "amzn1.echo-api.session.XYZ"
}
```

This is a synchronous fire-and-forget call; a `200 OK` with an empty body confirms
success.

> **Note:** Sessions auto-expire after 30 minutes of inactivity regardless of whether
> the service is called. Calling it explicitly is recommended to keep memory usage low.

---

## Recommended Agent Configuration for Voice

Alexa responses should be short, plain sentences — no markdown, no bullet lists.
Configure agents in this priority order:

| Priority | Agent type | Purpose |
|---|---|---|
| 10 | Home Assistant (local) | Device control commands |
| 20 | Ollama (local model) | General questions, with a voice-appropriate system prompt |
| 50 | Existing integration (optional) | Cloud LLM fallback |

For Ollama agents set a system prompt that keeps responses concise:

> "You are a helpful voice assistant. Respond in plain, spoken English only.
> Keep responses under 2 sentences. Do not use markdown, lists, or formatting."

See [alexa_integration.yaml](../../config/examples/alexa_integration.yaml) for a
complete configuration example.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `401 Unauthorized` from HA | Token is wrong or expired — regenerate in HA Profile |
| `404 Not Found` for `conversation.neuralbridge` | Integration not set up, or entity ID is different — verify in HA |
| Alexa says "there was a problem" | HA is unreachable — check DNS, TLS cert, and firewall |
| No conversation history on turn 2 | Alexa Skill not passing the same `conversation_id` on each turn |
| Memory grows over time | Call `clear_conversation` service when Alexa session ends |
