# TTS V1

## Runtime flow

```text
agent.intent
  -> macOS say
  -> agent.speech.started
  -> agent.speech.ended | agent.speech.failed
```

All agent speech lifecycle events are persisted into the same episode history.
Unlike `agent.intent`, physical agent speech extends episode activity. An active
agent speech action prevents the 15-second idle closer from cutting the episode
while the agent is still talking.

## Barge-in

While TTS is active, an external `speech.started` event immediately terminates
the `say` subprocess. The terminal event is persisted as:

```text
agent.speech.ended
status=interrupted
interrupt_reason=external_speech_started
```

## Running

TTS is enabled by default:

```bash
python core.py
```

Optional voice/rate:

```bash
python core.py --tts-voice Yuna --tts-rate 180
```

List voices available on the Mac:

```bash
say -v '?'
```

Run without audible output while keeping reasoning:

```bash
python core.py --no-tts
```

## Current limitation: acoustic self-hearing

The software distinguishes agent output (`source=mouth`) from microphone input
(`source=ear`), but V1 does not yet implement acoustic echo cancellation. If the
Mac speaker is loud enough for the microphone to trigger VAD, the agent may
mistake its own audio for external speech and interrupt itself.

Headphones/EarPods as the output device are the safest V1 setup. Proper echo
reference/AEC or explicit self-audio suppression should be the next audio-layer
improvement before relying on speakerphone-style full-duplex barge-in.
