# Runtime Config + S2 Fast/Deliberate Reasoning

## Run

```bash
python core.py --config ./config.toml
```

`config.toml` is now the normal place to tune runtime behavior. Existing CLI flags still work and override the config for a single run.

Example:

```bash
python core.py --config ./config.toml --vad-threshold 0.75 --turn-grace-ms 250
```

## S2 reasoning modes

Normal turns use the fast path:

```text
speech.final
  -> 300 ms turn grace
  -> S2 fast (thinking=false, 192 tokens)
  -> respond / wait
```

For a turn that actually needs deeper reasoning, the fast pass may return `intent=deliberate`:

```text
S2 fast (non-thinking)
  -> deliberate
  -> S2 deliberate (thinking=true, larger token budget)
  -> respond / wait
```

The selected mode is persisted on `agent.intent.payload.reasoning_mode` as `fast` or `deliberate`.

Both request-level `chat_template_kwargs.enable_thinking` and Qwen3's `/no_think`/`/think` soft switches are used. This keeps fast and deliberate requests independent even though they share one llama-server.

## Important config sections

```toml
[vad]
threshold = 0.70
min_silence_ms = 600
speech_pad_ms = 200

[turn]
grace_ms = 300

[reasoner.fast]
temperature = 0.30
max_tokens = 192
timeout_sec = 30.0
thinking = false

[reasoner.deliberate]
enabled = true
temperature = 0.35
max_tokens = 768
timeout_sec = 60.0
thinking = true
```

Rolling and final consolidation have their own LLM settings too, so their thinking modes can be tuned separately without changing realtime S2.
