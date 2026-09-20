# Embodied Agent

로컬 Mac에서 동작하는 음성 기반 embodied agent 실험 프로젝트입니다.

현재 버전은 마이크 입력을 실시간으로 감지하고, Whisper STT 결과를 episode에 기록하며, Qwen3 기반 S2 realtime reasoner가 응답 여부와 내용을 판단합니다. 진행 중인 경험은 working memory로 압축되고, 종료된 episode는 final consolidation을 통해 장기기억 형태로 정리됩니다. Reasoner의 응답은 macOS `say`를 통해 실제 음성으로 출력되며 agent의 intent/action도 episode history에 기록됩니다.

## 현재 동작 흐름

```text
Mic
 ↓
Silero VAD
 ↓
speech.started / speech.ended
 ↓
Whisper STT worker
 ↓
speech.final
 ↓
TurnCoordinator
 ↓
S2 Fast Reasoner        thinking=false
 ├─ respond / wait
 └─ deliberate
       ↓
   S2 Deliberate        thinking=true
       ↓
   respond / wait
 ↓
agent.intent
 ↓
macOS say
 ↓
agent.speech.started
 ↓
agent.speech.ended
```

동시에 background에서는:

```text
speech.final
 ↓
Rolling Consolidation
 ↓
Working Memory

Episode closed + pending STT = 0
 ↓
Final Consolidation
 ↓
Long-term episode memory
```

## 주요 설계 원칙

- 마이크/VAD는 Whisper STT를 기다리지 않습니다.
- episode membership은 STT 완료 시점이 아니라 실제 발화 시간축에서 결정합니다.
- `speech.final`은 발화 종료 이벤트가 아니라 STT 결과 확정 이벤트입니다.
- conversation turn과 memory episode는 서로 다른 시간축을 사용합니다.
- 새 발화가 들어오면 이전 realtime reasoning 결과는 stale 처리합니다.
- realtime LLM 요청과 rolling/final background 요청은 별도 lane을 사용합니다.
- `agent.intent`와 실제 agent speech는 모두 history에 영속화합니다.
- working memory보다 최신 raw tail이 항상 우선합니다.

---

# 1. 요구 환경

현재 구현은 macOS 기준입니다.

권장 환경:

```text
macOS
Python 3.11+
llama.cpp llama-server
whisper.cpp whisper-cli
마이크 입력 장치
```

Python 패키지:

```bash
python -m pip install numpy sounddevice torch silero-vad
```

`sounddevice`가 PortAudio 관련 오류를 내는 경우 macOS에서 다음이 필요할 수 있습니다.

```bash
brew install portaudio
```

## Whisper 준비

`whisper.cpp`의 `whisper-cli`와 모델 파일이 필요합니다.

기본 `config.toml`은 다음 경로를 가정합니다.

```toml
[paths]
whisper = "~/whisper.cpp/build/bin/whisper-cli"
whisper_model = "~/whisper.cpp/models/ggml-small.bin"
```

로컬 설치 위치가 다르면 `config.toml`만 수정하면 됩니다.

## LLM 준비

현재 기본 모델은 Qwen3-4B GGUF를 llama.cpp server로 사용하는 구성을 가정합니다.

예:

```bash
llama-server \
  -hf Qwen/Qwen3-4B-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8080 \
  -c 8192 \
  -np 2
```

`-np 2`를 권장합니다. Realtime Reasoner와 rolling/final background inference가 서로를 완전히 직렬로 기다리지 않게 하기 위해서입니다.

---

# 2. 빠른 실행

먼저 LLM server를 실행합니다.

```bash
llama-server \
  -hf Qwen/Qwen3-4B-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8080 \
  -c 8192 \
  -np 2
```

그 다음 프로젝트 디렉터리에서:

```bash
python core.py --config ./config.toml
```

기본 설정에서는 다음 기능이 모두 켜집니다.

```text
VAD
STT
Episode persistence
Working memory
Rolling consolidation
Final consolidation
Realtime reasoner
Fast → deliberate escalation
TTS
Barge-in
```

종료는 `Ctrl+C`를 사용합니다.

---

# 3. config.toml

주요 런타임 설정은 `config.toml`에서 관리합니다.

## 경로

```toml
[paths]
ear = "./ear.py"
whisper = "~/whisper.cpp/build/bin/whisper-cli"
whisper_model = "~/whisper.cpp/models/ggml-small.bin"
memory_db = "./data/agent.db"
```

## Audio / VAD

```toml
[audio]
language = "ko"
threads = 6
# device = "MacBook Microphone"

[vad]
threshold = 0.70
min_silence_ms = 600
speech_pad_ms = 200
max_utterance_sec = 20.0
stt_queue_max = 8
```

`threshold`가 높을수록 VAD가 덜 민감해집니다.

대략적인 시작점:

```text
0.50  Silero 기본값, 비교적 민감
0.65  조금 보수적
0.70  현재 권장 시작점
0.75+ 작은 목소리를 놓칠 가능성 증가
```

`min_silence_ms`는 발화 시작 민감도가 아니라 "얼마나 조용해야 발화가 끝났다고 볼지"를 결정합니다.

## Episode

```toml
[episode]
idle_sec = 15.0
```

마지막 실제 interaction 이후 이 시간 동안 새 발화가 없으면 episode를 닫습니다.

## LLM

```toml
[llm]
url = "http://127.0.0.1:8080"
model = "local"
realtime_concurrency = 2
```

## Conversation turn

```toml
[turn]
grace_ms = 300
```

`speech.final` 이후 상대가 바로 말을 이어가는지 짧게 기다리는 시간입니다.

이 시간 안에 새 `speech.started`가 들어오면 pending realtime reasoning을 취소합니다.

## S2 Fast Reasoner

```toml
[reasoner.fast]
temperature = 0.30
max_tokens = 192
timeout_sec = 30.0
thinking = false
```

일반적인 대화 turn은 fast path가 처리합니다.

Qwen3의 thinking을 끄기 때문에 짧은 응답에서는 latency를 줄이고, reasoning token이 output budget을 모두 소진하는 문제도 피합니다.

## S2 Deliberate Reasoner

```toml
[reasoner.deliberate]
enabled = true
temperature = 0.35
max_tokens = 768
timeout_sec = 60.0
thinking = true
```

Fast Reasoner가 복잡한 다단계 사고가 필요하다고 판단한 경우에만 deliberate path로 승격합니다.

```text
Fast / no-think
    ↓
simple → respond
complex → deliberate
              ↓
         thinking=true
              ↓
            respond
```

## Rolling / Final memory

```toml
[rolling]
batch = 3
delay_sec = 8.0
temperature = 0.15
max_tokens = 800
timeout_sec = 120.0
thinking = true

[final]
poll_sec = 3.0
temperature = 0.15
max_tokens = 1024
timeout_sec = 120.0
thinking = true
```

Realtime response와 달리 memory consolidation은 latency보다 품질이 중요하므로 기본적으로 thinking mode를 사용합니다.

## TTS

```toml
[tts]
enabled = true
command = "/usr/bin/say"
# voice = "Yuna"
# rate = 180
```

macOS voice 목록 확인:

```bash
say -v '?'
```

TTS를 끄려면:

```toml
[tts]
enabled = false
```

또는 한 번만:

```bash
python core.py --no-tts
```

---

# 4. CLI override

`config.toml`이 기본 설정이고 CLI argument가 최종 override입니다.

예:

```bash
python core.py \
  --config ./config.toml \
  --vad-threshold 0.75 \
  --turn-grace-ms 200 \
  --reasoner-max-tokens 128
```

Thinking도 일시적으로 바꿀 수 있습니다.

```bash
# Fast reasoner thinking ON
python core.py --reasoner-thinking

# Fast reasoner thinking OFF
python core.py --no-reasoner-thinking

# Deliberate path 비활성화
python core.py --no-deliberate-enabled
```

전체 CLI 옵션은:

```bash
python core.py --help
```

로 확인합니다.

---

# 5. 주요 시간축

현재 서로 다른 목적의 timer가 존재합니다.

```text
VAD min_silence_ms
≈ 600ms
→ 한 utterance가 끝났는지 판단

Turn grace_ms
≈ 300ms
→ 상대가 conversation turn을 넘겼는지 판단

Episode idle_sec
≈ 15s
→ 하나의 memory episode를 종료할지 판단
```

이 세 값은 같은 timeout이 아닙니다.

---

# 6. Event lifecycle

외부 음성:

```text
speech.started
speech.ended
speech.final
```

`speech.final`은 Whisper가 성공적으로 transcription을 끝냈을 때 발생합니다.

STT 실패 시:

```text
speech.failed
```

Reasoner:

```text
agent.intent
reasoner.failed
```

실제 agent speech:

```text
agent.speech.started
agent.speech.ended
agent.speech.failed
```

Barge-in이 발생하면 `agent.speech.ended`에 interrupted 상태가 기록됩니다.

```text
status = interrupted
interrupt_reason = external_speech_started
```

---

# 7. Persistence

기본 DB:

```text
./data/agent.db
```

SQLite + WAL을 사용합니다.

주요 데이터:

```text
events
episodes
episode_events
episode_working_memory
```

`agent.intent`도 episode history에 저장됩니다.

다만 intent와 실행된 action은 구분합니다.

```text
agent.intent
= 무엇을 하기로 판단했는가

agent.speech.started / ended
= 실제로 행동했는가
```

Reasoner의 내부 판단 자체는 episode idle timer를 연장하지 않지만, 실제 agent speech는 physical interaction으로 취급합니다.

---

# 8. 테스트

전체 테스트:

```bash
python -m unittest -v
```

ResourceWarning까지 오류로 처리:

```bash
python -W error::ResourceWarning -m unittest -v
```

현재 테스트 범위에는 다음이 포함됩니다.

- physical-time episode assignment
- delayed STT / late `speech.final`
- working-memory CAS
- stale raw tail preservation
- rolling single-flight/coalescing
- final consolidation readiness
- realtime/background LLM lane
- stale reasoning discard
- fast/non-thinking Reasoner
- deliberate/thinking escalation
- config merge / CLI override
- reasoner event persistence
- TTS lifecycle
- barge-in

---

# 9. 로그에서 자주 보는 항목

정상적으로 시작되면 stderr에서 대략 다음을 볼 수 있습니다.

```text
[core] memory db: ...
[core] config: ./config.toml
[core] episode idle timeout: 15.0s
[core] turn grace: 300ms
[core] reasoner fast: thinking=False ...
[stt] worker started
```

Reasoner가 fast path에서 바로 응답하면:

```text
reasoning_mode=fast
```

복잡한 질문을 deliberate path로 승격하면:

```text
reasoning_mode=deliberate
```

이 값은 `agent.intent`에도 저장됩니다.

---

# 10. 현재 알려진 제한사항

## Acoustic self-hearing

현재 agent speech와 external speech는 논리적 event source로는 분리되어 있지만, Mac 스피커에서 나온 TTS가 실제 마이크로 다시 들어오는 acoustic echo cancellation은 아직 없습니다.

따라서 스피커 출력이 마이크에 크게 들어오면 자기 TTS를 외부 발화로 오인해 barge-in할 수 있습니다.

현재는 이어폰/헤드폰 출력 사용이 가장 안정적입니다.

## VAD start debounce

현재 `vad.threshold` 튜닝은 가능하지만, 짧은 충격음이 순간적으로 speech probability를 넘는 문제를 줄이기 위한 별도 `min_speech_ms` start debounce는 아직 구현 대상입니다.

---

# 11. 프로젝트 파일

```text
core.py            전체 orchestration
ear.py             mic capture + Silero VAD
stt.py             background Whisper worker
episode.py         realtime episode assignment
memory.py          SQLite persistence
context.py         realtime context snapshot
rolling.py         active episode working-memory consolidation
consolidation.py   final episode consolidation
llm_client.py      realtime/background LLM lanes
reasoner.py        S2 fast + deliberate reasoning
turn.py            turn grace / interruption / stale control
tts.py             macOS say + barge-in
config.py          TOML config loader
config.toml        기본 runtime 설정
```

추가 설명:

```text
README_CONFIG.md   config 상세 설명
README_TTS.md      TTS/barge-in 상세 설명
```

---

# 현재 상태 한 줄 요약

현재 시스템은 외부 음성을 듣고, 실제 시간축에 경험을 기록하고, 진행 중인 경험을 working memory로 유지하면서, 빠른 S2와 필요시 깊은 S2 reasoning을 통해 응답을 결정하고, 실제 음성 행동까지 history에 남기는 로컬 embodied agent 프로토타입입니다.
