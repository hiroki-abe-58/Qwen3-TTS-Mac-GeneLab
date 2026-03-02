<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Apple Silicon Mac에 완전 최적화된 Qwen3-TTS 포크<br>
  듀얼 엔진 (MLX + PyTorch)으로 네이티브 Mac TTS 경험 제공
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <strong>한국어</strong> |
  <a href="README_RU.md">Русский</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_IT.md">Italiano</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_PT.md">Português</a>
</p>

<p align="center">
  <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/stargazers">
    <img src="../assets/star.gif" alt="Star this repo!" width="580">
  </a>
  <br>
  <sub>이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!</sub>
</p>

---

## 왜 Qwen3-TTS-Mac-GeneLab인가?

| 기능 | 공식 Qwen3-TTS | **본 프로젝트** |
|------|----------------|----------------|
| Apple Silicon 최적화 | 제한적 | **완전 지원** |
| MLX 네이티브 추론 | 없음 | **지원** (8bit/4bit 양자화) |
| PyTorch MPS | 수동 설정 필요 | **자동 전환** |
| GUI | 없음 | **10개 언어 Web UI** |
| Voice Clone | CLI만 지원 | **Web UI + Whisper 자동 전사** |
| 메모리 관리 | 없음 | **통합 메모리 최적화** |
| 설치 | 복잡함 | **한 줄 명령어** |

### 핵심 혁신

1. **듀얼 엔진 아키텍처**
   - MLX: Apple Silicon 네이티브, 8bit/4bit 양자화로 속도 및 메모리 효율 향상
   - PyTorch: Voice Clone용 자동 전환 (float32 CPU 실행)

2. **작업 기반 자동 최적화**
   - CustomVoice -> MLX 우선 (고속)
   - VoiceDesign -> MLX 우선 (고속)
   - VoiceClone -> PyTorch CPU (float32 필수)

3. **10개 언어 Web UI**
   - Gradio 기반의 직관적인 인터페이스
   - 상단 드롭다운에서 언어 전환 가능

---

## 시스템 요구 사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| 칩 | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| OS | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| 여유 저장 공간 | 10GB | 20GB+ |

> **Windows를 찾고 계신가요?** [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP)를 확인하세요 — NVIDIA GPU를 지원하는 Windows 네이티브 버전 (RTX 5090 테스트 완료).

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. 설치 (최초 1회만, 약 5~10분)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Web UI 실행

**방법 A: 더블 클릭 (권장)**

Finder에서 `run.command`를 더블 클릭하면 터미널에서 자동 실행됩니다.

**방법 B: 터미널에서 실행**

```bash
./run.sh
```

> 포트가 이미 사용 중인 경우 사용 가능한 포트가 자동으로 감지됩니다.

### 4. 브라우저에서 열기

http://localhost:7860 을 엽니다 (포트가 변경된 경우 터미널 출력을 확인하세요)

---

## Web UI 기능

### Custom Voice
9개의 프리셋 화자로 음성을 생성합니다. 감정 제어와 10개 언어를 지원합니다.

### Voice Design
텍스트로 음성 특성을 설명하여 그에 맞는 음성을 생성합니다.

### Voice Clone
단 3초의 참조 오디오로 목소리를 클론합니다. Whisper 자동 전사를 지원합니다.

> **참고**: Voice Clone에는 **Base 모델** (~3.8GB)이 필요하며, 처음 사용 시 자동으로 다운로드됩니다.

### Settings
엔진 선택 (AUTO/MLX/PyTorch), 메모리 모니터, 모델 관리.

---

## CLI 사용법

```python
from mac import DualEngine, TaskType
import soundfile as sf

engine = DualEngine()

result = engine.generate(
    text="안녕하세요, 음성 합성 데모입니다.",
    task_type=TaskType.CUSTOM_VOICE,
    language="Korean",
    speaker="Vivian",
)

sf.write("output.wav", result.audio, result.sample_rate)
```

---

## 디렉터리 구조

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # 설치 스크립트
├── run.sh                # 실행 스크립트 (터미널)
├── run.command           # 실행 파일 (더블 클릭)
├── pyproject.toml        # 프로젝트 설정
├── requirements-mac.txt  # Mac 의존성 패키지
├── mac/                  # Mac 전용 코드
│   ├── engine.py         # 듀얼 엔진 매니저
│   ├── device_utils.py   # 디바이스 감지
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # 메인 애플리케이션
│   ├── i18n_utils.py     # 국제화 유틸리티
│   ├── components/       # 탭 컴포넌트
│   └── i18n/             # 10개 언어 파일
├── qwen_tts/             # TTS 코어 (업스트림)
└── docs/                 # 다국어 README
```

---

## 문제 해결

| 오류 | 원인 | 해결 방법 |
|------|------|----------|
| `conda not found` | Miniforge 미설치 | `./setup_mac.sh` 실행 |
| `No space left on device` | 디스크 공간 부족 | 10GB 이상의 여유 공간 확보 |
| `RuntimeError: MPS backend` | 미지원 MPS 작업 | `PYTORCH_ENABLE_MPS_FALLBACK=1` 설정 |
| `Out of memory` | 메모리 부족 | 다른 앱을 종료하거나 양자화 모델 사용 |

---

## 감사의 말

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## 라이선스

[Apache License 2.0](../LICENSE)

---

## 기여하기

Issue와 Pull Request를 환영합니다!
