<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Apple Silicon Mac fully optimized Qwen3-TTS fork<br>
  Dual engine (MLX + PyTorch) for native Mac TTS experience
</p>

<p align="center">
  English |
  <a href="docs/README_JA.md">日本語</a> |
  <a href="docs/README_ZH.md">中文</a> |
  <a href="docs/README_KO.md">한국어</a> |
  <a href="docs/README_RU.md">Русский</a> |
  <a href="docs/README_ES.md">Español</a> |
  <a href="docs/README_IT.md">Italiano</a> |
  <a href="docs/README_DE.md">Deutsch</a> |
  <a href="docs/README_FR.md">Français</a> |
  <a href="docs/README_PT.md">Português</a>
</p>

<p align="center">
  <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/stargazers">
    <img src="assets/star.gif" alt="Please star this repo!" width="580">
  </a>
  <br>
  <sub>If you find this project useful, please consider giving it a star — it helps a lot!</sub>
</p>

---

## Why Qwen3-TTS-Mac-GeneLab?

| Feature | Official Qwen3-TTS | **This Project** |
|---------|-------------------|------------------|
| Apple Silicon Optimization | Limited | **Full Support** |
| MLX Native Inference | No | **Yes** (8bit/4bit quantization) |
| PyTorch MPS | Manual setup required | **Auto-switch** |
| GUI | None | **10-language Web UI** |
| Voice Clone | CLI only | **Web UI + Whisper auto-transcription** |
| Memory Management | None | **Unified Memory optimized** |
| Setup | Complex | **One command** |

### Key Innovations

1. **Dual Engine Architecture**
   - MLX: Apple Silicon native, 8bit/4bit quantization for speed & memory efficiency
   - PyTorch: Auto-switch for Voice Clone (float32 CPU execution)

2. **Task-based Auto Optimization**
   - CustomVoice -> MLX preferred (fast)
   - VoiceDesign -> MLX preferred (fast)
   - VoiceClone -> PyTorch CPU (float32 required)

3. **10-Language Web UI**
   - Gradio-based intuitive interface
   - Switch language from the dropdown at the top

---

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Chip | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| OS | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Free Storage | 10GB | 20GB+ |

> **Looking for Windows?** Check out [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — Windows native version with NVIDIA GPU support (RTX 5090 tested).

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Setup (first time only, ~5-10 min)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Launch the Web UI

**Option A: Double-click (recommended)**

Double-click `run.command` in Finder to auto-launch in Terminal.

**Option B: From terminal**

```bash
./run.sh
```

> If the port is already in use, an available port is automatically detected.

### 4. Open in Browser

Open http://localhost:7860 (check the terminal output if the port was changed)

---

## Web UI Features

### Custom Voice
Generate speech with 9 preset speakers. Supports emotion control and 10 languages.

### Voice Design
Describe voice characteristics in text to generate matching speech.

### Voice Clone
Clone a voice from just 3 seconds of reference audio with Whisper auto-transcription.

> **Note**: Voice Clone requires the **Base model** (~3.8GB), downloaded automatically on first use.

### Settings
Engine selection (AUTO/MLX/PyTorch), memory monitor, model management.

---

## CLI Usage

```python
from mac import DualEngine, TaskType
import soundfile as sf

engine = DualEngine()

result = engine.generate(
    text="Hello, this is a voice synthesis demo.",
    task_type=TaskType.CUSTOM_VOICE,
    language="English",
    speaker="Vivian",
)

sf.write("output.wav", result.audio, result.sample_rate)
```

---

## Directory Structure

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Setup script
├── run.sh                # Launch script (terminal)
├── run.command           # Launch file (double-click)
├── pyproject.toml        # Project configuration
├── requirements-mac.txt  # Mac dependencies
├── mac/                  # Mac-specific code
│   ├── engine.py         # Dual engine manager
│   ├── device_utils.py   # Device detection
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Main application
│   ├── i18n_utils.py     # i18n utility
│   ├── components/       # Tab components
│   └── i18n/             # 10 language files
├── qwen_tts/             # TTS core (upstream)
└── docs/                 # Multilingual README
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `conda not found` | Miniforge not installed | Run `./setup_mac.sh` |
| `No space left on device` | Insufficient disk space | Ensure 10GB+ free |
| `RuntimeError: MPS backend` | Unsupported MPS operation | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Low memory | Close other apps or use quantized models |

---

## Acknowledgments

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## License

[Apache License 2.0](LICENSE)

---

## Contributing

Issues and Pull Requests are welcome!
