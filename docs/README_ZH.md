<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Apple Silicon Mac 全面优化的 Qwen3-TTS 分支<br>
  双引擎 (MLX + PyTorch) 带来原生 Mac TTS 体验
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <strong>中文</strong> |
  <a href="README_KO.md">한국어</a> |
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
  <sub>如果您觉得这个项目有用，请给个 Star，非常感谢！</sub>
</p>

---

## 为什么选择 Qwen3-TTS-Mac-GeneLab？

| 功能 | 官方 Qwen3-TTS | **本项目** |
|------|----------------|-----------|
| Apple Silicon 优化 | 有限 | **全面支持** |
| MLX 原生推理 | 否 | **支持** (8bit/4bit 量化) |
| PyTorch MPS | 需手动配置 | **自动切换** |
| GUI | 无 | **10 语言 Web UI** |
| Voice Clone | 仅限 CLI | **Web UI + Whisper 自动转录** |
| 内存管理 | 无 | **统一内存优化** |
| 安装配置 | 复杂 | **一条命令** |

### 核心创新

1. **双引擎架构**
   - MLX: Apple Silicon 原生，8bit/4bit 量化，速度快、内存高效
   - PyTorch: Voice Clone 自动切换 (float32 CPU 执行)

2. **基于任务的自动优化**
   - CustomVoice -> 优先使用 MLX (快速)
   - VoiceDesign -> 优先使用 MLX (快速)
   - VoiceClone -> PyTorch CPU (需要 float32)

3. **10 语言 Web UI**
   - 基于 Gradio 的直观界面
   - 通过顶部下拉菜单切换语言

---

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 芯片 | Apple Silicon (M1) | M2 Pro / M3+ |
| 内存 | 16GB | 32GB+ |
| 操作系统 | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| 可用存储空间 | 10GB | 20GB+ |

> **需要 Windows 版本？** 请查看 [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — 支持 NVIDIA GPU 的 Windows 原生版（已测试 RTX 5090）。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. 安装配置 (仅需首次执行，约 5-10 分钟)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. 启动 Web UI

**方式 A: 双击启动 (推荐)**

在 Finder 中双击 `run.command`，将在终端中自动启动。

**方式 B: 从终端启动**

```bash
./run.sh
```

> 如果端口被占用，系统会自动检测可用端口。

### 4. 在浏览器中打开

打开 http://localhost:7860 (如果端口有变更，请查看终端输出)

---

## Web UI 功能

### Custom Voice
使用 9 个预设说话人生成语音。支持情感控制和 10 种语言。

### Voice Design
通过文字描述声音特征来生成匹配的语音。

### Voice Clone
仅需 3 秒参考音频即可克隆声音，支持 Whisper 自动转录。

> **注意**: Voice Clone 需要 **Base 模型** (~3.8GB)，首次使用时会自动下载。

### Settings
引擎选择 (AUTO/MLX/PyTorch)、内存监控、模型管理。

---

## CLI 用法

```python
from mac import DualEngine, TaskType
import soundfile as sf

engine = DualEngine()

result = engine.generate(
    text="你好，这是一个语音合成演示。",
    task_type=TaskType.CUSTOM_VOICE,
    language="Chinese",
    speaker="Vivian",
)

sf.write("output.wav", result.audio, result.sample_rate)
```

---

## 目录结构

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # 安装脚本
├── run.sh                # 启动脚本 (终端)
├── run.command           # 启动文件 (双击)
├── pyproject.toml        # 项目配置
├── requirements-mac.txt  # Mac 依赖
├── mac/                  # Mac 专用代码
│   ├── engine.py         # 双引擎管理器
│   ├── device_utils.py   # 设备检测
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # 主应用
│   ├── i18n_utils.py     # 国际化工具
│   ├── components/       # 标签页组件
│   └── i18n/             # 10 种语言文件
├── qwen_tts/             # TTS 核心 (上游)
└── docs/                 # 多语言 README
```

---

## 故障排除

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `conda not found` | 未安装 Miniforge | 运行 `./setup_mac.sh` |
| `No space left on device` | 磁盘空间不足 | 确保有 10GB 以上的可用空间 |
| `RuntimeError: MPS backend` | 不支持的 MPS 操作 | 设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | 内存不足 | 关闭其他应用或使用量化模型 |

---

## 致谢

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## 许可证

[Apache License 2.0](../LICENSE)

---

## 贡献

欢迎提交 Issue 和 Pull Request！
