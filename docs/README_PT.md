<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Fork do Qwen3-TTS totalmente otimizado para Apple Silicon Mac<br>
  Motor duplo (MLX + PyTorch) para uma experiência TTS nativa no Mac
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_RU.md">Русский</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_IT.md">Italiano</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_FR.md">Français</a> |
  **Português**
</p>

<p align="center">
  <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/stargazers">
    <img src="../assets/star.gif" alt="Star this repo!" width="580">
  </a>
  <br>
  <sub>Se este projeto foi útil para você, considere dar uma estrela — ajuda muito!</sub>
</p>

---

## Por que Qwen3-TTS-Mac-GeneLab?

| Recurso | Qwen3-TTS Oficial | **Este Projeto** |
|---------|-------------------|------------------|
| Otimização Apple Silicon | Limitada | **Suporte completo** |
| Inferência nativa MLX | Não | **Sim** (quantização 8bit/4bit) |
| PyTorch MPS | Configuração manual necessária | **Troca automática** |
| Interface gráfica | Nenhuma | **Web UI em 10 idiomas** |
| Clonagem de voz | Apenas CLI | **Web UI + transcrição automática Whisper** |
| Gerenciamento de memória | Nenhum | **Otimizado para Unified Memory** |
| Instalação | Complexa | **Um único comando** |

### Inovações principais

1. **Arquitetura de motor duplo**
   - MLX: Nativo para Apple Silicon, quantização 8bit/4bit para velocidade e eficiência de memória
   - PyTorch: Troca automática para Voice Clone (execução em CPU float32)

2. **Otimização automática baseada em tarefas**
   - CustomVoice -> MLX preferido (rápido)
   - VoiceDesign -> MLX preferido (rápido)
   - VoiceClone -> PyTorch CPU (requer float32)

3. **Web UI em 10 idiomas**
   - Interface intuitiva baseada em Gradio
   - Troque o idioma no menu suspenso no topo

---

## Requisitos do sistema

| Item | Mínimo | Recomendado |
|------|--------|-------------|
| Chip | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| SO | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Armazenamento livre | 10GB | 20GB+ |

> **Procurando a versão Windows?** Confira o [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — versão nativa para Windows com suporte a GPU NVIDIA (testado com RTX 5090).

---

## Início rápido

### 1. Clonar o repositório

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Configuração (apenas na primeira vez, ~5-10 min)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Iniciar a Web UI

**Opção A: Duplo clique (recomendado)**

Dê um duplo clique em `run.command` no Finder para iniciar automaticamente no Terminal.

**Opção B: Pelo terminal**

```bash
./run.sh
```

> Se a porta já estiver em uso, uma porta disponível é detectada automaticamente.

### 4. Abrir no navegador

Abra http://localhost:7860 (verifique a saída do terminal se a porta foi alterada)

---

## Funcionalidades da Web UI

### Custom Voice
Gere fala com 9 locutores predefinidos. Suporta controle de emoções e 10 idiomas.

### Voice Design
Descreva as características da voz em texto para gerar uma fala correspondente.

### Voice Clone
Clone uma voz a partir de apenas 3 segundos de áudio de referência com transcrição automática Whisper.

> **Nota**: Voice Clone requer o **modelo Base** (~3.8GB), baixado automaticamente no primeiro uso.

### Settings
Seleção de motor (AUTO/MLX/PyTorch), monitor de memória, gerenciamento de modelos.

---

## Uso via CLI

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

## Estrutura de diretórios

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Script de configuração
├── run.sh                # Script de inicialização (terminal)
├── run.command           # Arquivo de inicialização (duplo clique)
├── pyproject.toml        # Configuração do projeto
├── requirements-mac.txt  # Dependências para Mac
├── mac/                  # Código específico para Mac
│   ├── engine.py         # Gerenciador de motor duplo
│   ├── device_utils.py   # Detecção de dispositivo
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Aplicação principal
│   ├── i18n_utils.py     # Utilitário i18n
│   ├── components/       # Componentes das abas
│   └── i18n/             # 10 arquivos de idioma
├── qwen_tts/             # Núcleo TTS (upstream)
└── docs/                 # README multilíngue
```

---

## Solução de problemas

| Erro | Causa | Solução |
|------|-------|---------|
| `conda not found` | Miniforge não instalado | Execute `./setup_mac.sh` |
| `No space left on device` | Espaço em disco insuficiente | Certifique-se de ter 10GB+ livres |
| `RuntimeError: MPS backend` | Operação MPS não suportada | Defina `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Memória insuficiente | Feche outros aplicativos ou use modelos quantizados |

---

## Agradecimentos

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Licença

[Apache License 2.0](../LICENSE)

---

## Contribuir

Issues e Pull Requests são bem-vindos!
