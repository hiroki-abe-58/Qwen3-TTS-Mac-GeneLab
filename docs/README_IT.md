<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Fork di Qwen3-TTS completamente ottimizzato per Apple Silicon Mac<br>
  Doppio motore (MLX + PyTorch) per un'esperienza TTS nativa su Mac
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_RU.md">Русский</a> |
  <a href="README_ES.md">Español</a> |
  **Italiano** |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_PT.md">Português</a>
</p>

<p align="center">
  <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/stargazers">
    <img src="../assets/star.gif" alt="Star this repo!" width="580">
  </a>
  <br>
  <sub>Se trovi utile questo progetto, considera di dargli una stella — aiuta molto!</sub>
</p>

---

## Perché Qwen3-TTS-Mac-GeneLab?

| Funzionalità | Qwen3-TTS Ufficiale | **Questo Progetto** |
|--------------|---------------------|---------------------|
| Ottimizzazione Apple Silicon | Limitata | **Supporto completo** |
| Inferenza nativa MLX | No | **Sì** (quantizzazione 8bit/4bit) |
| PyTorch MPS | Configurazione manuale necessaria | **Cambio automatico** |
| Interfaccia grafica | Nessuna | **Web UI in 10 lingue** |
| Clonazione vocale | Solo CLI | **Web UI + trascrizione automatica Whisper** |
| Gestione della memoria | Nessuna | **Ottimizzata per Unified Memory** |
| Installazione | Complessa | **Un solo comando** |

### Innovazioni chiave

1. **Architettura a doppio motore**
   - MLX: Nativo per Apple Silicon, quantizzazione 8bit/4bit per velocità ed efficienza di memoria
   - PyTorch: Cambio automatico per Voice Clone (esecuzione CPU float32)

2. **Ottimizzazione automatica basata sui task**
   - CustomVoice -> MLX preferito (veloce)
   - VoiceDesign -> MLX preferito (veloce)
   - VoiceClone -> PyTorch CPU (richiede float32)

3. **Web UI in 10 lingue**
   - Interfaccia intuitiva basata su Gradio
   - Cambia lingua dal menu a tendina in alto

---

## Requisiti di sistema

| Elemento | Minimo | Consigliato |
|----------|--------|-------------|
| Chip | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| SO | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Spazio libero | 10GB | 20GB+ |

> **Cerchi la versione Windows?** Dai un'occhiata a [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — versione nativa per Windows con supporto GPU NVIDIA (testato con RTX 5090).

---

## Avvio rapido

### 1. Clona il repository

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Configurazione (solo la prima volta, ~5-10 min)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Avvia la Web UI

**Opzione A: Doppio clic (consigliato)**

Fai doppio clic su `run.command` nel Finder per avviare automaticamente nel Terminal.

**Opzione B: Dal terminale**

```bash
./run.sh
```

> Se la porta è già in uso, viene automaticamente rilevata una porta disponibile.

### 4. Apri nel browser

Apri http://localhost:7860 (controlla l'output del terminale se la porta è cambiata)

---

## Funzionalità della Web UI

### Custom Voice
Genera il parlato con 9 speaker preimpostati. Supporta il controllo delle emozioni e 10 lingue.

### Voice Design
Descrivi le caratteristiche della voce in testo per generare un parlato corrispondente.

### Voice Clone
Clona una voce da soli 3 secondi di audio di riferimento con trascrizione automatica Whisper.

> **Nota**: Voice Clone richiede il **modello Base** (~3.8GB), scaricato automaticamente al primo utilizzo.

### Settings
Selezione del motore (AUTO/MLX/PyTorch), monitoraggio della memoria, gestione dei modelli.

---

## Utilizzo da CLI

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

## Struttura delle directory

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Script di configurazione
├── run.sh                # Script di avvio (terminale)
├── run.command           # File di avvio (doppio clic)
├── pyproject.toml        # Configurazione del progetto
├── requirements-mac.txt  # Dipendenze per Mac
├── mac/                  # Codice specifico per Mac
│   ├── engine.py         # Gestore doppio motore
│   ├── device_utils.py   # Rilevamento dispositivo
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Applicazione principale
│   ├── i18n_utils.py     # Utilità i18n
│   ├── components/       # Componenti delle schede
│   └── i18n/             # 10 file di lingua
├── qwen_tts/             # Core TTS (upstream)
└── docs/                 # README multilingue
```

---

## Risoluzione dei problemi

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| `conda not found` | Miniforge non installato | Esegui `./setup_mac.sh` |
| `No space left on device` | Spazio su disco insufficiente | Assicurati di avere 10GB+ liberi |
| `RuntimeError: MPS backend` | Operazione MPS non supportata | Imposta `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Memoria insufficiente | Chiudi altre applicazioni o usa modelli quantizzati |

---

## Ringraziamenti

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Licenza

[Apache License 2.0](../LICENSE)

---

## Contribuire

Le Issue e le Pull Request sono benvenute!
