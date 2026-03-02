<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Vollständig für Apple Silicon Mac optimierter Qwen3-TTS-Fork<br>
  Dual-Engine (MLX + PyTorch) für ein natives Mac-TTS-Erlebnis
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_RU.md">Русский</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_IT.md">Italiano</a> |
  **Deutsch** |
  <a href="README_FR.md">Français</a> |
  <a href="README_PT.md">Português</a>
</p>

<p align="center">
  <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/stargazers">
    <img src="../assets/star.gif" alt="Star this repo!" width="580">
  </a>
  <br>
  <sub>Wenn Ihnen dieses Projekt gefällt, geben Sie ihm bitte einen Stern — das hilft sehr!</sub>
</p>

---

## Warum Qwen3-TTS-Mac-GeneLab?

| Funktion | Offizielles Qwen3-TTS | **Dieses Projekt** |
|----------|----------------------|-------------------|
| Apple Silicon-Optimierung | Eingeschränkt | **Volle Unterstützung** |
| Native MLX-Inferenz | Nein | **Ja** (8bit/4bit-Quantisierung) |
| PyTorch MPS | Manuelle Einrichtung erforderlich | **Automatischer Wechsel** |
| Benutzeroberfläche | Keine | **Web UI in 10 Sprachen** |
| Stimmklonen | Nur CLI | **Web UI + automatische Whisper-Transkription** |
| Speicherverwaltung | Keine | **Für Unified Memory optimiert** |
| Einrichtung | Komplex | **Ein einziger Befehl** |

### Wichtige Innovationen

1. **Dual-Engine-Architektur**
   - MLX: Nativ für Apple Silicon, 8bit/4bit-Quantisierung für Geschwindigkeit und Speichereffizienz
   - PyTorch: Automatischer Wechsel für Voice Clone (float32 CPU-Ausführung)

2. **Aufgabenbasierte automatische Optimierung**
   - CustomVoice -> MLX bevorzugt (schnell)
   - VoiceDesign -> MLX bevorzugt (schnell)
   - VoiceClone -> PyTorch CPU (float32 erforderlich)

3. **Web UI in 10 Sprachen**
   - Intuitive Oberfläche basierend auf Gradio
   - Sprache über das Dropdown-Menü oben wechseln

---

## Systemanforderungen

| Element | Minimum | Empfohlen |
|---------|---------|-----------|
| Chip | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| Betriebssystem | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Freier Speicher | 10GB | 20GB+ |

> **Suchen Sie die Windows-Version?** Schauen Sie sich [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) an — native Windows-Version mit NVIDIA-GPU-Unterstützung (getestet mit RTX 5090).

---

## Schnellstart

### 1. Repository klonen

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Einrichtung (nur beim ersten Mal, ~5-10 Min.)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Web UI starten

**Option A: Doppelklick (empfohlen)**

Doppelklicke auf `run.command` im Finder, um automatisch im Terminal zu starten.

**Option B: Über das Terminal**

```bash
./run.sh
```

> Wenn der Port bereits belegt ist, wird automatisch ein verfügbarer Port erkannt.

### 4. Im Browser öffnen

Öffne http://localhost:7860 (überprüfe die Terminal-Ausgabe, falls der Port geändert wurde)

---

## Web UI-Funktionen

### Custom Voice
Sprachgenerierung mit 9 voreingestellten Sprechern. Unterstützt Emotionssteuerung und 10 Sprachen.

### Voice Design
Beschreibe Stimmeigenschaften als Text, um eine passende Sprachausgabe zu erzeugen.

### Voice Clone
Klone eine Stimme aus nur 3 Sekunden Referenz-Audio mit automatischer Whisper-Transkription.

> **Hinweis**: Voice Clone benötigt das **Base-Modell** (~3.8GB), das beim ersten Gebrauch automatisch heruntergeladen wird.

### Settings
Motorauswahl (AUTO/MLX/PyTorch), Speicherüberwachung, Modellverwaltung.

---

## CLI-Nutzung

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

## Verzeichnisstruktur

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Einrichtungsskript
├── run.sh                # Startskript (Terminal)
├── run.command           # Startdatei (Doppelklick)
├── pyproject.toml        # Projektkonfiguration
├── requirements-mac.txt  # Mac-Abhängigkeiten
├── mac/                  # Mac-spezifischer Code
│   ├── engine.py         # Dual-Engine-Manager
│   ├── device_utils.py   # Geräteerkennung
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Hauptanwendung
│   ├── i18n_utils.py     # i18n-Dienstprogramm
│   ├── components/       # Tab-Komponenten
│   └── i18n/             # 10 Sprachdateien
├── qwen_tts/             # TTS-Kern (Upstream)
└── docs/                 # Mehrsprachige README
```

---

## Fehlerbehebung

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| `conda not found` | Miniforge nicht installiert | Führe `./setup_mac.sh` aus |
| `No space left on device` | Unzureichender Speicherplatz | Stelle sicher, dass 10GB+ frei sind |
| `RuntimeError: MPS backend` | Nicht unterstützte MPS-Operation | Setze `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Zu wenig Arbeitsspeicher | Schließe andere Apps oder verwende quantisierte Modelle |

---

## Danksagungen

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Lizenz

[Apache License 2.0](../LICENSE)

---

## Mitwirken

Issues und Pull Requests sind willkommen!
