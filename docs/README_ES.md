<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Fork de Qwen3-TTS totalmente optimizado para Apple Silicon Mac<br>
  Motor dual (MLX + PyTorch) para una experiencia TTS nativa en Mac
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_RU.md">Русский</a> |
  **Español** |
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
  <sub>Si este proyecto te resulta útil, considera darle una estrella — ¡ayuda mucho!</sub>
</p>

---

## ¿Por qué Qwen3-TTS-Mac-GeneLab?

| Característica | Qwen3-TTS Oficial | **Este Proyecto** |
|----------------|-------------------|-------------------|
| Optimización Apple Silicon | Limitada | **Soporte completo** |
| Inferencia nativa MLX | No | **Sí** (cuantización 8bit/4bit) |
| PyTorch MPS | Configuración manual necesaria | **Cambio automático** |
| Interfaz gráfica | Ninguna | **Web UI en 10 idiomas** |
| Clonación de voz | Solo CLI | **Web UI + transcripción automática Whisper** |
| Gestión de memoria | Ninguna | **Optimizado para Unified Memory** |
| Instalación | Compleja | **Un solo comando** |

### Innovaciones clave

1. **Arquitectura de motor dual**
   - MLX: Nativo para Apple Silicon, cuantización 8bit/4bit para velocidad y eficiencia de memoria
   - PyTorch: Cambio automático para Voice Clone (ejecución en CPU float32)

2. **Optimización automática basada en tareas**
   - CustomVoice -> MLX preferido (rápido)
   - VoiceDesign -> MLX preferido (rápido)
   - VoiceClone -> PyTorch CPU (requiere float32)

3. **Web UI en 10 idiomas**
   - Interfaz intuitiva basada en Gradio
   - Cambiar idioma desde el menú desplegable superior

---

## Requisitos del sistema

| Elemento | Mínimo | Recomendado |
|----------|--------|-------------|
| Chip | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| SO | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Almacenamiento libre | 10GB | 20GB+ |

> **¿Buscas la versión para Windows?** Consulta [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — versión nativa para Windows con soporte de GPU NVIDIA (probado con RTX 5090).

---

## Inicio rápido

### 1. Clonar el repositorio

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Configuración (solo la primera vez, ~5-10 min)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Iniciar la Web UI

**Opción A: Doble clic (recomendado)**

Haz doble clic en `run.command` en Finder para iniciar automáticamente en Terminal.

**Opción B: Desde la terminal**

```bash
./run.sh
```

> Si el puerto ya está en uso, se detecta automáticamente un puerto disponible.

### 4. Abrir en el navegador

Abre http://localhost:7860 (consulta la salida de la terminal si el puerto cambió)

---

## Funciones de la Web UI

### Custom Voice
Genera voz con 9 hablantes preestablecidos. Soporta control de emociones y 10 idiomas.

### Voice Design
Describe las características de la voz en texto para generar una voz coincidente.

### Voice Clone
Clona una voz a partir de solo 3 segundos de audio de referencia con transcripción automática Whisper.

> **Nota**: Voice Clone requiere el **modelo Base** (~3.8GB), que se descarga automáticamente en el primer uso.

### Settings
Selección de motor (AUTO/MLX/PyTorch), monitor de memoria, gestión de modelos.

---

## Uso por CLI

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

## Estructura del directorio

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Script de configuración
├── run.sh                # Script de inicio (terminal)
├── run.command           # Archivo de inicio (doble clic)
├── pyproject.toml        # Configuración del proyecto
├── requirements-mac.txt  # Dependencias para Mac
├── mac/                  # Código específico para Mac
│   ├── engine.py         # Gestor de motor dual
│   ├── device_utils.py   # Detección de dispositivo
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Aplicación principal
│   ├── i18n_utils.py     # Utilidad i18n
│   ├── components/       # Componentes de pestañas
│   └── i18n/             # 10 archivos de idioma
├── qwen_tts/             # Núcleo TTS (upstream)
└── docs/                 # README multilingüe
```

---

## Solución de problemas

| Error | Causa | Solución |
|-------|-------|----------|
| `conda not found` | Miniforge no instalado | Ejecuta `./setup_mac.sh` |
| `No space left on device` | Espacio en disco insuficiente | Asegúrate de tener 10GB+ libres |
| `RuntimeError: MPS backend` | Operación MPS no soportada | Configura `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Memoria insuficiente | Cierra otras aplicaciones o usa modelos cuantizados |

---

## Agradecimientos

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Licencia

[Apache License 2.0](../LICENSE)

---

## Contribuir

¡Las Issues y Pull Requests son bienvenidas!
