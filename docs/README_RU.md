<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Полностью оптимизированный форк Qwen3-TTS для Apple Silicon Mac<br>
  Двойной движок (MLX + PyTorch) для нативного TTS на Mac
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <a href="README_KO.md">한국어</a> |
  <strong>Русский</strong> |
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
  <sub>Если проект оказался полезным, поставьте звезду — это очень помогает!</sub>
</p>

---

## Почему Qwen3-TTS-Mac-GeneLab?

| Возможность | Официальный Qwen3-TTS | **Этот проект** |
|-------------|----------------------|-----------------|
| Оптимизация Apple Silicon | Ограниченная | **Полная поддержка** |
| Нативный вывод MLX | Нет | **Да** (квантизация 8bit/4bit) |
| PyTorch MPS | Требуется ручная настройка | **Автопереключение** |
| GUI | Нет | **Web UI на 10 языках** |
| Voice Clone | Только CLI | **Web UI + автотранскрипция Whisper** |
| Управление памятью | Нет | **Оптимизация единой памяти** |
| Установка | Сложная | **Одна команда** |

### Ключевые инновации

1. **Архитектура двойного движка**
   - MLX: нативный для Apple Silicon, квантизация 8bit/4bit для скорости и эффективности памяти
   - PyTorch: автопереключение для Voice Clone (выполнение float32 на CPU)

2. **Автооптимизация на основе задач**
   - CustomVoice -> предпочтительно MLX (быстро)
   - VoiceDesign -> предпочтительно MLX (быстро)
   - VoiceClone -> PyTorch CPU (требуется float32)

3. **Web UI на 10 языках**
   - Интуитивный интерфейс на базе Gradio
   - Переключение языка через выпадающее меню вверху

---

## Системные требования

| Параметр | Минимум | Рекомендуется |
|----------|---------|---------------|
| Чип | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16 ГБ | 32 ГБ+ |
| ОС | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Свободное место | 10 ГБ | 20 ГБ+ |

> **Ищете версию для Windows?** Смотрите [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — нативная версия для Windows с поддержкой NVIDIA GPU (протестировано на RTX 5090).

---

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Установка (только первый раз, ~5-10 мин)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Запуск Web UI

**Способ A: Двойной клик (рекомендуется)**

Дважды щёлкните по `run.command` в Finder — терминал запустится автоматически.

**Способ B: Из терминала**

```bash
./run.sh
```

> Если порт уже занят, свободный порт будет определён автоматически.

### 4. Открыть в браузере

Откройте http://localhost:7860 (если порт был изменён, проверьте вывод терминала)

---

## Возможности Web UI

### Custom Voice
Генерация речи с 9 предустановленными голосами. Поддержка управления эмоциями и 10 языков.

### Voice Design
Опишите характеристики голоса текстом — и получите подходящую речь.

### Voice Clone
Клонирование голоса всего по 3 секундам эталонного аудио с автотранскрипцией Whisper.

> **Примечание**: для Voice Clone необходима **Base-модель** (~3.8 ГБ), которая автоматически загружается при первом использовании.

### Settings
Выбор движка (AUTO/MLX/PyTorch), мониторинг памяти, управление моделями.

---

## Использование CLI

```python
from mac import DualEngine, TaskType
import soundfile as sf

engine = DualEngine()

result = engine.generate(
    text="Привет, это демонстрация синтеза речи.",
    task_type=TaskType.CUSTOM_VOICE,
    language="Russian",
    speaker="Vivian",
)

sf.write("output.wav", result.audio, result.sample_rate)
```

---

## Структура каталогов

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Скрипт установки
├── run.sh                # Скрипт запуска (терминал)
├── run.command           # Файл запуска (двойной клик)
├── pyproject.toml        # Конфигурация проекта
├── requirements-mac.txt  # Зависимости для Mac
├── mac/                  # Код для Mac
│   ├── engine.py         # Менеджер двойного движка
│   ├── device_utils.py   # Определение устройства
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Главное приложение
│   ├── i18n_utils.py     # Утилита i18n
│   ├── components/       # Компоненты вкладок
│   └── i18n/             # 10 языковых файлов
├── qwen_tts/             # Ядро TTS (upstream)
└── docs/                 # Многоязычные README
```

---

## Устранение неполадок

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `conda not found` | Miniforge не установлен | Запустите `./setup_mac.sh` |
| `No space left on device` | Недостаточно места на диске | Освободите минимум 10 ГБ |
| `RuntimeError: MPS backend` | Неподдерживаемая операция MPS | Установите `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Нехватка памяти | Закройте другие приложения или используйте квантизированные модели |

---

## Благодарности

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Лицензия

[Apache License 2.0](../LICENSE)

---

## Участие в проекте

Мы приветствуем Issues и Pull Requests!
