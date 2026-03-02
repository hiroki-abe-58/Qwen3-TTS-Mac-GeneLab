<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Apple Silicon Mac に完全最適化された Qwen3-TTS フォーク<br>
  デュアルエンジン (MLX + PyTorch) によるネイティブ Mac TTS 体験
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <strong>日本語</strong> |
  <a href="README_ZH.md">中文</a> |
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
  <sub>このプロジェクトが役に立ったら、ぜひスターをお願いします！</sub>
</p>

---

## なぜ Qwen3-TTS-Mac-GeneLab なのか？

| 機能 | 公式 Qwen3-TTS | **本プロジェクト** |
|------|----------------|-------------------|
| Apple Silicon 最適化 | 限定的 | **フルサポート** |
| MLX ネイティブ推論 | なし | **対応** (8bit/4bit 量子化) |
| PyTorch MPS | 手動設定が必要 | **自動切替** |
| GUI | なし | **10言語対応 Web UI** |
| Voice Clone | CLI のみ | **Web UI + Whisper 自動文字起こし** |
| メモリ管理 | なし | **ユニファイドメモリ最適化** |
| セットアップ | 複雑 | **ワンコマンド** |

### 主な革新点

1. **デュアルエンジンアーキテクチャ**
   - MLX: Apple Silicon ネイティブ、8bit/4bit 量子化による高速化とメモリ効率
   - PyTorch: Voice Clone 用の自動切替 (float32 CPU 実行)

2. **タスクベースの自動最適化**
   - CustomVoice -> MLX 優先 (高速)
   - VoiceDesign -> MLX 優先 (高速)
   - VoiceClone -> PyTorch CPU (float32 必須)

3. **10言語対応 Web UI**
   - Gradio ベースの直感的なインターフェース
   - 上部のドロップダウンから言語を切り替え可能

---

## システム要件

| 項目 | 最低要件 | 推奨 |
|------|---------|------|
| チップ | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| OS | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| 空きストレージ | 10GB | 20GB+ |

> **Windows をお探しですか？** [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) をご覧ください — NVIDIA GPU 対応の Windows ネイティブ版（RTX 5090 動作確認済み）。

---

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. セットアップ (初回のみ、約 5〜10 分)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Web UI の起動

**方法 A: ダブルクリック (推奨)**

Finder で `run.command` をダブルクリックすると、ターミナルで自動起動します。

**方法 B: ターミナルから**

```bash
./run.sh
```

> ポートが使用中の場合、利用可能なポートが自動検出されます。

### 4. ブラウザで開く

http://localhost:7860 を開きます (ポートが変更された場合はターミナル出力を確認してください)

---

## Web UI の機能

### Custom Voice
9 種類のプリセットスピーカーで音声を生成します。感情制御と 10 言語に対応しています。

### Voice Design
テキストで声の特徴を記述し、それに合った音声を生成します。

### Voice Clone
わずか 3 秒のリファレンス音声から声をクローンします。Whisper による自動文字起こしに対応しています。

> **注意**: Voice Clone には **Base モデル** (~3.8GB) が必要です。初回使用時に自動ダウンロードされます。

### Settings
エンジン選択 (AUTO/MLX/PyTorch)、メモリモニター、モデル管理。

---

## CLI の使い方

```python
from mac import DualEngine, TaskType
import soundfile as sf

engine = DualEngine()

result = engine.generate(
    text="こんにちは、音声合成のデモです。",
    task_type=TaskType.CUSTOM_VOICE,
    language="Japanese",
    speaker="Vivian",
)

sf.write("output.wav", result.audio, result.sample_rate)
```

---

## ディレクトリ構成

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # セットアップスクリプト
├── run.sh                # 起動スクリプト (ターミナル)
├── run.command           # 起動ファイル (ダブルクリック)
├── pyproject.toml        # プロジェクト設定
├── requirements-mac.txt  # Mac 依存パッケージ
├── mac/                  # Mac 固有コード
│   ├── engine.py         # デュアルエンジンマネージャー
│   ├── device_utils.py   # デバイス検出
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # メインアプリケーション
│   ├── i18n_utils.py     # i18n ユーティリティ
│   ├── components/       # タブコンポーネント
│   └── i18n/             # 10 言語ファイル
├── qwen_tts/             # TTS コア (上流)
└── docs/                 # 多言語 README
```

---

## トラブルシューティング

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `conda not found` | Miniforge 未インストール | `./setup_mac.sh` を実行 |
| `No space left on device` | ディスク容量不足 | 10GB 以上の空きを確保 |
| `RuntimeError: MPS backend` | 未対応の MPS 操作 | `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定 |
| `Out of memory` | メモリ不足 | 他のアプリを終了するか量子化モデルを使用 |

---

## 謝辞

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## ライセンス

[Apache License 2.0](../LICENSE)

---

## コントリビューション

Issue や Pull Request を歓迎します！
