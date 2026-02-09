<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  <strong>世界初</strong>: Apple Silicon Mac に完全最適化された Qwen3-TTS フォーク<br>
  MLX + PyTorch MPS デュアルエンジンによる、Mac ネイティブ TTS 体験
</p>

<p align="center">
  <a href="README_EN.md">English</a> | 日本語
</p>

<p align="center">
  <img src="assets/UI-customVoice.png" alt="Custom Voice UI" width="80%">
</p>

---

## なぜ Qwen3-TTS-Mac-GeneLab か？

| 機能 | 公式 Qwen3-TTS | **本プロジェクト** |
|------|----------------|-------------------|
| Apple Silicon 最適化 | 限定的 | **フルサポート** |
| MLX ネイティブ推論 | 非対応 | **対応** (8bit/4bit 量子化) |
| PyTorch MPS | 手動設定が必要 | **自動切替** |
| GUI | なし | **日本語対応 Web UI** |
| Voice Clone | コマンドラインのみ | **Web UI + Whisper 自動書き起こし** |
| メモリ管理 | なし | **Unified Memory 最適化** |
| セットアップ | 複雑 | **ワンコマンド** |

### 主な革新点

1. **デュアルエンジンアーキテクチャ**
   - MLX: Apple Silicon ネイティブ、8bit/4bit 量子化で高速・省メモリ
   - PyTorch: Voice Clone 等の float32 必須タスクに自動切替 (CPU 実行)

2. **タスクに応じた自動最適化**
   - CustomVoice → MLX 優先 (高速)
   - VoiceDesign → MLX 優先 (高速)
   - VoiceClone → PyTorch CPU (float32 精度必須、MPS 非互換回避)

3. **完全日本語対応 Web UI**
   - Gradio ベースの直感的な UI
   - 日本語/英語の切り替え可能

---

## 動作環境

| 項目 | 最小要件 | 推奨 |
|------|----------|------|
| チップ | Apple Silicon (M1) | M2 Pro / M3 以上 |
| RAM | 16GB | 32GB 以上 |
| OS | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| 空きストレージ | 10GB | 20GB 以上 |

> **注意**: M1/M2 の 8GB モデルでも 4bit 量子化モデルで動作しますが、品質と速度が低下します。

---

## クイックスタート

### 1. リポジトリをクローン

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. セットアップ（初回のみ、約5〜10分）

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

セットアップスクリプトが自動で以下を実行します:
- Homebrew 依存関係のインストール (sox, ffmpeg, portaudio)
- Miniforge conda 環境の作成 (Python 3.11)
- MLX, PyTorch MPS, Gradio 等のインストール
- 環境変数の設定
- 動作確認テスト

### 3. Web UI を起動

**方法 A: ダブルクリック（推奨）**

Finder で `run.command` をダブルクリックすると Terminal が開いて自動起動します。

**方法 B: ターミナルから**

```bash
./run.sh
```

> ポートが既に使用中の場合は、自動的に空きポートを検出して起動します。

### 4. ブラウザでアクセス

http://localhost:7860 を開く（ポートが変更された場合はターミナルに表示される URL を確認）

---

## Web UI の機能

### カスタムボイス (Custom Voice)

<p align="center">
  <img src="assets/UI-customVoice.png" alt="Custom Voice Tab" width="700">
</p>

9種類のプリセットスピーカーから選択して音声を生成。感情指示も可能。

| スピーカー | 特徴 | 言語 |
|-----------|------|------|
| Chelsie | 明るい若い女性 | 英語/中国語 |
| Ethan | 穏やかな若い男性 | 英語/中国語 |
| Aiden | 渋みのある男性 | 英語/中国語 |
| Bella | 温かみのある女性 | 英語/中国語 |
| Vivian | エネルギッシュな女性 | 英語/中国語 |
| Lucas | 明るい若い男性 | 英語/中国語 |
| Eleanor | 上品な女性 | 英語/中国語 |
| Alexander | 力強い男性 | 英語/中国語 |
| Serena | 癒やしの女性 | 英語/中国語 |

**日本語テキストも入力可能**ですが、発音はネイティブではない場合があります。

### ボイスデザイン (Voice Design)

<p align="center">
  <img src="assets/UI-voiceDesign.png" alt="Voice Design Tab" width="700">
</p>

テキストで声の特徴を説明し、その特徴に合った声を生成。

```
例: "A calm middle-aged male voice with a warm, reassuring tone."
例: "高めのピッチで、元気いっぱいの若い女性の声"
```

### ボイスクローン (Voice Clone)

<p align="center">
  <img src="assets/UI-voiceClone.png" alt="Voice Clone Tab" width="700">
</p>

わずか **3秒** の参照音声から、その声で新しいテキストを読み上げ。

- **Whisper 自動書き起こし**: 参照音声のテキストを自動認識
- **ICL モード**: 参照テキスト込みで高品質クローン
- **X-Vector モード**: 声質のみ抽出（テキスト不要）

> **注意**: Voice Clone は **Base モデル** (約 3.8GB) が必要です。初回実行時に自動ダウンロードされます。

### 設定

<p align="center">
  <img src="assets/UI-settings.png" alt="Settings Tab" width="700">
</p>

- エンジン選択 (AUTO / MLX / PyTorch MPS)
- メモリモニター
- モデル管理

---

## MLX vs PyTorch MPS

| 項目 | MLX (デフォルト) | PyTorch (CPU/MPS) |
|------|-----------------|-------------|
| 推論速度 | **高速** | 中速 |
| メモリ効率 | **優秀** (量子化対応) | 普通 |
| Voice Clone | 対応 | **float32 CPU で安定** |
| 量子化 | **4bit/8bit** | 非対応 |
| 精度 | やや低い場合あり | **高い** |

**自動切替ロジック:**
- CustomVoice/VoiceDesign → MLX を優先
- VoiceClone → PyTorch CPU を使用 (float32 必須、MPS の Placeholder storage 問題回避)

---

## CLI での使用

```python
from mac import DualEngine, TaskType

# エンジン初期化
engine = DualEngine()

# CustomVoice で生成
result = engine.generate(
    text="こんにちは、今日はいい天気ですね。",
    task_type=TaskType.CUSTOM_VOICE,
    language="Japanese",
    speaker="Vivian",
)

# 音声を保存
import soundfile as sf
sf.write("output.wav", result.audio, result.sample_rate)
print(f"生成時間: {result.generation_time:.2f}秒")
print(f"エンジン: {result.engine_used}")
```

### Voice Clone の例

```python
import librosa
from mac import DualEngine, TaskType

engine = DualEngine()

# 参照音声を読み込み
ref_audio, ref_sr = librosa.load("reference.wav", sr=None)

# Voice Clone で生成
result = engine.generate(
    text="この声で新しいテキストを読み上げます。",
    task_type=TaskType.VOICE_CLONE,
    language="Japanese",
    reference_audio=ref_audio,
    reference_text="参照音声のテキスト内容",
    reference_sr=ref_sr,
)

sf.write("cloned_output.wav", result.audio, result.sample_rate)
```

---

## ディレクトリ構造

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # セットアップスクリプト
├── run.sh                # 起動スクリプト（ターミナル用）
├── run.command           # 起動ファイル（ダブルクリック用）
├── pyproject.toml        # プロジェクト設定
├── requirements-mac.txt  # Mac 用依存関係
│
├── mac/                  # Mac 固有のコード
│   ├── __init__.py
│   ├── engine.py         # デュアルエンジン管理
│   ├── device_utils.py   # デバイス検出・dtype 選択
│   ├── memory_manager.py # Unified Memory 管理
│   ├── whisper_transcriber.py  # Whisper 書き起こし
│   └── benchmark.py      # パフォーマンス計測
│
├── ui/                   # Gradio Web UI
│   ├── app.py            # メインアプリ
│   ├── components/       # タブコンポーネント
│   │   ├── custom_voice_tab.py
│   │   ├── voice_design_tab.py
│   │   ├── voice_clone_tab.py
│   │   └── settings_tab.py
│   └── i18n/             # 多言語対応
│       ├── ja.json
│       └── en.json
│
├── qwen_tts/             # 元の TTS コア (upstream)
│   ├── core/
│   │   ├── models/       # モデル定義
│   │   └── tokenizer_*/  # トークナイザー
│   └── inference/        # 推論ラッパー
│
└── examples/             # サンプルコード
    └── mac_quickstart.py
```

---

## トラブルシューティング

### インストール時の問題

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `zsh: command not found: conda` | Miniforge 未インストール | セットアップスクリプトを再実行 |
| `brew: command not found` | Homebrew 未インストール | [Homebrew](https://brew.sh) をインストール |
| `No space left on device` | ディスク容量不足 | 10GB 以上の空きを確保 |

### 実行時の問題

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `仮想環境が見つかりません` | conda 環境がアクティブでない | `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate qwen3-tts-mac-genelab` |
| `RuntimeError: MPS backend` | MPS 未対応の操作 | `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定（セットアップ済みなら不要） |
| `Out of memory` | メモリ不足 | 他のアプリを閉じるか、量子化モデルを使用 |
| `probability tensor contains inf` | Voice Clone で float16 使用 | PyTorch (float32 CPU) が自動選択されているか確認 |

### Voice Clone の問題

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `'default'` エラー | transformers バージョン非互換 | 本フォークでは互換パッチ適用済み。`pip install -e .` を再実行 |
| 参照音声の認識失敗 | 音声が短すぎる/ノイズが多い | 3秒以上のクリアな音声を使用 |
| 生成音声の品質が低い | 参照テキストが不正確 | Whisper の書き起こし結果を確認・修正 |

---

## メモリ使用量の目安

| モデル | dtype | VRAM 使用量 | 推奨 RAM |
|--------|-------|------------|----------|
| 1.7B CustomVoice | bf16 | ~3.4 GB | 16GB |
| 1.7B CustomVoice | 8bit | ~1.7 GB | 16GB |
| 1.7B CustomVoice | 4bit | ~0.9 GB | 8GB |
| 1.7B Base (Voice Clone) | float32 | ~6.8 GB | 32GB |
| 0.6B | bf16 | ~1.2 GB | 8GB |

> **Tips**: 複数モデルを切り替える場合、前のモデルは自動的にアンロードされます。

---

## 環境変数

セットアップスクリプトが `.env` ファイルに自動設定します:

```bash
# MPS フォールバック (非対応操作を CPU で実行)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# MPS メモリ上限 (0.0=無制限、PyTorch 2.10+ 互換)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# トークナイザーの並列化を無効化 (警告回避)
export TOKENIZERS_PARALLELISM=false
```

---

## アップストリームからの更新

元の Qwen3-TTS リポジトリの更新を取り込むには:

```bash
# upstream を追加 (初回のみ)
git remote add upstream https://github.com/QwenLM/Qwen3-TTS.git

# 更新を取得・マージ
git fetch upstream
git merge upstream/main --allow-unrelated-histories
```

---

## 謝辞

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Alibaba Qwen チームによる元リポジトリ
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) — Apple MLX 音声ライブラリ
- [mlx-community](https://huggingface.co/mlx-community) — 量子化済み MLX モデル
- [OpenAI Whisper](https://github.com/openai/whisper) — 音声認識モデル

---

## ライセンス

[Apache License 2.0](LICENSE) (元リポジトリと同一)

---

## コントリビューション

Issue や Pull Request を歓迎します！

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. Pull Request を作成

---

<p align="center">
  Made with ❤️ for the Apple Silicon community
</p>
