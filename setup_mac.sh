#!/bin/bash
# =============================================================================
# Qwen3-TTS-Mac-GeneLab セットアップスクリプト
# Apple Silicon Mac 専用
# =============================================================================

set -e

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}━━━ $1 ━━━${NC}"
}

# エラーハンドリング
handle_error() {
    log_error "セットアップ中にエラーが発生しました (行 $1)"
    log_error "エラーを修正してから再度実行してください。"
    echo ""
    echo "トラブルシューティング:"
    echo "  1. ネットワーク接続を確認してください"
    echo "  2. ディスク容量を確認してください (最低 10GB 必要)"
    echo "  3. 既存の環境を削除して再試行:"
    echo "     conda env remove -n qwen3-tts-mac-genelab"
    echo "  4. Issue を報告: https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/issues"
    exit 1
}

trap 'handle_error $LINENO' ERR

# ヘッダー表示
clear
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                                                                   ║${NC}"
echo -e "${BOLD}║   ${CYAN}Qwen3-TTS-Mac-GeneLab${NC}${BOLD}                                         ║${NC}"
echo -e "${BOLD}║   ${GREEN}Apple Silicon Mac Optimized TTS${NC}${BOLD}                               ║${NC}"
echo -e "${BOLD}║                                                                   ║${NC}"
echo -e "${BOLD}║   MLX + PyTorch MPS Dual-Engine Architecture                      ║${NC}"
echo -e "${BOLD}║                                                                   ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Step 1: システム要件の確認
# =============================================================================
log_step "Step 1/7: システム要件の確認"

# アーキテクチャ検証
log_info "アーキテクチャを確認中..."
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    log_error "このスクリプトは Apple Silicon Mac (arm64) 専用です。"
    log_error "検出されたアーキテクチャ: $ARCH"
    log_error ""
    log_error "Intel Mac をお使いの場合は、公式リポジトリをご利用ください:"
    log_error "https://github.com/QwenLM/Qwen3-TTS"
    exit 1
fi
log_success "Apple Silicon (arm64) を検出しました"

# macOS バージョン確認
log_info "macOS バージョンを確認中..."
MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [ "$MACOS_MAJOR" -lt 14 ]; then
    log_warning "macOS $MACOS_VERSION を検出"
    log_warning "macOS 14 (Sonoma) 以降を強く推奨します。"
    log_warning "一部の機能が正常に動作しない可能性があります。"
else
    log_success "macOS $MACOS_VERSION を検出しました"
fi

# メモリ確認
log_info "システムメモリを確認中..."
TOTAL_MEM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
if [ "$TOTAL_MEM_GB" -lt 8 ]; then
    log_error "メモリ ${TOTAL_MEM_GB}GB は不足しています。最低 8GB 必要です。"
    exit 1
elif [ "$TOTAL_MEM_GB" -lt 16 ]; then
    log_warning "メモリ ${TOTAL_MEM_GB}GB を検出。16GB 以上を推奨します。"
    log_warning "4bit 量子化モデルのみ使用可能な場合があります。"
else
    log_success "メモリ ${TOTAL_MEM_GB}GB を検出しました"
fi

# ディスク容量確認
log_info "ディスク容量を確認中..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREE_SPACE_GB=$(df -g "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
if [ "$FREE_SPACE_GB" -lt 10 ]; then
    log_error "ディスク空き容量 ${FREE_SPACE_GB}GB は不足しています。"
    log_error "最低 10GB の空き容量が必要です（モデルダウンロード含む）。"
    log_error "不要なファイルを削除してから再試行してください。"
    exit 1
elif [ "$FREE_SPACE_GB" -lt 20 ]; then
    log_warning "ディスク空き容量 ${FREE_SPACE_GB}GB を検出。"
    log_warning "20GB 以上を推奨します（複数モデル使用時）。"
else
    log_success "ディスク空き容量 ${FREE_SPACE_GB}GB を検出しました"
fi

# =============================================================================
# Step 2: Homebrew パッケージ
# =============================================================================
log_step "Step 2/7: Homebrew パッケージのインストール"

# Homebrew 確認・インストール
log_info "Homebrew を確認中..."
if ! command -v brew &> /dev/null; then
    log_warning "Homebrew が見つかりません。インストールします..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
log_success "Homebrew が利用可能です"

# Homebrew パッケージインストール
BREW_PACKAGES="sox ffmpeg portaudio"
for pkg in $BREW_PACKAGES; do
    if ! brew list "$pkg" &> /dev/null; then
        log_info "$pkg をインストール中..."
        brew install "$pkg"
    else
        log_success "$pkg は既にインストール済み"
    fi
done

# =============================================================================
# Step 3: Conda/Miniforge
# =============================================================================
log_step "Step 3/7: Python 環境のセットアップ"

# Miniforge/Conda 確認
log_info "Conda/Miniforge を確認中..."
CONDA_FOUND=false

# conda コマンドの存在確認
if command -v conda &> /dev/null; then
    CONDA_FOUND=true
    CONDA_BASE=$(conda info --base)
    log_success "Conda が見つかりました: $CONDA_BASE"
elif [ -f "$HOME/miniforge3/bin/conda" ]; then
    CONDA_FOUND=true
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    log_success "Miniforge3 が見つかりました"
elif [ -f "$HOME/mambaforge/bin/conda" ]; then
    CONDA_FOUND=true
    eval "$($HOME/mambaforge/bin/conda shell.bash hook)"
    log_success "Mambaforge が見つかりました"
elif [ -f "/opt/homebrew/Caskroom/miniforge/base/bin/conda" ]; then
    CONDA_FOUND=true
    eval "$(/opt/homebrew/Caskroom/miniforge/base/bin/conda shell.bash hook)"
    log_success "Homebrew Miniforge が見つかりました"
fi

if [ "$CONDA_FOUND" = false ]; then
    log_warning "Conda/Miniforge が見つかりません。Miniforge をインストールします..."
    brew install miniforge
    eval "$(/opt/homebrew/Caskroom/miniforge/base/bin/conda shell.bash hook)"
    conda init zsh bash 2>/dev/null || true
    log_success "Miniforge をインストールしました"
fi

# conda を確実にアクティベート
source "$(conda info --base)/etc/profile.d/conda.sh"

# 仮想環境名
ENV_NAME="qwen3-tts-mac-genelab"

# 既存環境の確認
log_info "仮想環境 '$ENV_NAME' を確認中..."
if conda env list | grep -q "^$ENV_NAME "; then
    log_warning "環境 '$ENV_NAME' は既に存在します。"
    echo ""
    read -p "既存の環境を削除して再作成しますか？ (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "環境を削除中..."
        conda env remove -n "$ENV_NAME" -y
    else
        log_info "既存の環境を使用します。"
    fi
fi

# 仮想環境作成
if ! conda env list | grep -q "^$ENV_NAME "; then
    log_info "Python 3.11 仮想環境を作成中..."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

# 仮想環境をアクティベート
log_info "仮想環境をアクティベート中..."
conda activate "$ENV_NAME"
log_success "仮想環境 '$ENV_NAME' をアクティベートしました"

# =============================================================================
# Step 4: コアパッケージのインストール
# =============================================================================
log_step "Step 4/7: コアパッケージのインストール"

# pip アップグレード
log_info "pip をアップグレード中..."
pip install --upgrade pip

# PyTorch (MPS 対応版) インストール
log_info "PyTorch (Apple Silicon MPS 対応) をインストール中..."
pip install torch torchvision torchaudio

# =============================================================================
# Step 5: MLX パッケージのインストール
# =============================================================================
log_step "Step 5/7: MLX パッケージのインストール"

log_info "MLX をインストール中..."
pip install mlx mlx-lm

log_info "mlx-audio をインストール中..."
# mlx-audio 0.4.x は huggingface_hub>=1.0 要求で transformers 4.x と非互換 (Issue #1)
pip install "mlx-audio>=0.3,<0.4" "huggingface-hub<1.0"

# =============================================================================
# Step 6: 追加依存関係のインストール
# =============================================================================
log_step "Step 6/7: 追加依存関係のインストール"

log_info "transformers / accelerate をインストール中..."
# transformers 5.x は API 破壊的変更により同梱 qwen_tts が動作しないため上限固定 (Issue #1)
pip install "transformers>=4.51,<5.0" accelerate

log_info "音声処理ライブラリをインストール中..."
pip install librosa soundfile einops

log_info "UI / ユーティリティをインストール中..."
# gradio 6.6+ はキュー(fn_index)処理が非互換で KeyError を起こすため上限固定 (Issue #1)
pip install psutil "gradio>=6.0,<6.6"

log_info "numpy バージョンを調整中 (MPS 互換性のため)..."
pip install "numpy<2"

log_info "onnxruntime をインストール中..."
pip install onnxruntime

log_info "プロジェクトを開発モードでインストール中..."
pip install -e "$SCRIPT_DIR"

# =============================================================================
# Step 7: 設定と動作確認
# =============================================================================
log_step "Step 7/7: 設定と動作確認"

# 環境変数の設定ファイル作成
log_info "環境変数設定ファイルを作成中..."
cat > "$SCRIPT_DIR/.env" << 'EOF'
# Qwen3-TTS-Mac-GeneLab 環境変数
# このファイルは run.sh によって自動的に読み込まれます

# MPS フォールバック (非対応操作を CPU で実行)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# MPS メモリ上限 (70%)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# トークナイザーの並列化を無効化 (警告回避)
export TOKENIZERS_PARALLELISM=false

# HuggingFace キャッシュディレクトリ
export HF_HOME="${HOME}/.cache/huggingface"
EOF
log_success "環境変数設定ファイルを作成しました"

# 動作確認テスト
log_info "動作確認テストを実行中..."
echo ""

python << 'PYTEST'
import sys

def check_module(name, import_name=None, version_attr="__version__"):
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, "N/A")
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: インポートエラー ({e})")
        return False

print(f"Python: {sys.version.split()[0]}")
print("")
print("パッケージ確認:")

all_ok = True
all_ok &= check_module("PyTorch", "torch")
all_ok &= check_module("Gradio", "gradio")
all_ok &= check_module("Transformers", "transformers")
all_ok &= check_module("MLX", "mlx.core", "_version")
all_ok &= check_module("mlx-audio", "mlx_audio")
all_ok &= check_module("librosa")
all_ok &= check_module("soundfile")

print("")

# MPS 確認
import torch
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders): ✓ 利用可能")
else:
    print("MPS (Metal Performance Shaders): ✗ 利用不可")
    all_ok = False

# MLX Metal 確認
try:
    import mlx.core as mx
    # 簡単な計算をテスト
    x = mx.array([1.0, 2.0, 3.0])
    y = mx.sum(x)
    mx.eval(y)
    print("MLX Metal Backend: ✓ 動作確認OK")
except Exception as e:
    print(f"MLX Metal Backend: ✗ エラー ({e})")
    all_ok = False

print("")

if all_ok:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("動作確認テスト: すべて成功!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
else:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("動作確認テスト: 一部失敗")
    print("上記のエラーを確認してください")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sys.exit(1)
PYTEST

# =============================================================================
# 完了メッセージ
# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║                                                                   ║${NC}"
echo -e "${GREEN}${BOLD}║   セットアップ完了!                                               ║${NC}"
echo -e "${GREEN}${BOLD}║                                                                   ║${NC}"
echo -e "${GREEN}${BOLD}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}次のステップ:${NC}"
echo ""
echo -e "  ${CYAN}1.${NC} Web UI を起動:"
echo -e "     ${YELLOW}./run.sh${NC}"
echo ""
echo -e "  ${CYAN}2.${NC} ブラウザでアクセス:"
echo -e "     ${YELLOW}http://localhost:7860${NC}"
echo ""
echo -e "${BOLD}手動で環境をアクティベートする場合:${NC}"
echo -e "     ${YELLOW}conda activate $ENV_NAME${NC}"
echo ""
echo -e "${BOLD}注意:${NC}"
echo "  - 初回実行時にモデルが自動ダウンロードされます (約 4GB)"
echo "  - Voice Clone には Base モデル (追加 4GB) が必要です"
echo ""
