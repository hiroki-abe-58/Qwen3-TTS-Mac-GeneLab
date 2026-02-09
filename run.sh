#!/bin/bash
# =============================================================================
# Qwen3-TTS-Mac-GeneLab 起動スクリプト
# =============================================================================

set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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

# ヘッダー表示
echo ""
echo -e "${CYAN}${BOLD}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║   Qwen3-TTS-Mac-GeneLab                                           ║${NC}"
echo -e "${CYAN}${BOLD}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 環境変数設定
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false

# .env ファイルがあれば読み込み
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Conda 環境をアクティベート
ENV_NAME="qwen3-tts-mac-genelab"

log_info "Python 環境を確認中..."

# conda の初期化 - 複数のパターンを試行
CONDA_INITIALIZED=false

# パターン 1: Homebrew Miniforge
if [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
    CONDA_INITIALIZED=true
# パターン 2: ユーザーホームの miniforge3
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    CONDA_INITIALIZED=true
# パターン 3: ユーザーホームの mambaforge
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
    CONDA_INITIALIZED=true
# パターン 4: Anaconda
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_INITIALIZED=true
# パターン 5: conda info から取得
elif command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        CONDA_INITIALIZED=true
    fi
fi

if [ "$CONDA_INITIALIZED" = false ]; then
    log_error "conda が見つかりません。"
    echo ""
    echo "解決策:"
    echo "  1. Miniforge をインストール: brew install miniforge"
    echo "  2. セットアップを再実行: ./setup_mac.sh"
    echo ""
    exit 1
fi

# 環境をアクティベート
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    log_error "仮想環境 '$ENV_NAME' が見つかりません。"
    echo ""
    echo "解決策:"
    echo "  1. セットアップを実行してください:"
    echo "     ${YELLOW}./setup_mac.sh${NC}"
    echo ""
    echo "  2. または手動で環境を作成:"
    echo "     conda create -n $ENV_NAME python=3.11"
    echo "     conda activate $ENV_NAME"
    echo "     pip install -e ."
    echo ""
    exit 1
fi

log_success "環境 '$ENV_NAME' をアクティベートしました"

# 引数のパース
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
SHARE="${SHARE:-false}"

show_help() {
    echo ""
    echo "使い方: ./run.sh [オプション]"
    echo ""
    echo "オプション:"
    echo "  --host HOST    バインドするホスト (デフォルト: 0.0.0.0)"
    echo "  --port PORT    使用するポート (デフォルト: 7860)"
    echo "  --share        Gradio 共有リンクを生成"
    echo "  -h, --help     このヘルプを表示"
    echo ""
    echo "環境変数:"
    echo "  HOST           --host と同等"
    echo "  PORT           --port と同等"
    echo "  SHARE          'true' で --share と同等"
    echo ""
    echo "例:"
    echo "  ./run.sh                      # デフォルト設定で起動"
    echo "  ./run.sh --port 8080          # ポート 8080 で起動"
    echo "  ./run.sh --share              # 共有リンクを生成"
    echo "  PORT=8080 ./run.sh            # 環境変数でポート指定"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# ポートが使用中なら自動的に空きポートを探す
ORIGINAL_PORT=$PORT
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    log_warning "ポート $PORT は既に使用中です。次のポートを試します..."
    PORT=$((PORT + 1))
    # 100 ポート試してダメなら諦める
    if [ $((PORT - ORIGINAL_PORT)) -ge 100 ]; then
        log_error "ポート ${ORIGINAL_PORT}〜${PORT} はすべて使用中です。"
        exit 1
    fi
done
if [ "$PORT" != "$ORIGINAL_PORT" ]; then
    log_success "空きポート $PORT を自動選択しました。"
fi

# 設定表示
echo ""
log_info "設定:"
echo "  ホスト:       $HOST"
echo "  ポート:       $PORT"
echo "  共有リンク:   $SHARE"
echo ""

# URL 表示
if [ "$HOST" = "0.0.0.0" ]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "localhost")
    echo -e "${GREEN}${BOLD}アクセス URL:${NC}"
    echo -e "  ローカル:   ${YELLOW}http://localhost:$PORT${NC}"
    echo -e "  ネットワーク: ${YELLOW}http://$LOCAL_IP:$PORT${NC}"
else
    echo -e "${GREEN}${BOLD}アクセス URL:${NC}"
    echo -e "  ${YELLOW}http://$HOST:$PORT${NC}"
fi

if [ "$SHARE" = "true" ]; then
    echo -e "  共有リンク: ${YELLOW}(起動後に表示されます)${NC}"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Ctrl+C で終了${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Web UI を起動
cd "$SCRIPT_DIR"

SHARE_ARG=""
if [ "$SHARE" = "true" ]; then
    SHARE_ARG="--share"
fi

python -m ui.app --host "$HOST" --port "$PORT" $SHARE_ARG
