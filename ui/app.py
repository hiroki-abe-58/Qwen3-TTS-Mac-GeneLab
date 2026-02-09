# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS-Mac-JP Gradio Web UI

メインアプリケーション。4つのタブ構成:
1. カスタムボイス - 9種のプリセットスピーカー
2. ボイスデザイン - テキスト記述でボイス生成
3. ボイスクローン - 参照音声でクローン
4. 設定 - エンジン選択、メモリモニタ
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import gradio as gr

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# i18n ロード
UI_DIR = Path(__file__).parent
I18N_DIR = UI_DIR / "i18n"


def load_i18n(lang: str = "ja") -> dict[str, Any]:
    """i18n ファイルをロードする。

    Args:
        lang: 言語コード (ja, en)

    Returns:
        翻訳辞書
    """
    i18n_file = I18N_DIR / f"{lang}.json"
    if not i18n_file.exists():
        logger.warning(f"i18n ファイルが見つかりません: {i18n_file}")
        return {}

    with open(i18n_file, "r", encoding="utf-8") as f:
        return json.load(f)


# グローバル翻訳辞書
_i18n: dict[str, Any] = {}


def t(key: str, default: str | None = None) -> str:
    """翻訳キーから文字列を取得する。

    Args:
        key: ドット区切りのキー (例: "tabs.custom_voice")
        default: デフォルト値

    Returns:
        翻訳された文字列
    """
    keys = key.split(".")
    value: Any = _i18n
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default if default is not None else key
    return str(value) if not isinstance(value, dict) else (default or key)


# カスタム CSS
CUSTOM_CSS = """
/* Qwen3-TTS-Mac-JP カスタムテーマ */
:root {
    --primary-color: #4A90D9;
    --primary-hover: #357ABD;
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-tertiary: #0f3460;
    --text-primary: #eaeaea;
    --text-secondary: #a0a0a0;
    --border-color: #2a2a4e;
    --success-color: #4ade80;
    --warning-color: #fbbf24;
    --error-color: #f87171;
}

/* ダークモード対応 */
.dark {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --text-primary: #eaeaea;
}

/* ヘッダー */
.header-container {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
}

.header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* タブスタイル */
.tab-nav button {
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
}

.tab-nav button.selected {
    background: var(--primary-color) !important;
    color: white !important;
}

/* プライマリボタン */
.primary-btn {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(74, 144, 217, 0.4) !important;
}

/* ステータスバー */
.status-bar {
    display: flex;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: var(--bg-secondary);
    border-radius: 8px;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success-color);
}

.status-dot.warning {
    background: var(--warning-color);
}

.status-dot.error {
    background: var(--error-color);
}

/* 音声プレーヤー */
audio {
    width: 100%;
    border-radius: 8px;
}

/* スピーカーカード */
.speaker-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    transition: border-color 0.2s;
    cursor: pointer;
}

.speaker-card:hover {
    border-color: var(--primary-color);
}

.speaker-card.selected {
    border-color: var(--primary-color);
    background: rgba(74, 144, 217, 0.1);
}

/* メモリモニター */
.memory-bar {
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
}

.memory-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--success-color), var(--primary-color));
    transition: width 0.3s;
}

/* スクロールバー非表示 */
::-webkit-scrollbar {
    width: 0;
    height: 0;
}
"""


def create_header() -> gr.HTML:
    """ヘッダーを作成する。"""
    return gr.HTML(
        """
        <div class="header-container">
            <h1 class="header-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
                Qwen3-TTS-Mac-GeneLab
            </h1>
            <p class="header-subtitle">Apple Silicon Mac に最適化された音声合成</p>
        </div>
        """
    )


def create_app() -> gr.Blocks:
    """Gradio アプリケーションを作成する。"""
    global _i18n
    _i18n = load_i18n("ja")

    # コンポーネントをインポート
    from ui.components.custom_voice_tab import create_custom_voice_tab
    from ui.components.settings_tab import create_settings_tab
    from ui.components.voice_clone_tab import create_voice_clone_tab
    from ui.components.voice_design_tab import create_voice_design_tab

    with gr.Blocks(
        title="Qwen3-TTS-Mac-GeneLab",
    ) as app:
        # ヘッダー
        create_header()

        # タブ
        with gr.Tabs() as tabs:
            with gr.TabItem(t("tabs.custom_voice", "カスタムボイス"), id="custom_voice"):
                create_custom_voice_tab()

            with gr.TabItem(t("tabs.voice_design", "ボイスデザイン"), id="voice_design"):
                create_voice_design_tab()

            with gr.TabItem(t("tabs.voice_clone", "ボイスクローン"), id="voice_clone"):
                create_voice_clone_tab()

            with gr.TabItem(t("tabs.settings", "設定"), id="settings"):
                create_settings_tab()

        # フッター
        gr.HTML(
            """
            <div style="text-align: center; padding: 1rem; color: var(--text-secondary); font-size: 0.85rem;">
                <p>Powered by <a href="https://github.com/QwenLM/Qwen3-TTS" target="_blank" style="color: var(--primary-color);">Qwen3-TTS</a> | 
                Fork: <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab" target="_blank" style="color: var(--primary-color);">Qwen3-TTS-Mac-GeneLab</a></p>
            </div>
            """
        )

    return app


def main() -> None:
    """メインエントリーポイント。"""
    parser = argparse.ArgumentParser(description="Qwen3-TTS-Mac-GeneLab Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="ホストアドレス")
    parser.add_argument("--port", type=int, default=7860, help="ポート番号")
    parser.add_argument("--share", action="store_true", help="Gradio 共有リンクを生成")
    parser.add_argument("--lang", type=str, default="ja", choices=["ja", "en"], help="UI 言語")

    args = parser.parse_args()

    # 言語設定
    global _i18n
    _i18n = load_i18n(args.lang)

    logger.info(f"Qwen3-TTS-Mac-GeneLab Web UI を起動中...")
    logger.info(f"ホスト: {args.host}, ポート: {args.port}")

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()
