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
import logging
import sys
from pathlib import Path
from typing import Any

import gradio as gr

from ui.i18n_utils import UI_LANGUAGES, load_i18n, t

# ページ全体に常駐する JS: アップロードエリアのテキストを英語に固定する。
# gr.HTML 内の <script> は Gradio がストリップするため、
# gr.Blocks(js=...) で注入してページ読込時に1度だけ評価させる。
_UPLOAD_FIX_JS = """
() => {
  /* ---- 非英語パターン検出用正規表現 ---- */
  var RE_DROP  = /ドロップ|拖放|드롭|Перетащите|Arrastra|Trascina|ziehen|[Dd][eé]posez|Arraste/;
  var RE_CLICK = /アップロード|上传|업로드|загруз|[Ss]ubir|[Cc]aricare|Hochladen|[Tt][eé]l[eé]charger|[Ee]nviar/;
  var RE_OR    = /^[\\s\\-\\u2013\\u2014]*(または|或|또는|или|oder|ou)([\\s\\-\\u2013\\u2014]*$)/;

  function fix() {
    document.querySelectorAll('span').forEach(function(s) {
      var t = s.textContent;
      if (!t || t.length > 120 || t.length < 1) return;

      var hasDrop  = RE_DROP.test(t);
      var hasClick = RE_CLICK.test(t);

      if (hasDrop && hasClick) {
        s.textContent = 'Drop Audio Here - or - Click to Upload';
      } else if (hasDrop) {
        s.textContent = 'Drop Audio Here';
      } else if (hasClick) {
        s.textContent = 'Click to Upload';
      } else if (RE_OR.test(t.trim())) {
        s.textContent = '- or -';
      }
    });
  }

  /* MutationObserver: @gr.render による再描画にも追従 (切断しない) */
  new MutationObserver(function() { requestAnimationFrame(fix); })
    .observe(document.body, {childList: true, subtree: true});

  /* 初回実行 + 定期フォールバック */
  setTimeout(fix, 300);
  setInterval(fix, 3000);
}
"""

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_major_minor(version: str) -> tuple[int, int]:
    """'4.57.6' のようなバージョン文字列から (major, minor) を取り出す。"""
    parts = version.split(".")
    try:
        major = int(parts[0])
    except (IndexError, ValueError):
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        minor = 0
    return major, minor


def check_dependency_versions() -> None:
    """非互換な依存バージョンを検知して明確な復旧手順を提示する (Issue #1)。

    - transformers 5.x: check_model_inputs() のシグネチャ変更で
      PyTorch モデルロードが全滅する。
    - gradio 6.6+: キュー (fn_index) 処理が非互換で KeyError を起こす。
    """
    problems: list[str] = []

    try:
        import transformers

        tf_major, _ = _parse_major_minor(transformers.__version__)
        if tf_major >= 5:
            problems.append(
                f"transformers {transformers.__version__} は非対応です "
                "(5.x は API 破壊的変更により Voice Clone 等が失敗します)。"
                "対応: >=4.51,<5.0"
            )
    except ImportError:
        pass

    try:
        gr_major, gr_minor = _parse_major_minor(gr.__version__)
        if (gr_major, gr_minor) >= (6, 6):
            problems.append(
                f"gradio {gr.__version__} は非対応です "
                "(6.6+ はキュー処理が非互換で KeyError を起こします)。"
                "対応: >=6.0,<6.6"
            )
    except Exception:
        pass

    if problems:
        logger.error("=" * 70)
        logger.error("依存パッケージのバージョンが非互換です (Issue #1)")
        for p in problems:
            logger.error("  - %s", p)
        logger.error("復旧コマンド:")
        logger.error(
            "  pip install 'transformers>=4.51,<5.0' 'gradio>=6.0,<6.6'"
        )
        logger.error("=" * 70)


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

/* 言語セレクター */
.lang-selector {
    max-width: 160px;
    margin-left: auto;
}
"""


def create_header() -> gr.HTML:
    """ヘッダーを作成する。"""
    return gr.HTML(
        f"""
        <div class="header-container">
            <h1 class="header-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
                {t("app.title")}
            </h1>
            <p class="header-subtitle">{t("app.subtitle")}</p>
        </div>
        """
    )


def create_app(default_lang: str = "ja") -> gr.Blocks:
    """Gradio アプリケーションを作成する。"""
    load_i18n(default_lang)

    # コンポーネントをインポート
    from ui.components.custom_voice_tab import create_custom_voice_tab
    from ui.components.settings_tab import create_settings_tab
    from ui.components.voice_clone_tab import create_voice_clone_tab
    from ui.components.voice_design_tab import create_voice_design_tab

    with gr.Blocks(
        title="Qwen3-TTS-Mac-GeneLab",
    ) as app:
        # 言語セレクター（最上部）
        with gr.Row():
            gr.HTML("<div style='flex:1'></div>")
            lang_selector = gr.Dropdown(
                choices=UI_LANGUAGES,
                value=default_lang,
                label="",
                container=False,
                scale=0,
                min_width=160,
                elem_classes=["lang-selector"],
            )

        # 言語変更時に動的再描画
        @gr.render(inputs=[lang_selector])
        def render_content(selected_lang: str) -> None:
            load_i18n(selected_lang)

            create_header()

            with gr.Tabs():
                with gr.TabItem(t("tabs.custom_voice"), id="custom_voice"):
                    create_custom_voice_tab()

                with gr.TabItem(t("tabs.voice_design"), id="voice_design"):
                    create_voice_design_tab()

                with gr.TabItem(t("tabs.voice_clone"), id="voice_clone"):
                    create_voice_clone_tab()

                with gr.TabItem(t("tabs.settings"), id="settings"):
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
    parser.add_argument("--lang", type=str, default="ja",
                        choices=["ja", "en", "zh", "ko", "ru", "es", "it", "de", "fr", "pt"],
                        help="UI 言語")

    args = parser.parse_args()

    check_dependency_versions()

    logger.info("Qwen3-TTS-Mac-GeneLab Web UI を起動中...")
    logger.info(f"ホスト: {args.host}, ポート: {args.port}, 言語: {args.lang}")

    app = create_app(default_lang=args.lang)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS,
        js=_UPLOAD_FIX_JS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()
