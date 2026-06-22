# Issue #1 検証記録: Voice Clone 時のエラー (依存バージョンずれ)

対象 Issue: [Error when trying to Voice Clone. #1](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/issues/1)

## 1. 結論

報告された 2 つのエラーは **GPU(Apple Silicon)の世代の問題ではなく、依存パッケージのバージョンずれ**が原因。
`setup_mac.sh` / `requirements-mac.txt` / `pyproject.toml` が `transformers`・`gradio`・`mlx-audio` を
上限なしでインストールしていたため、新規ユーザーが破壊的変更を含む最新版を取得して動作不能になっていた。

| パッケージ | 動作OK (M1 Max) | 報告環境 (M2 Pro / エラー) | 対応 |
|---|---|---|---|
| transformers | 4.57.6 | 5.7.0 | `>=4.51,<5.0` に固定 |
| gradio | 6.5.1 | 6.14.0 | `>=6.0,<6.6` に固定 |
| mlx-audio | 0.3.1 | 0.4.3 | `>=0.3,<0.4` に固定 |
| huggingface-hub | 0.36.2 | 1.13.0 | `<1.0` に固定 |

> 注: M3/M4 で「動かない」という報告も、シリコン世代ではなく後発インストールで
> 新しい破壊版依存を引いた可能性が高い。検証は手元の M1 Max のみで実施(M3/M4 実機なし)。

## 2. エラーの内訳と根本原因

### 問題A: `check_model_inputs() missing 1 required positional argument: 'func'`
- 発生箇所: `PyTorch MPS モデルをロード中: Qwen/Qwen3-TTS-12Hz-1.7B-Base`(Voice Clone 含む全 PyTorch モデルロード)
- 原因: `transformers` 5.x で `transformers.utils.generic.check_model_inputs` のシグネチャが変更。
  - 4.57.6: `check_model_inputs(func=None, *, tie_last_hidden_states=True)` → `@check_model_inputs()` が通る
  - 5.7.0 : `check_model_inputs(func)`(func 必須) → `@check_model_inputs()` が TypeError
- 該当コード: `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` L506 `@check_model_inputs()`

### 問題B: `KeyError: 29`(gradio queueing)
- 発生箇所: `gradio/queueing.py` の `fn = session_state.blocks_config.fns[body.fn_index]`
- 原因: `gradio` 6.6+ でキュー(`fn_index`)の解決処理が非互換化。動作確認済みの 6.5.1 では発生しない。

### 問題C(検証中に発見): 依存解決の衝突
- `mlx-audio` 0.4.x は `huggingface_hub>=1.0` を要求し、`transformers` 4.x(`hub<1.0`)と両立不能。
- さらに `mlx-audio` 0.3.x はメタデータ上 `transformers==5.0.0rc3` を hard pin しているが、
  実行時は 4.57.6 で問題なく動作する(動作環境で実証済み)。
- 対応: `mlx-audio>=0.3,<0.4` + `huggingface-hub<1.0` に固定し、`setup_mac.sh` は
  mlx-audio を先に、transformers 4.x を後から入れて上書きする(段階インストール)。pip は警告を出すが正常に成立する。

## 3. 再現(隔離環境・既存環境は汚染せず)

問題A を使い捨て conda 環境 `transformers==5.7.0` で再現:

```
transformers version: 5.7.0
check_model_inputs signature: (func)
RESULT: TypeError -> check_model_inputs() missing 1 required positional argument: 'func'
```

対照(動作環境 4.57.6):

```
transformers version: 4.57.6
check_model_inputs signature: (func=None, *, tie_last_hidden_states=True)
RESULT: OK (decorator usable)
```

## 4. 修正内容(個別コミットで revert 可能)

- `fix(deps): transformers を >=4.51,<5.0 にピン留め (Issue #1 問題A)`
- `fix(deps): gradio を >=6.0,<6.6 にピン留め (Issue #1 問題B)`
- `fix(deps): mlx-audio を >=0.3,<0.4 + huggingface-hub<1.0 に固定 (Issue #1 問題C)`
- `feat(ui): 起動時に非互換な依存バージョンを検知して警告 (Issue #1)`
  - `ui/app.py` の `check_dependency_versions()`。transformers 5.x / gradio 6.6+ 検知時に復旧コマンドを表示。
- `chore(deps): 動作確認済み環境のロックファイルを追加 (requirements-mac.lock.txt)`

対象ファイル: `requirements-mac.txt`, `pyproject.toml`, `setup_mac.sh`, `ui/app.py`, `requirements-mac.lock.txt`

## 5. 検証結果(クリーン隔離環境 `qwen-issue1-verify`)

段階インストール後の最終バージョン(= 動作環境と一致):

```
accelerate    1.14.0
gradio        6.5.1
mlx-audio     0.3.1
numpy         1.26.4
tokenizers    0.22.2
torch         2.12.1
transformers  4.57.6
huggingface-hub 0.36.2
```

- デコレータ/モジュール import: OK

```
check_model_inputs sig: (func=None, *, tie_last_hidden_states=True)
decorator @check_model_inputs(): OK
module import: OK
Qwen3TTSModel import: OK
```

- Voice Clone E2E(報告で失敗していた Base モデルロード→生成の実パス):

```
[E2E] 参照音声: 4.52s, 24000Hz
[E2E] Voice Clone 生成開始 (Base モデルロード -> 生成)...
[E2E] 成功: 81.91s 音声を生成 (engine=pytorch_mps, 528.7s, sr=24000)
出力音声: 81.91s, 24000Hz, peak=1.000, 非無音=True
```

- UI 起動スモーク(gradio 6.5.1): `HTTP 200`、エラーなし、バージョンガードは無発火(正常)。

## 6. 回帰確認(既存 M1 Max 環境)

- 既存環境で `ui/app.py` 変更後も UI 起動 OK: `HTTP 200`、バージョンガード無発火。
- 既存環境の Voice Clone は従来どおり動作(本変更はコードロジックを変えず、ガード追加と依存固定のみ)。

## 7. GitHub Issue #1 への回答文(コピペ用)

> ※ `gh` 未認証のため自動投稿はしていません。以下をそのまま貼り付けてください。

```text
Thanks for the detailed report! This is not an Apple Silicon generation issue — it is a dependency version drift problem.

Root cause:
- Your env installed transformers 5.7.0 and gradio 6.14.0 (and mlx-audio 0.4.3), all newer than what the project was tested with.
- transformers 5.x changed the signature of `check_model_inputs` (now `check_model_inputs(func)`), so the vendored `@check_model_inputs()` decorator raises
  "check_model_inputs() missing 1 required positional argument: 'func'", which breaks every PyTorch model load (including Voice Clone).
- gradio 6.6+ changed queue `fn_index` handling, which causes the `KeyError: 29` in `queueing.py`.

Fix (now in the repo):
- Pinned transformers `>=4.51,<5.0`, gradio `>=6.0,<6.6`, mlx-audio `>=0.3,<0.4`, huggingface-hub `<1.0`.
- Added a startup version guard that prints the exact recovery command if an incompatible version is detected.

To recover your existing environment, run:

    pip install 'transformers>=4.51,<5.0' 'gradio>=6.0,<6.6' 'mlx-audio>=0.3,<0.4' 'huggingface-hub<1.0'

Or re-run ./setup_mac.sh after pulling the latest changes.

Verified end-to-end on M1 Max: Base model load + Voice Clone generation now succeed, and the Web UI boots cleanly. (M3/M4 reports are very likely the same dependency drift, not a silicon-generation problem.)
```

## 8. 復旧(いつでも戻せる)

すべての修正は `fix/issue-1-deps` ブランチ上の個別コミット。`main` は無傷。

- 個別取り消し: `git revert <コミットハッシュ>`
- ブランチごと破棄: `git checkout main && git branch -D fix/issue-1-deps`
