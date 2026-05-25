# patcolour

選択的に色を残すための小さな Python CLI。`skirts-colour` 本体の前段で、写真や素材の
"どこを残し、どこを記憶色へ落とすか" を試すための独立ツールであり、将来的には
`name-name` や他ツール群から呼ばれる画像加工基盤でもあります。

## ドキュメント

| ファイル | 内容 | 言語 |
|---|---|---|
| `README.md` | エンドユーザー向け概要と使い方 | 英語 |
| `docs/overview.md` | 何を目指すツールか、何をまだやらないか | 英語 |
| `docs/spec.md` | CLI 仕様と処理ルール | 英語 |
| `docs/roadmap.md` | 品質改善・Issue ドリブンの進行管理 | 日本語 |
| `CLAUDE.md` | AI 向け内部メモ | 日本語 |

## 現在の構造

```
src/patcolour/
├── cli.py      # argparse 入口
└── filter.py   # マスク生成とモノクロ合成

tests/
├── test_cli.py
└── test_filter.py
```

## 開発コマンド

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
uv run patcolour --help
```

## 設計メモ

- `patcolour` は「色を残す領域をどう指定するか」が本質
- 呼び出し元が増える前提なので、CLI 仕様と Python API の安定性を軽視しない
- 現状の `--auto-detect` は広い HSV 検出なので、狙い通りに当たりにくい
- 本命は RGB 直比較ではなく、Lab / LCh / xyY など知覚寄り色空間への変換
- まずは `--sample` + Lab chroma 半径で「この色相帯を残す」を実装し、後で厳密化する
- 画像品質の改善は、主観だけでなく再現可能なサンプル画像と期待結果で管理する
- `skirts-colour` の本丸ロジックを先に埋め込まず、このリポジトリでは小さく検証する

## 実装ルール

- 新しい選択ロジックを入れるときは、最低 1 件は synthetic test を追加する
- "良さそう" だけで閾値を固定しない。期待画像の観点を `docs/roadmap.md` か Issue に残す
- Python 実行は `uv run python3` を使う
