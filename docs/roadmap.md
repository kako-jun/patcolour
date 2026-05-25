# patcolour Roadmap

## 現状

- CLI の最小機能はある
- README / docs / CI / tests は整備済み
- ただし肝心の「狙い通りに色が残るか」はまだ弱い
- `docs/research-notes.md` に色空間・relative guide・blob 選別の前提メモを残した
- neon 夜景のような高コントラスト input も検証対象として意識し始めた
- 方針: 最初の実戦投入から full guide surface を持つ。relative / negative を後回しにしない

## 近い課題

- `--auto-detect` の誤検出を減らす
- 花、リボン、服飾など小領域での keep-color 精度を上げる
- バッチ処理時に「マスクが見つからない」「色がほぼ残っていない」を検知しやすくする
- Lab / LCh / xyY ベースの色距離ロジックを比較し、最終的な本命を決める
- positive/negative guide を入れて「この紫は残すが、あの紫は落とす」を可能にする

## 推奨着手順

1. **Issue #4: relative guides as the primary CLI path**
   - まず人間の指示方法を使いやすくする
2. **Issue #5: negative-guide semantics**
   - 「この色は違う」を言えるようにする
3. **Issue #2: Lab/LCh vs xyY comparison**
   - guide 前提で最も良い色距離ロジックを決める
4. **Issue #6: connected-component scoring**
   - 「全部の紫」ではなく「その花」へ寄せる
5. **Issue #7: preprocess stage**
   - noisy な写真で mask の安定性を上げる
6. **Issue #3: golden-image regression suite**
   - 比較対象が揃ってから保守基盤に固定する
7. **Issue #1: human-guided selective color umbrella**
   - 上の実験結果を統合して最終ワークフローを整える

## 優先順位の考え方

- 先に **guide UX** を固める
- 次に **color-distance logic** を比較する
- その後で **blob selection** と **preprocess** で難しい写真に対応する
- regression suite は比較条件が見えた段階で固定する

## CLI 方針

- 最初から full option surface を用意する
- 少なくとも以下は初期段階で揃える:
  - positive / negative point
  - positive / negative rect
  - positive / negative ellipse
  - relative variants
- 実装の中身は段階的に磨いても、引数の契約は早めに固定する

## 予定している拡張

- `--preprocess` で写真のノイズや質感を先に整える
- 自然言語指定ワークフローのための選択戦略整理
- 期待結果サンプル画像を用いた回帰テスト
- 他ツールから安全に呼べる API / exit code / 出力規約の固定
