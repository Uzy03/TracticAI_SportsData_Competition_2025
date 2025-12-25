# Similar CK Retrieval Visualization

Streamlitアプリケーションによる類似CK検索結果の可視化システム。

## 概要

この可視化システムは、類似CK検索システムの検索結果を視覚的に確認・評価するためのWebアプリケーションです。

## 機能

- **検索結果の可視化**: クエリCKと類似CKをサッカーコート上に表示
- **インタラクティブな操作**: Plotlyによるズーム、パン、ホバー情報表示
- **表示オプション**: 
  - 速度ベクトルの表示ON/OFF
  - ベクトルのスケール調整
  - 選手IDの表示ON/OFF
  - グリッドレイアウト（1列～4列）

## インストール

必要なライブラリをインストール：

```bash
pip install -r requirements.txt
```

または、プロジェクトルートから：

```bash
pip install streamlit plotly pandas numpy pyyaml
```

## 実行方法

プロジェクトルートから実行：

```bash
streamlit run tacticai/retrieval/visualization/app.py
```

または、`tacticai/retrieval/visualization/`ディレクトリから：

```bash
streamlit run app.py
```

ブラウザが自動的に開き、`http://localhost:8501`でアプリケーションが起動します。

## 使用方法

1. **設定**（サイドバー）:
   - Config file path: 設定ファイルのパス（例: `configs/multitask_receiver_shot_d2.yaml`）
   - Index file path: 検索インデックスファイルのパス（例: `runs/retrieval/index_d2.pkl`）
   - Data path: データセットのパス（例: `data/processed_ck/receiver_train/data.pickle`）
   - Query sample index: クエリサンプルのインデックス
   - Top-k results: 取得する類似CKの数

2. **表示オプション**（サイドバー）:
   - Show velocity vectors: 速度ベクトルを表示するか
   - Vector scale: ベクトルのスケール係数
   - Show player IDs: 選手IDを表示するか
   - Number of columns: 結果表示の列数

3. **検索実行**: 「🔍 Search」ボタンをクリック

4. **結果の確認**: 
   - クエリCKが上部に表示されます
   - その下にTop-k類似CKがグリッド形式で表示されます
   - 各結果には類似度とインデックスが表示されます

## ファイル構成

```
tacticai/retrieval/visualization/
├── app.py              # Streamlitメインアプリケーション
├── utils.py            # コート描画、データロード関数
├── requirements.txt    # 必要なライブラリ
└── README.md           # このファイル
```

## 注意事項

- データは単一フレーム（スナップショット）を想定しています
- 時系列データ（複数フレーム）の場合は、将来的に拡張が必要です
- 大きなデータセットの場合、初回ロードに時間がかかる場合があります（キャッシュ機能により2回目以降は高速化されます）

## トラブルシューティング

- **エラー: Module not found**: `pip install -r requirements.txt`で必要なライブラリをインストールしてください
- **エラー: File not found**: 設定ファイル、インデックスファイル、データファイルのパスが正しいか確認してください
- **エラー: Index out of range**: クエリサンプルのインデックスがデータセットの範囲内か確認してください

