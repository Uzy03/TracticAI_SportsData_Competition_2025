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

### ローカル環境

プロジェクトルートから実行：

```bash
streamlit run tacticai/retrieval/visualization/app.py
```

または、`tacticai/retrieval/visualization/`ディレクトリから：

```bash
streamlit run app.py
```

ブラウザが自動的に開き、`http://localhost:8501`でアプリケーションが起動します。

### リモートSSH環境（Dockerコンテナ含む）

リモートサーバーで実行する場合、以下の手順が必要です：

1. **SSHポートフォワーディングの設定**（ローカルマシン側）:
```bash
ssh -L 8501:localhost:8501 <user>@<remote_host>
# Dockerコンテナの場合
ssh -L 8501:localhost:8501 -p <port> <user>@<remote_host>
# または、コンテナにポートマッピングがある場合
docker exec -it <container_name> bash
```

2. **リモートサーバー（コンテナ内）でStreamlitを起動**:
```bash
# コンテナ内で実行
streamlit run tacticai/retrieval/visualization/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

3. **ローカルブラウザでアクセス**:
- `http://localhost:8501` を開く

**注意**: エラーが出る場合は、ターミナルに表示されるエラーメッセージを確認してください。

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

### よくあるエラー

- **エラー: Module not found**: 
  - `pip install streamlit plotly` で必要なライブラリをインストールしてください
  - または `pip install -e .` でプロジェクトを再インストールしてください

- **エラー: File not found**: 
  - 設定ファイル、インデックスファイル、データファイルのパスが正しいか確認してください
  - リモート環境では絶対パスまたはプロジェクトルートからの相対パスを使用してください

- **エラー: Index out of range**: 
  - クエリサンプルのインデックスがデータセットの範囲内か確認してください

- **エラー: ページを開けません / Connection refused**:
  - **リモートSSH環境の場合**: SSHポートフォワーディングが正しく設定されているか確認してください
  - Streamlitが正常に起動しているか、ターミナルのログを確認してください
  - `--server.address 0.0.0.0` オプションを指定して起動してください
  - ポートが他のプロセスで使用されていないか確認してください: `lsof -i :8501`

- **アプリケーションが起動しない**:
  - ターミナルで直接実行してエラーメッセージを確認してください:
    ```bash
    python -m streamlit run tacticai/retrieval/visualization/app.py
    ```
  - インポートエラーの場合は、プロジェクトルートで実行しているか確認してください
  - `PYTHONPATH`を設定する場合: `PYTHONPATH=/workspace:$PYTHONPATH streamlit run ...`

