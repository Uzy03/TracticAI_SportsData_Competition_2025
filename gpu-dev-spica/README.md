# GPU Dev Environment for Spica

Tesla V100S (CUDA 11.6 対応ドライバ) を搭載したサーバ「spica」向けの GPU Docker 環境です。

## クイックスタート

```bash
# 1. ビルド
cd gpu-dev-spica
docker compose build

# 2. 起動
docker compose up -d

# 3. コンテナに入る
docker exec -it gpu-dev-spica bash

# 4. GPU確認
nvidia-smi

# 5. 依存関係インストール
pip install -e .
pip install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio
```

## 構成

```
gpu-dev-spica/
├── Dockerfile              # CUDA 11.6 + cuDNN 8 develイメージ
├── docker-compose.yml      # Docker Compose設定
├── .dockerignore           # コンテナに含めないファイル
└── README.md
```

## セットアップ手順

### 1. NVIDIA Container Toolkit の導入（サーバ側）

```bash
# 管理者権限で実行
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Dockerイメージのビルド

```bash
cd gpu-dev-spica
docker compose build
```

### 3. コンテナの起動

```bash
docker compose up -d
```

### 4. コンテナに入る

```bash
docker exec -it gpu-dev-spica bash
```

### 5. GPUの確認

```bash
nvidia-smi
```

### 6. プロジェクトの依存関係をインストール（コンテナ内）

```bash
source ~/.venv/bin/activate
cd /workspace
pip install -e .

# PyTorch (CUDA 11.6 ビルド) が必要な場合
pip install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio

# 追加ライブラリ（必要に応じて）
pip install pytorch-lightning lightning wandb tensorboard
```

## JupyterLabの起動（任意）

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
```

SSH ポートフォワーディング：

```bash
ssh -L 8888:localhost:8888 <user>@<server-ip>
# ブラウザで http://localhost:8888
```

## 便利なコマンド

```bash
# コンテナの停止
docker compose down

# コンテナの再起動
docker compose restart

# ログの確認
docker compose logs -f

# イメージの再ビルド
docker compose build --no-cache
```

## データの扱い

`.dockerignore` により大容量データやキャッシュはイメージに含めず、ホスト側と共有します。

- `SoccerData/`
- `data/raw/`, `data/processed/`, `data/processed_ck/`
- `checkpoints/`, `runs/`, `logs/`

## PyTorch をイメージに含める場合

Dockerfile の該当行（コメントアウト済み）を有効化して再ビルドしてください：

```dockerfile
# ---- PyTorch (CUDA 11.6) を使う場合 ----
RUN ~/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio
```

```bash
docker compose build --no-cache
```

## トラブルシューティング

- GPU が認識されない場合：`docker --version`, `docker info | grep -i runtime`, コンテナ内で `nvidia-smi` を確認
- 権限エラーの場合：`docker exec -it gpu-dev-spica id` で UID/GID を確認し、必要に応じて Dockerfile の `ARG UID` を変更
- ポート競合の場合：`docker-compose.yml` の `ports` セクションでホスト側のポート番号を変更


