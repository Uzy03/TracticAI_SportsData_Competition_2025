# GPU Dev Environment

リモートSSHサーバ用のGPU Docker環境です。

## クイックスタート

```bash
# 1. ビルド
cd gpu-dev
docker compose build

# 2. 起動
docker compose up -d

# 3. コンテナに入る
docker exec -it gpu-dev bash

# 4. GPU確認
nvidia-smi

# 5. 依存関係インストール
pip install -e .
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## 構成

```
gpu-dev/
├── Dockerfile              # CUDA 12.4 + cuDNN 9 develイメージ
├── docker-compose.yml      # Docker Compose設定
├── .dockerignore           # コンテナに含めないファイル
├── .devcontainer/
│   └── devcontainer.json   # VSCode/Cursor用（任意）
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
# gpu-devディレクトリに移動
cd gpu-dev

# Dockerイメージをビルド
docker compose build
```

### 3. コンテナの起動

```bash
docker compose up -d
```

### 4. コンテナに入る

```bash
docker exec -it gpu-dev bash
```

### 5. GPUの確認

```bash
nvidia-smi
```

### 6. プロジェクトの依存関係をインストール（コンテナ内）

```bash
# 仮想環境を有効化
source ~/.venv/bin/activate

# または、Dockerfileで既にPATHに設定済み

# プロジェクトの依存関係をインストール
cd /workspace
pip install -e .

# PyTorch (CUDA 12.4) のインストール（必要に応じて）
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# 追加ライブラリ（必要に応じて）
pip install pytorch-lightning lightning wandb tensorboard
```

## JupyterLabの起動（任意）

### コンテナ内で起動

```bash
# コンテナ内で実行
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
```

### ローカルからアクセス

```bash
# SSHポートフォワーディング
ssh -L 8888:localhost:8888 <user>@<server-ip>

# ブラウザでアクセス
# http://localhost:8888
```

## 便利なコマンド

```bash
# コンテナの停止
docker compose down

# コンテナの再起動
docker compose restart

# ログの確認
docker compose logs -f

# コンテナの削除（データは残る）
docker compose down

# イメージの再ビルド
docker compose build --no-cache
```

## データの扱い

`.dockerignore`により、以下のデータはコンテナに含まれません：

- `SoccerData/` - 生データ
- `data/raw/`, `data/processed/` - 前処理データ
- `checkpoints/`, `runs/` - モデルとログ

これらはボリュームマウントで共有されます（`docker-compose.yml`参照）。

## VSCode/Cursorから使用する場合

`.devcontainer/devcontainer.json`があるため、VSCode/Cursorで「Reopen in Container」を選択すると、自動的にコンテナ環境が起動します。

## ビルド時のオプション

DockerfileのPyTorchインストール行のコメントを外せば、イメージにPyTorchを含められます：

```dockerfile
# ---- PyTorch (CUDA 12.4) を使う場合 ----
RUN ~/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

その場合は、再ビルドしてください：

```bash
docker compose build --no-cache
```

## トラブルシューティング

### GPUが認識されない場合

```bash
# Dockerのバージョン確認
docker --version

# NVIDIA runtimeの確認
docker info | grep -i runtime

# コンテナ内で確認
nvidia-smi
```

### 権限エラーの場合

```bash
# コンテナのUIDを確認
docker exec -it gpu-dev id

# 必要に応じてUIDを変更（DockerfileのARG UID=1000を変更して再ビルド）
```

### ポート競合の場合

`docker-compose.yml`の`ports`セクションでポート番号を変更：

```yaml
ports:
  - "8889:8888"  # 左側を変更
```

