# Issue: Spicaサーバ用Docker環境構築の完成

## タイトル

**Spicaサーバ用Docker環境（gpu-dev-spica）のビルドエラー修正とセットアップ完成**

## 説明

### 背景

現在、remoteSSHサーバ「scorpio」が故障し、代替サーバ「spica」への接続が必要になった。現在のDocker環境（`gpu-dev`）はscorpio用にCUDA 12.4で構築されているが、spicaはTesla V100Sを搭載しており、CUDA 11.6対応のドライバを使用しているため、別のDocker環境（`gpu-dev-spica`）が必要。

### 現状

- `gpu-dev-spica/`ディレクトリにDockerfileとdocker-compose.ymlが存在
- DockerfileはCUDA 11.6 + cuDNN 8 + Ubuntu 20.04ベースで構成されている
- Python 3.10.14をソースからビルドしようとしている
- **ビルド時にエラーが発生しており、完成していない**

### 目標

1. `gpu-dev-spica`のDockerfileを修正してビルドエラーを解決
2. `gpu-dev`と同様に動作するDocker環境を構築
3. PyTorch（CUDA 11.6版）が正しくインストールできることを確認
4. プロジェクトの依存関係（`pyproject.toml`の依存関係）が正しくインストールできることを確認

### 期待される動作

```bash
cd gpu-dev-spica
docker compose build  # エラーなくビルド完了
docker compose up -d
docker exec -it gpu-dev-spica bash
nvidia-smi  # GPU認識確認
pip install -e .
pip install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio
python tacticai/train/train_receiver.py --config configs/receiver.yaml  # 学習が実行できる
```

## 必要な情報・確認事項

### 1. Spicaサーバの環境情報

以下の情報を確認して提供してください：

```bash
# Spicaサーバで実行
nvidia-smi  # CUDA Driver Versionを確認
nvcc --version  # CUDA Toolkitバージョン（インストールされている場合）
docker --version
docker info | grep -i runtime
cat /etc/os-release  # OS情報
```

**確認したいこと:**
- CUDA Driver Versionが11.6以上か（V100SはCUDA 11.xまで対応）
- DockerとNVIDIA Container Toolkitが正しくインストールされているか

### 2. ビルドエラーの詳細

`docker compose build`を実行した際のエラーメッセージを共有してください：

```bash
cd gpu-dev-spica
docker compose build 2>&1 | tee build.log
```

**特に確認したいエラー:**
- Python 3.10.14のソースビルド時のエラー（`make`コマンドでのコンパイルエラー）
- 依存関係のインストール時のエラー
- 権限関連のエラー
- その他のビルドエラー

### 3. 既存のgpu-dev環境との違い

比較すべき主な違い：

| 項目 | gpu-dev (scorpio) | gpu-dev-spica (spica) |
|------|-------------------|----------------------|
| CUDA | 12.4.1 | 11.6.2 |
| cuDNN | 9 | 8 |
| Ubuntu | 22.04 | 20.04 |
| Python | システムのpython3 | ソースからビルド (3.10.14) |
| PyTorch CUDA | cu124 | cu116 |

### 4. 修正が必要な可能性がある箇所

#### a) Pythonビルドの問題

現在のDockerfileでは、Python 3.10.14をソースからビルドしようとしているが、これがエラーの原因である可能性が高い。

**修正案1**: システムのPython 3.10を使用（Ubuntu 20.04では`python3.10`パッケージが利用可能）
**修正案2**: Python 3.10のソースビルドを修正（依存関係の追加、ビルドオプションの調整）

#### b) PyTorchの依存関係

`pyproject.toml`では`torch>=2.0.0`を要求しているが、CUDA 11.6用のPyTorchがインストール可能か確認が必要。

#### c) その他の依存関係

以下の依存関係がCUDA 11.6環境で動作するか確認：
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- その他（`pyproject.toml`参照）

### 5. 確認すべきファイル

以下のファイルを確認・比較してください：

1. **gpu-dev/Dockerfile** vs **gpu-dev-spica/Dockerfile**
   - 違いを明確にして、必要な修正を特定

2. **gpu-dev/docker-compose.yml** vs **gpu-dev-spica/docker-compose.yml**
   - 設定の違いを確認

3. **pyproject.toml**
   - 依存関係のバージョン要件を確認
   - CUDA 11.6環境でも動作するか

### 6. 実装時の注意点

1. **Pythonのビルド時間**: ソースからビルドすると時間がかかるため、可能であればパッケージ版を使用
2. **メモリ使用量**: ビルド時にメモリ不足が発生する可能性がある
3. **互換性**: Ubuntu 20.04のリポジトリが古いため、一部のパッケージで問題が発生する可能性がある

## 実装方針（推奨）

### 方針A: システムのPython 3.10を使用（推奨）

Ubuntu 20.04では`python3.10`パッケージが利用可能なため、ソースビルドを避ける：

```dockerfile
# システムのPython 3.10をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    # ... その他の依存関係

# 仮想環境を作成
RUN python3.10 -m venv ~/.venv
```

### 方針B: Python 3.10のソースビルドを修正

ソースビルドを続ける場合、必要なビルド依存関係を追加：

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ... 既存の依存関係に加えて
    libffi-dev libbz2-dev liblzma-dev \
    # Pythonビルドに必要な追加依存関係
```

## 完了条件

- [ ] `docker compose build`がエラーなく完了する
- [ ] `docker compose up -d`でコンテナが正常に起動する
- [ ] コンテナ内で`nvidia-smi`が正常に動作し、GPUが認識される
- [ ] `pip install -e .`でプロジェクトの依存関係がインストールされる
- [ ] `pip install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio`でPyTorchがインストールされる
- [ ] サンプル学習コマンド（`python tacticai/train/train_receiver.py --config configs/receiver.yaml`）が正常に実行される（最低限の動作確認）
- [ ] README.mdにセットアップ手順が記載されている

## 関連ファイル

- `gpu-dev-spica/Dockerfile`
- `gpu-dev-spica/docker-compose.yml`
- `gpu-dev-spica/README.md`
- `gpu-dev/Dockerfile`（参考）
- `pyproject.toml`（依存関係確認用）

