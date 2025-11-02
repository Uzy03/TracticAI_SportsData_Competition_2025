# 学習実行ガイド

このドキュメントでは、コンテナ内で受け手予測モデルを学習する方法を説明します。

## クイックスタート

### 1. コンテナに入る

```bash
docker exec -it gpu-dev bash
```

### 2. プロジェクトのセットアップ

```bash
# 仮想環境は自動的に有効化されているはず
# 念のため確認
which python
# 出力: /home/dev/.venv/bin/python

# プロジェクトをインストール
cd /workspace
pip install -e .

# PyTorch をインストール（まだの場合）
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

### 3. 受け手予測モデルの学習

#### 基本コマンド

```bash
python tacticai/train/train_receiver.py --config configs/receiver.yaml
```

#### デバッグモード（小規模データで動作確認）

```bash
python tacticai/train/train_receiver.py --config configs/receiver.yaml --debug_overfit
```

#### チェックポイントから再開

```bash
python tacticai/train/train_receiver.py --config configs/receiver.yaml --resume checkpoints/receiver/best.ckpt
```

## エポック数の変更方法

### 方法1: 設定ファイルを編集（推奨）

```bash
# 設定ファイルを開く
nano configs/receiver.yaml
# または
vim configs/receiver.yaml
```

**変更箇所：**

```yaml
# Training configuration
train:
  batch_size: 1
  epochs: 3    # ← この数値を変更
  amp: true    # Automatic mixed precision
```

例：50エポックに変更する場合

```yaml
train:
  batch_size: 1
  epochs: 50
  amp: true
```

### 方法2: コマンドライン引数（将来の拡張）

現在のスクリプトはコマンドライン引数でエポック数を指定する機能はないため、**設定ファイルを編集する方法**を使用してください。

## 学習結果の確認

### チェックポイントの保存場所

```bash
checkpoints/receiver/best.ckpt
```

### 学習履歴の保存場所

```bash
runs/receiver_training_history.json
```

### 学習ログの確認

```bash
tail -f runs/receiver_training.log
```

### TensorBoard（将来の拡張）

```bash
tensorboard --logdir runs
```

## 設定ファイルの主要パラメータ

`configs/receiver.yaml` で設定可能な主要なパラメータ：

```yaml
# モデル設定
model:
  input_dim: 8          # ノード特徴の次元
  hidden_dim: 128       # 隠れ層の次元
  num_classes: 22       # プレイヤー数
  num_layers: 3         # レイヤー数
  num_heads: 4          # アテンションヘッド数
  dropout: 0.2          # ドロップアウト率

# 最適化設定
optimizer:
  type: "adam"          # Adam optimizer
  lr: 0.0003            # 学習率
  weight_decay: 0.0001  # 重み減衰

# 学習設定
train:
  batch_size: 1         # バッチサイズ
  epochs: 3             # エポック数 ← これを変更
  amp: true             # 混合精度学習

# Early stopping設定
early_stopping:
  patience: 10          # 改善が見られない場合の待機エポック数
```

## 推奨されるエポック数

データセットの規模に応じて：

- **小規模テスト**: 3-5 エポック
- **中規模データ**: 10-30 エポック
- **大規模データ**: 50-100+ エポック

Early stoppingが設定されているため、過学習を防ぐことができます。

## トラブルシューティング

### GPUが認識されない場合

```bash
# GPU確認
nvidia-smi

# PyTorchのCUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー

`configs/receiver.yaml` の `batch_size` を減らす：

```yaml
train:
  batch_size: 1  # これ以上減らせない（現在1）
```

### データが見つからない

```bash
# データの存在確認
ls -lh data/processed_ck/receiver_train/
ls -lh data/processed_ck/receiver_val/
```

## その他の学習タスク

### シュート予測

```bash
python tacticai/train/train_shot.py --config configs/shot.yaml
```

### パス生成（CVAE）

```bash
python tacticai/train/train_cvae.py --config configs/cvae.yaml
```

### 逐次学習（全タスク）

```bash
python scripts/sequential_training.py --config configs/sequential_training.yaml --phase receiver
```

## Makefileコマンド

便利なショートカット：

```bash
# デバッグ用の最小データで学習
make debug-train

# サンプル実行
make sample-run
```

## 学習の監視

### リアルタイムログ確認

```bash
# 別のターミナルで
docker exec -it gpu-dev bash
tail -f runs/receiver_training.log
```

### 学習履歴の可視化（将来の拡張）

学習履歴JSONを読み込んで可視化するスクリプトを作成できます。

