# TacticAI Reproduction

TacticAIの再現を目的とした0→1実装プロジェクトです。3つのタスク（Receiver Prediction、Shot Prediction、Tactic Generation）を同一コードベースで実行可能な骨組みを提供します。

## 概要

このプロジェクトは、サッカーの戦術分析における以下の3つのタスクを実装します：

- **Receiver Prediction**: パスの受け手ノード分類（22クラス想定、Top-k評価あり）
- **Shot Prediction**: シーンからのシュート起こりやすさ二値分類（AUC等）
- **Tactic Generation**: CVAE系のセットプレー配置の条件付き生成・再構成

## 特徴

- **GATv2ベース**: Message Passingを採用（PyTorch Geometric不要）
- **D2等価性**: 水平・垂直反転に対する等価性を配慮
- **データアダプタ層**: 独自スキーマに差し替えやすい設計
- **最小依存**: PyTorch、numpy、pandas等の基本的なライブラリのみ使用

## インストール

### 必要な環境

- Python 3.10+
- PyTorch 2.0+
- CUDA（推奨）

### セットアップ

```bash
# リポジトリのクローン
git clone <repository-url>
cd tacticai-repl

# 依存関係のインストール
make setup

# または手動で
pip install -e .
pip install -e ".[dev]"
```

## 使用方法

### 1. データの前処理

```bash
# 独自データの前処理
python scripts/preprocess.py --input data/raw --output data/processed

# テスト用ダミーデータの作成
python scripts/preprocess.py --input data/raw --output data/processed --dummy
```

### 2. 学習の実行

```bash
# Receiver Prediction
python tacticai/train/train_receiver.py --config configs/receiver.yaml

# Shot Prediction  
python tacticai/train/train_shot.py --config configs/shot.yaml

# Tactic Generation (CVAE)
python tacticai/train/train_cvae.py --config configs/cvae.yaml
```

### 3. 評価の実行

```bash
# Receiver Prediction
python tacticai/eval/eval_receiver.py --config configs/receiver.yaml --checkpoint checkpoints/receiver/best.ckpt

# Shot Prediction
python tacticai/eval/eval_shot.py --config configs/shot.yaml --checkpoint checkpoints/shot/best.ckpt

# Tactic Generation
python tacticai/eval/eval_cvae.py --config configs/cvae.yaml --checkpoint checkpoints/cvae/best.ckpt --generate --samples 16
```

### 4. デバッグ・テスト

```bash
# オーバーフィットテスト
python tacticai/train/train_receiver.py --config configs/receiver.yaml --debug_overfit

# テストの実行
make test

# リンター・フォーマッター
make lint
make format
```

## プロジェクト構造

```
tacticai-repl/
├── configs/                 # 設定ファイル
│   ├── receiver.yaml
│   ├── shot.yaml
│   └── cvae.yaml
├── data/                    # データディレクトリ
│   ├── raw/                # 生データ（ユーザーが配置）
│   └── processed/          # 前処理済みデータ
├── tacticai/               # メインパッケージ
│   ├── models/            # モデル実装
│   │   ├── gatv2.py       # GATv2層・ネットワーク
│   │   ├── mlp_heads.py   # タスク別ヘッド
│   │   └── cvae.py        # CVAE実装
│   ├── modules/           # ユーティリティモジュール
│   │   ├── graph_builder.py
│   │   ├── transforms.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── utils.py
│   ├── dataio/            # データ入出力
│   │   ├── dataset.py
│   │   └── schema.py
│   ├── train/             # 学習スクリプト
│   └── eval/              # 評価スクリプト
├── scripts/               # ユーティリティスクリプト
│   ├── preprocess.py
│   └── scaffold_tasks.py
├── tests/                 # テスト
└── README.md
```

## データ形式

### 入力データの想定形式

各タスクで使用するデータは以下の列を含むCSV/Parquetファイルとして想定しています：

#### 基本列
- `x`, `y`: プレイヤーの位置座標
- `vx`, `vy`: プレイヤーの速度（オプション）
- `height`, `weight`: プレイヤーの属性（オプション）
- `team`: チーム情報（0/1）
- `ball`: ボール保持情報（0/1）

#### タスク別の追加列

**Receiver Prediction:**
- `receiver_id`: 正解の受け手ID（0-21）

**Shot Prediction:**
- `shot_occurred`: シュート発生フラグ（0/1）

**Tactic Generation:**
- `target_x`, `target_y`: 目標位置座標
- `condition`: 条件情報（8次元ベクトル）

### データ差し替え手順

独自データを使用する場合は、以下の手順で差し替え可能です：

1. **スキーマの修正**: `tacticai/dataio/schema.py`の各スキーマクラスで列名を調整
2. **データの配置**: `data/raw/`にCSV/Parquetファイルを配置
3. **前処理の実行**: `scripts/preprocess.py`で前処理

#### 暫定マッピング例

独自データの列名が不明な場合の暫定マッピング例：

```python
# 位置情報
position_columns = ["pos_x", "pos_y"]  # または ["x", "y"]

# 速度情報
velocity_columns = ["vel_x", "vel_y"]  # または ["vx", "vy"]

# チーム情報
team_column = "team_id"  # または "side"

# ボール情報
ball_column = "has_ball"  # または "ball_owner"

# ターゲット情報
receiver_column = "target_player"  # または "receiver_id"
shot_column = "is_shot"  # または "shot_occurred"
```

## 損失関数とハイパーパラメータ

### 損失関数の詳細

#### 1. Receiver Prediction (受け手予測)

**Cross-Entropy Loss with Label Smoothing**
```
L = -Σ y_i * log(p_i)
```
where `p_i = softmax(logits_i)` and `y_i = (1 - α) * y_i + α / num_classes`

- **α (label_smoothing)**: 0.1 - 過学習防止と汎化性能向上
- **weight**: 1.0 - 損失の重み
- **pos_weight**: null - クラス不均衡がない場合

**Focal Loss (オプション)**
```
L = -α * (1-p_t)^γ * log(p_t)
```
- **α**: 1.0 - 稀なクラスの重み
- **γ**: 2.0 - 焦点パラメータ

#### 2. Shot Prediction (シュート予測)

**Binary Cross-Entropy Loss with Label Smoothing**
```
L = -[y * log(p) + (1-y) * log(1-p)]
```
where `p = sigmoid(logits)` and `y = (1 - α) * y + α * 0.5`

- **α (label_smoothing)**: 0.1 - 過学習防止
- **pos_weight**: 1.0 - 正例の重み（クラス不均衡時は調整）
- **weight**: 1.0 - 損失の重み

**Focal Loss (オプション)**
```
L = -α * (1-p_t)^γ * log(p_t)
```
- **α**: 0.25 - 正例の重み
- **γ**: 2.0 - 焦点パラメータ

#### 3. CVAE Generation (戦術生成)

**Enhanced CVAE Loss with Guidance and Constraints**
```
L_total = λ_recon * L_recon + λ_kl * L_kl + λ_guidance * L_guidance + λ_movement * L_movement + λ_velocity * L_velocity
```

**Reconstruction Loss**
```
L_recon = MSE(pred, target)  # or L1, Huber
```
- **λ_recon**: 1.0 - 再構成損失の重み

**KL Divergence Loss**
```
L_kl = β * KL(q(z|x) || p(z))
```
- **β**: 1.0 - β-VAEパラメータ（離散化制御）
- **λ_kl**: 1.0 - KL損失の重み

**Shot Guidance Loss**
```
L_guidance = MSE(shot_prob, guidance_target) * guidance_lambda
```
- **λ_guidance**: 0.1 - ガイダンス損失の重み
- **max_lambda**: 2.0 - 最大ガイダンス強度

**Movement Constraint Loss**
```
L_movement = max(0, ||pred_pos - orig_pos||_2 - max_movement)
```
- **λ_movement**: 0.1 - 移動制約の重み
- **max_movement**: 10.0 - 最大許容移動量（メートル）

**Velocity Constraint Loss**
```
L_velocity = max(0, ||velocity||_2 - max_velocity)
```
- **λ_velocity**: 0.1 - 速度制約の重み
- **max_velocity**: 5.0 - 最大許容速度（m/s）

### ハイパーパラメータの根拠

#### 学習率 (Learning Rate)
- **初期値**: 3e-4 - Adam最適化器での標準的な値
- **減衰**: Cosine annealing with warmup
- **Warmup epochs**: 5 - 学習初期の安定化

#### 重み減衰 (Weight Decay)
- **値**: 1e-4 - L2正則化による過学習防止
- **適用**: 全パラメータに適用

#### ドロップアウト
- **値**: 0.2 - 20%の確率でニューロンを無効化
- **目的**: 過学習防止と汎化性能向上

#### バッチサイズ
- **Receiver**: 8 - メモリ効率と学習安定性のバランス
- **Shot**: 2 - メモリ制約下での最小値
- **CVAE**: 4 - 生成タスクでのメモリ効率

#### D2等価性パラメータ
- **Group pooling**: false - ランダム反転を使用
- **Transforms**: hflip, vflip - 水平・垂直反転
- **Weight sharing**: true - 4ビュー間での重み共有

#### 早期停止
- **Patience**: 10-15 epochs - 過学習防止
- **Min delta**: 0.001 - 改善の最小閾値
- **Restore best weights**: true - 最良モデルの復元

### 評価指標

#### Receiver Prediction
- **Accuracy**: 正解率
- **Top-k Accuracy**: 上位k個の予測に正解が含まれる割合
- **F1 Score**: 精度と再現率の調和平均

#### Shot Prediction
- **Accuracy**: 正解率
- **Precision**: 正例の予測精度
- **Recall**: 正例の再現率
- **F1 Score**: 精度と再現率の調和平均
- **AUC**: ROC曲線下面積

#### CVAE Generation
- **Reconstruction Loss**: 再構成誤差
- **KL Loss**: KL divergence
- **Total Loss**: 総合損失
- **MSE**: 平均二乗誤差
- **MAE**: 平均絶対誤差

### 設定ファイルの更新

各タスクの設定ファイル（`configs/*.yaml`）に上記の損失関数とハイパーパラメータが詳細に記載されています。必要に応じて調整してください。

### 順次学習プロトコル

受け手→ショット→生成の順で学習する場合：

1. **Phase 1 (Receiver)**: エンコーダを事前学習
2. **Phase 2 (Shot)**: エンコーダを凍結してショット予測を学習
3. **Phase 3 (Generation)**: エンコーダを凍結してCVAEを学習

詳細は `configs/sequential_training.yaml` を参照してください。

## よくある問題と解決方法

### 1. メモリ不足

```bash
# バッチサイズを小さくする
# configs/*.yaml の batch_size を 32 や 16 に変更
```

### 2. CUDA関連エラー

```bash
# CPUで実行
python tacticai/train/train_receiver.py --config configs/receiver.yaml
# または configs/*.yaml で device: "cpu" に設定
```

### 3. データ形式エラー

```bash
# スキーマを確認・修正
# tacticai/dataio/schema.py の列名を確認
```

### 4. 学習が収束しない

```bash
# 学習率を下げる
# configs/*.yaml の lr を 1e-4 や 1e-5 に変更

# デバッグモードで確認
python tacticai/train/train_receiver.py --config configs/receiver.yaml --debug_overfit
```

## 新規タスクの追加

新しいタスクを追加する場合は、`scaffold_tasks.py`を使用：

```bash
python scripts/scaffold_tasks.py <task_name>
```

これにより、新タスク用のスキーマ、データセット、ヘッド、学習スクリプトの雛形が生成されます。

## 開発・貢献

### コード規約

- **フォーマット**: black + ruff
- **型チェック**: mypy
- **テスト**: pytest

### 開発コマンド

```bash
# コード品質チェック
make check-all

# テスト実行
make test

# クリーンアップ
make clean
```

## ライセンス

MIT License

## 参考資料

- [TacticAI: An AI System for Football Match Analysis](https://arxiv.org/abs/2311.16039)
- [Graph Attention Networks v2](https://arxiv.org/abs/2105.14491)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## 更新履歴

- v0.1.0: 初期実装完了
  - 3タスクの基本実装
  - GATv2ベースのモデル
  - D2等価性対応
  - データアダプタ層
  - テストスイート

---

**注意**: この実装はTacticAIの再現を目的とした骨組みです。精度の一致は保証されませんが、モデル構造・入出力・学習フローはTacticAI系タスクの再現性を重視しています。
