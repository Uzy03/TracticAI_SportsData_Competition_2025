# Receiver Prediction Model ハイパーパラメータ設定（2025-12-04）

現在使用中のハイパーパラメータ設定を記録します。

## 設定ファイル
- **ファイル**: `configs/receiver_full_simplified.yaml`
- **最終更新**: 2025-12-04
- **使用実績**: `runs/training_20251204_072755.log`で使用

## モデル設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `input_dim` | 16 | ノード特徴量次元（x, y, vx, vy, height, weight, ball_possession + キッカー相対4次元 + ゴール相対4次元 + team_id） |
| `hidden_dim` | 512 | GATv2の隠れ層次元数 |
| `num_classes` | 22 | 分類クラス数（プレイヤー数） |
| `num_layers` | 3 | GATv2レイヤー数 |
| `num_heads` | 4 | アテンションヘッド数 |
| `dropout` | 0.0 | ドロップアウト率 |
| `edge_dim` | 10 | エッジ特徴量次元（dx, dy, dist_ij, angle_ij, same_team, dvx, dvy, rel_speed, from_kicker, to_kicker） |

## オプティマイザー設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `type` | adam | オプティマイザータイプ |
| `lr` | 0.001 | 学習率 |
| `weight_decay` | 0.0 | L2正則化の重み |

## スケジューラー設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `type` | none | スケジューラータイプ（使用なし） |

## 学習設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `batch_size` | 32 | バッチサイズ |
| `epochs` | 50 | 最大エポック数 |
| `amp` | false | Automatic Mixed Precisionの使用（無効） |
| `grad_clip.enabled` | false | 勾配クリッピングの有効化（無効） |

## データ設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `train_path` | `data/processed_ck/receiver_train/data.pickle` | 訓練データパス |
| `val_path` | `data/processed_ck/receiver_val/data.pickle` | 検証データパス |
| `test_path` | `data/processed_ck/receiver_test/data.pickle` | テストデータパス |
| `format` | pickle | データ形式 |
| `merge_val_to_train` | true | **ValデータをTrainに統合**（Train: 261 → 316サンプル） |

## 損失関数設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `type` | cross_entropy | 損失関数タイプ |
| `label_smoothing` | 0.0 | ラベルスムージング率 |
| `weight` | 1.0 | 損失の重み |
| `focal_loss.enabled` | false | Focal Lossの使用（無効） |
| `class_weights` | null | クラスごとの重み（使用なし） |
| `regularization.l2_weight` | 0.0 | L2正則化の重み |
| `regularization.dropout_rate` | 0.0 | ドロップアウト正則化の重み |

## D2等変性設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `group_pool` | false | グループプーリングの使用（無効） |
| `transforms.hflip` | false | 水平フリップのデータ拡張（無効） |
| `transforms.vflip` | false | 垂直フリップのデータ拡張（無効） |

## Early Stopping設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `patience` | 20 | 改善が見られない場合の待機エポック数 |
| `min_delta` | 0.0005 | 改善とみなす最小変化量 |

**注意**: `merge_val_to_train: true`の場合、Early stoppingは無効化される（Train metricsベースでbest modelを保存）

## 評価設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `batch_size` | 5 | 評価時のバッチサイズ |
| `min_cands_eval` | 1 | 評価に必要な最小候補数 |

## その他設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `device` | auto | デバイス（自動選択） |
| `seed` | 42 | 乱数シード |
| `num_workers` | 4 | データローダーのワーカー数 |
| `prefetch_factor` | 2 | プリフェッチ係数 |
| `persistent_workers` | true | ワーカーをエポック間で保持 |
| `log_level` | INFO | ログレベル |
| `log_dir` | runs | ログ保存ディレクトリ |
| `checkpoint_dir` | checkpoints | チェックポイント保存ディレクトリ |

## この設定での性能結果

### 最終性能（エポック50）

| 指標 | Train | Test |
|------|-------|------|
| Loss | 2.3039 | 2.2940 |
| Top-1 Accuracy | 25.6% | 26.3% |
| Top-3 Accuracy | 48.4% | 40.4% |
| Top-5 Accuracy | 68.0% | 57.9% |

### Best Model性能（エポック43）

- **Train Top-3**: 58.5%
- **Test Top-3**: 40.4%

## 設定の特徴

1. **シンプルな設定**: 
   - Dropout: 0.0
   - Label smoothing: 0.0
   - Weight decay: 0.0
   - スケジューラー: なし
   - データ拡張: なし

2. **大きなモデルサイズ**:
   - hidden_dim: 512（デフォルト128より大きい）

3. **Val統合**:
   - `merge_val_to_train: true`で学習データを増加

4. **Early stopping設定**:
   - patience: 20, min_delta: 0.0005

## 以前の設定との主な違い

- **hidden_dim**: 128 → 512（増加）
- **dropout**: 0.1 → 0.0（無効化）
- **lr**: 0.0005 → 0.001（増加）
- **weight_decay**: 0.0001 → 0.0（無効化）
- **scheduler**: cosine → none（無効化）
- **amp**: true → false（無効化）
- **grad_clip**: enabled → disabled（無効化）
- **label_smoothing**: 0.1 → 0.0（無効化）
- **hflip/vflip**: true → false（無効化）
- **early_stopping.patience**: 50 → 20（減少）
- **early_stopping.min_delta**: 0.001 → 0.0005（減少）
- **epochs**: 200 → 50（減少）
- **data**: processed_ck_v3 → processed_ck（変更）
- **merge_val_to_train**: false → true（追加）

## 日付
2025-12-04

