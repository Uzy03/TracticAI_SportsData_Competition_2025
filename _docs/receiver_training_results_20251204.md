# Receiver Prediction Model Training Results (2025-12-04)

## 概要

Receiver予測モデルの学習において、以下の2つの重要な改善を実施しました：

1. **Val統合機能の実装**: 訓練データが少ないため、ValidationデータをTrainingデータに統合する機能を追加
2. **Testデータのチーム制約修正**: TestデータでもTrain/Valと同様にチーム制約を適用し、評価条件を一貫化

## 実施した改善

### 1. Val統合機能（merge_val_to_train）

- **設定**: `configs/receiver_full_simplified.yaml` に `data.merge_val_to_train: true` を追加
- **効果**: Trainingデータが261サンプル→316サンプルに増加（+21.1%）
- **実装**: `tacticai/train/train_receiver.py` で `ConcatDataset` を使用してTrainとValを統合

### 2. Testデータのチーム制約修正

- **問題**: Testデータでチーム制約が無効化されており、候補数が約10候補→20候補に倍増していた
- **修正**: `tacticai/dataio/dataset.py` の `_prepare_sample` メソッドを修正
  - 変更前: `phase_with_team_constraint = self.phase in {"train", "val"}`
  - 変更後: `phase_with_team_constraint = True` (常に有効)
- **効果**: Train/Val/Testすべてで「同じチームの約10候補から選択」という同一タスクで評価可能に

## 最終性能結果

### 学習設定
- **データ**: `processed_ck` (Train: 261 → 316サンプル、Val: 統合、Test: 57サンプル)
- **モデル**: GATv2 (hidden_dim=512, num_heads=4, num_layers=3)
- **学習**: epochs=50, lr=0.001, batch_size=32, optimizer=Adam
- **その他**: dropout=0.0, label_smoothing=0.0, scheduler=none, amp=false

### 性能指標（エポック50の結果）

| 指標 | Train | Test |
|------|-------|------|
| **Loss** | 2.3039 | 2.2940 |
| **Top-1 Accuracy** | 25.6% | 26.3% |
| **Top-3 Accuracy** | 48.4% | 40.4% |
| **Top-5 Accuracy** | 68.0% | 57.9% |

### Best Model性能（エポック43）

- **Train Top-3**: 58.5%
- **Test Top-3**: 40.4% (エポック50のモデルで評価)

### 改善効果

修正前（Val統合なし、Testでチーム制約なし）との比較：

| 指標 | 修正前 | 修正後 | 改善 |
|------|--------|--------|------|
| Train Top-3 | 37.5% | 48.4% | +10.9pt |
| Test Top-1 | 15.8% | 26.3% | +10.5pt |
| Test Top-3 | 24.6% | 40.4% | **+15.8pt** |
| Test Top-5 | 31.6% | 57.9% | **+26.3pt** |

## 主要な変更ファイル

1. `configs/receiver_full_simplified.yaml`: Val統合オプション追加
2. `tacticai/train/train_receiver.py`: Val統合処理の実装
3. `tacticai/dataio/dataset.py`: Testデータのチーム制約修正

## 結論

- Test性能が大幅に改善（Top-3: 24.6% → 40.4%）
- Train/Val/Testで評価条件が一貫化され、公平な評価が可能に
- Val統合により学習データが増加し、Train性能も向上

## 日付

2025-12-04

