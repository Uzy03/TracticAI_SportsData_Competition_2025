# シュート予測モデルの学習結果

## 概要

受け手予測で学習したGNNバックボーンを使用したシュート予測モデルの学習結果です。
- **モデル**: `ShotModelWithReceiver`（Pretrained backbone + Fine-tuning）
- **データ**: `data/processed_ck/shot_train/val/test/data.pickle`（約375サンプル）
- **タスク**: 2クラス分類（シュート発生: 0 or 1）
- **設定**: D2有効版とD2無効版の両方を評価

## 学習設定

### 共通設定
- **Pretrained backbone**: 受け手予測モデルから読み込み
- **Backbone fine-tuning**: `freeze_backbone: false`
- **Conditional prediction**: `use_receiver_for_conditioning: false`（直接予測）
- **Batch size**: 32
- **Epochs**: 100（Early stopping: patience=20）
- **Optimizer**: Adam
- **Loss**: Binary Cross Entropy
- **Metrics**: AUC-ROC, AUC-PR, Accuracy, F1

### D2無効版 (`shot_pretrained_backbone_no_d2.yaml`)
- **Learning rate**: `lr_backbone=0.0001`, `lr_head=0.0015`
- **Gradient clipping**: `max_norm=1.0`
- **Scheduler**: None
- **Weight decay**: 0.0001

### D2有効版 (`shot_pretrained_backbone_d2.yaml`)
- **Learning rate**: `lr_backbone=0.00005`, `lr_head=0.001`（より保守的）
- **Gradient clipping**: `max_norm=0.5`（より強いクリッピング）
- **Scheduler**: None
- **Weight decay**: 0.0001

## 学習結果

### D2無効版

**実行日時**: 2025-12-19 15:10:21  
**Config**: `configs/shot_pretrained_backbone_no_d2.yaml`

| Metric | Train (最終) | Val (Best) | Test (Best) |
|--------|--------------|------------|-------------|
| Loss | 0.6810 | 0.7030 | 0.7015 |
| AUC-ROC | 0.4871 | **0.6718** | **0.4580** |
| AUC-PR | 0.5580 | 0.7018 | 0.4847 |
| Accuracy | 0.5824 | 0.4727 | 0.5263 |
| F1 | 0.7361 | 0.6420 | 0.6897 |

**学習状況**:
- **Best Val AUC-ROC**: 0.6718（Epoch 12）
- **Test AUC-ROC**: 0.4580（Best model使用）
- **Early stopping**: Epoch 32で停止（20エポック改善なし）
- **学習傾向**: Train AUC-ROCが0.45-0.52の範囲で変動、安定していない

### D2有効版

**実行日時**: 2025-12-19 15:11:04  
**Config**: `configs/shot_pretrained_backbone_d2.yaml`

| Metric | Train (最終) | Val (Best) | Test (Best) |
|--------|--------------|------------|-------------|
| Loss | 0.6769 | 0.7230 | 0.7070 |
| AUC-ROC | 0.5168 | **0.6273** | **0.4370** |
| AUC-PR | 0.6125 | 0.5591 | 0.4934 |
| Accuracy | 0.5824 | 0.4727 | 0.5263 |
| F1 | 0.7361 | 0.6420 | 0.6897 |

**学習状況**:
- **Best Val AUC-ROC**: 0.6273（Epoch 5）
- **Test AUC-ROC**: 0.4370（Best model使用）
- **Early stopping**: Epoch 25で停止（20エポック改善なし）
- **学習傾向**: Train AUC-ROCが0.45-0.54の範囲で変動、D2無効版と同様に不安定

## 結果の比較

| 設定 | Best Val AUC-ROC | Test AUC-ROC | 学習エポック数 |
|------|------------------|--------------|----------------|
| D2無効 | 0.6718 | 0.4580 | 32 |
| D2有効 | 0.6273 | 0.4370 | 25 |

### 主な観察事項

1. **Test性能が低い**: 両方ともTest AUC-ROCが約0.44-0.46と、ランダムレベル（0.5）を下回っている
2. **Val性能との乖離**: Val AUC-ROCは0.62-0.67を記録しているが、Test性能は0.44-0.46と大きく乖離
3. **Train性能の不安定性**: Train AUC-ROCが0.45-0.55の範囲で大きく変動し、安定して向上していない
4. **D2の効果**: D2有効版はVal性能がやや低く（0.6273 vs 0.6718）、Test性能も低い（0.4370 vs 0.4580）

## 性能が低い理由（分析）

詳細は `_docs/shot_vs_receiver_analysis.md` を参照。

### 主な要因

1. **構造的情報の欠如**:
   - 受け手予測には候補マスク（cand_mask）があり、正解が22人の中の特定のサブセット（通常9-12人）に絞られる
   - シュート予測にはこのような構造的な制約がない

2. **タスクの抽象度**:
   - 受け手予測は「22人のプレイヤーから1人を選択」という具体的なタスク
   - シュート予測は「シュート発生/非発生」というより抽象的なタスク

3. **データ量の不足**:
   - 約375サンプルは、このような複雑なタスクには少なすぎる可能性がある
   - 受け手予測でもTest性能は低い（Top-1 Accuracy: 0.2281）が、Trainでは学習できている（Top-3: 0.82+）

4. **クラス不均衡**:
   - ラベル分布を確認した結果、クラス不均衡は主因ではない（Train: 45.6% vs 54.4%, Val: 52.7% vs 47.3%）

## 実装の詳細

### モデル構造

```
ShotModelWithReceiver:
  - Backbone: GATv2Network (D2無効) or GATv2Network4View (D2有効)
    - Pretrained weights from receiver prediction model
    - Fine-tuning enabled (lr_backbone << lr_head)
  - ReceiverHead: ReceiverHead (frozen, for reference only)
    - Pretrained weights from receiver prediction model
    - Not used when use_receiver_for_conditioning=false
  - ShotHead: ShotHeadNodeBased
    - Newly initialized, trained from scratch
```

### Forward処理（条件付き予測無効時）

1. **D2無効時**:
   ```
   H = backbone(x, edge_index, edge_attr)  # [B, N, hidden_dim]
   H_normalized = LayerNorm(H)
   shot_logits_per_node = shot_head(H_normalized)  # [B, N]
   shot_logits = shot_logits_per_node.mean(dim=1)  # [B] (mean pooling)
   ```

2. **D2有効時**:
   ```
   x_4view = apply_d2_transforms(x)  # 4 views
   H_4view = backbone(x_4view, edge_index, edge_attr)  # [B, 4, N, hidden_dim]
   H = H_4view.mean(dim=1)  # [B, N, hidden_dim] (average over 4 views)
   H_normalized = LayerNorm(H)
   shot_logits_per_node = shot_head(H_normalized)  # [B, N]
   shot_logits = shot_logits_per_node.mean(dim=1)  # [B]
   ```

## 今後の改善案

1. **データ拡張**: より多くのデータを収集する、または既存データの前処理を改善
2. **アーキテクチャの変更**: より複雑なReadout機構（Attention poolingなど）の導入
3. **条件付き予測の再検討**: `use_receiver_for_conditioning=true`として、受け手確率を活用した予測を試す
4. **ハイパーパラメータの調整**: より小さい学習率、より強い正則化など
5. **タスクの再設計**: より具体的なタスク（例：シュート発生確率の段階的予測）への変更

## 参考

- 設計図: `_docs/設計図.md`
- 受け手予測 vs シュート予測の分析: `_docs/shot_vs_receiver_analysis.md`
- コマンド一覧: `_docs/コマンド一覧.md`
- 学習ログ:
  - D2無効: `runs/shot/training_20251219_151021.log`
  - D2有効: `runs/shot/training_20251219_151104.log`
- CSV履歴:
  - D2無効: `runs/shot/training_history_20251219_151021.csv`
  - D2有効: `runs/shot/training_history_20251219_151104.csv`

