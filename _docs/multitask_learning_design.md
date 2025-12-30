# マルチタスク学習設計

## 目的

受け手予測（ノードレベル）とシュート予測（グラフレベル）を同時に学習することで、ノードレベルの表現とグラフレベルの表現の両方を最適化し、検索タスクに適した埋め込みを生成する。

## アーキテクチャ

```
Input Graph [B, N, input_dim]
    ↓
Shared Backbone (GATv2Network or GATv2Network4View)
    ↓
Node Embeddings [B, N, hidden_dim]
    ├─→ Receiver Head (NodeScoreHead)
    │   └─→ Receiver Logits [B, N] (or [N_attacking, 22] with filtering)
    │
    └─→ Graph Pooling (Mean)
        └─→ Graph Embedding [B, hidden_dim]
            └─→ Shot Head (ShotHeadNodeBased with mean pooling)
                └─→ Shot Logit [B, 1]
```

## 損失関数

```
Total Loss = λ_receiver × ReceiverLoss + λ_shot × ShotLoss
```

- **ReceiverLoss**: Cross-entropy loss for node-level classification (22 classes)
- **ShotLoss**: Binary cross-entropy loss for graph-level classification
- **λ_receiver, λ_shot**: Task weights (default: 1.0 each)

## 実装方針

### 1. MultiTaskModel クラス

- 共有バックボーン（GATv2Network or GATv2Network4View）
- 受け手予測ヘッド（ReceiverHead / NodeScoreHead）
- シュート予測ヘッド（ShotHeadNodeBased）

### 2. Forward メソッド

```python
def forward(self, x, edge_index, edge_attr, batch, ...):
    # Get node embeddings from backbone
    H = self.backbone(x, edge_index, edge_attr, batch=None)  # [N, hidden_dim]
    
    # Reshape to [B, N_per_graph, hidden_dim]
    H = H.view(B, N_per_graph, -1)
    
    # Receiver prediction (node-level)
    receiver_logits = self.receiver_head(H)  # [B, N, 22]
    
    # Shot prediction (graph-level)
    # Option 1: Mean pooling then shot head
    graph_emb = H.mean(dim=1)  # [B, hidden_dim]
    shot_logit = self.shot_head(graph_emb.unsqueeze(1)).squeeze(-1).squeeze(1)  # [B, 1]
    
    # Option 2: Shot head on nodes then mean pooling (current ShotHeadNodeBased design)
    shot_logits_per_node = self.shot_head(H)  # [B, N, 1]
    shot_logit = shot_logits_per_node.mean(dim=1).squeeze(-1)  # [B]
    
    return {
        'receiver_logits': receiver_logits,
        'shot_logit': shot_logit,
    }
```

### 3. 学習ループ

```python
for batch in dataloader:
    outputs = model(...)
    
    # Receiver loss (node-level)
    receiver_loss = criterion_receiver(outputs['receiver_logits'], receiver_targets)
    
    # Shot loss (graph-level)
    shot_loss = criterion_shot(outputs['shot_logit'], shot_targets)
    
    # Combined loss
    total_loss = lambda_receiver * receiver_loss + lambda_shot * shot_loss
    
    total_loss.backward()
    optimizer.step()
```

## データセット

- **Receiver data**: `data/processed_ck/receiver_train/data.pickle`
- **Shot data**: `data/processed_ck/shot_train/data.pickle`
- 両方のデータセットを統合または交互に使用

## 利点

1. **ノードレベルの表現**: 受け手予測タスクにより、ノードレベルの埋め込みが最適化される
2. **グラフレベルの表現**: シュート予測タスクにより、グラフレベルの埋め込みの多様性が促進される
3. **検索タスクへの適用**: 両方の表現が学習されるため、検索タスクで使用可能な埋め込みが生成される

## 課題

1. **タスクの重み付け**: `λ_receiver` と `λ_shot` のバランスを調整する必要がある
2. **データの整合性**: Receiver data と Shot data が同じサンプルに対応しているか確認
3. **学習の安定性**: 2つのタスクを同時に学習する際の安定性を確保

## 実装ファイル

- `tacticai/train/train_multitask.py`: マルチタスク学習スクリプト
- `tacticai/models/multitask_model.py`: MultiTaskModel クラス
- `configs/multitask_receiver_shot.yaml`: 設定ファイル

