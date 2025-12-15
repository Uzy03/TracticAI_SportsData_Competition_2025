# バックボーン保存機構の設計書

## 1. 概要

Receiver予測で学習したGNNバックボーン（エンコーダー）のみを抽出して保存する機能を実装する。
保存したバックボーンは、シュート予測や生成タスクでpretrainedモデルとして再利用できる。

## 2. 現状の確認

### 2.1 モデル構造
- `ReceiverModel` は `self.backbone` と `self.head` を持つ
- D2等変性使用時: `backbone = GATv2Network4View(...)`
- D2等変性不使用時: `backbone = GATv2Network(...)`
- 現在はモデル全体（backbone + head）が `save_checkpoint` で保存されている

### 2.2 チェックポイント保存場所
- 現在: `checkpoints/receiver/best.ckpt` （モデル全体）
- 追加予定:
  - `checkpoints/receiver/backbone_d2.ckpt` （D2等変性有効時のバックボーン）
  - `checkpoints/receiver/backbone_no_d2.ckpt` （D2等変性無効時のバックボーン）

## 3. 設計

### 3.1 保存する内容

#### 必須情報
1. **バックボーンのstate_dict**: `model.backbone.state_dict()`
2. **メタデータ**:
   - `input_dim`: ノード特徴次元
   - `hidden_dim`: 隠れ層次元
   - `num_layers`: GNNレイヤー数
   - `num_heads`: アテンションヘッド数
   - `dropout`: Dropout率
   - `edge_dim`: エッジ特徴次元
   - `use_d2_equivariance`: D2等変性使用フラグ（`True` or `False`）
   - `backbone_type`: `"GATv2Network4View"` or `"GATv2Network"`

#### オプション情報
- `training_config`: 学習時の設定（YAMLの一部）
- `checkpoint_epoch`: 元チェックポイントのエポック数
- `metrics`: 元チェックポイントの性能指標

### 3.2 保存形式

```python
backbone_checkpoint = {
    "backbone_state_dict": model.backbone.state_dict(),
    "metadata": {
        "input_dim": 16,
        "hidden_dim": 512,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.0,
        "edge_dim": 10,
        "use_d2_equivariance": True,
        "backbone_type": "GATv2Network4View",  # or "GATv2Network"
    },
    "checkpoint_info": {
        "source_checkpoint": "checkpoints/receiver/best.ckpt",
        "epoch": 200,
        "metrics": {...}  # オプション
    }
}
```

### 3.3 実装箇所

#### 3.3.1 新規関数: `save_backbone_checkpoint`
- **場所**: `tacticai/modules/utils.py`
- **シグネチャ**:
  ```python
  def save_backbone_checkpoint(
      model: nn.Module,
      metadata: Dict[str, Any],
      filepath: Union[str, Path],
      source_checkpoint: Optional[str] = None,
      epoch: Optional[int] = None,
      metrics: Optional[Dict[str, float]] = None,
  ) -> None:
      """Save backbone only checkpoint for transfer learning.
      
      Args:
          model: ReceiverModel instance with backbone attribute
          metadata: Model configuration metadata
          filepath: Path to save backbone checkpoint
          source_checkpoint: Path to original full checkpoint (optional)
          epoch: Epoch number of source checkpoint (optional)
          metrics: Performance metrics of source checkpoint (optional)
      """
  ```

#### 3.3.2 設定ファイルへの追加
- **ファイル**: `configs/receiver_full_simplified.yaml` など
- **追加項目**:
  ```yaml
  train:
    # ... existing config ...
    save_backbone: true  # バックボーンを保存するか
    backbone_save_path: null  # 保存パス（nullの場合は自動生成、D2有無に応じてbackbone_d2.ckpt または backbone_no_d2.ckpt）
  ```
  
  **注意**: `backbone_save_path` が指定されていない場合、D2等変性の有無に応じて自動的にファイル名を決定する

#### 3.3.3 train_receiver.pyへの統合
- **場所**: `tacticai/train/train_receiver.py`
- **変更箇所**: best model保存時（約2076行目と2086行目）
- **処理**:
  1. `save_backbone` 設定を確認
  2. `True` の場合、`save_backbone_checkpoint` を呼び出す
  3. メタデータを生成して渡す

### 3.4 実装の詳細

#### 3.4.1 メタデータの生成
```python
def _extract_backbone_metadata(model: ReceiverModel, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata for backbone checkpoint."""
    model_config = config["model"]
    d2_config = config.get("d2", {})
    
    metadata = {
        "input_dim": model_config["input_dim"],
        "hidden_dim": model_config["hidden_dim"],
        "num_layers": model_config["num_layers"],
        "num_heads": model_config["num_heads"],
        "dropout": model_config["dropout"],
        "edge_dim": model_config.get("edge_dim", 1),
        "use_d2_equivariance": d2_config.get("enabled", False),
        "backbone_type": "GATv2Network4View" if model.use_d2_equivariance else "GATv2Network",
    }
    return metadata
```

#### 3.4.2 保存処理の追加
best model保存時に以下を追加:
```python
# Best model保存後
if config.get("train", {}).get("save_backbone", False):
    backbone_path = config.get("train", {}).get("backbone_save_path")
    if backbone_path is None:
        # デフォルトパスを生成（D2有無に応じてファイル名を決定）
        use_d2 = config.get("d2", {}).get("enabled", False)
        backbone_filename = "backbone_d2.ckpt" if use_d2 else "backbone_no_d2.ckpt"
        backbone_path = Path(config.get("checkpoint_dir", "checkpoints")) / "receiver" / backbone_filename
    else:
        backbone_path = Path(backbone_path)
    
    metadata = _extract_backbone_metadata(model, config)
    save_backbone_checkpoint(
        model=model,
        metadata=metadata,
        filepath=backbone_path,
        source_checkpoint=str(checkpoint_path),
        epoch=epoch,
        metrics=val_metrics,  # or train_metrics if merge_val_to_train
    )
    logger.info(f"Backbone saved to {backbone_path} (D2 equivariance: {metadata['use_d2_equivariance']})")
```

### 3.5 エラーハンドリング

1. **backbone属性がない場合**: `AttributeError` を発生
2. **保存先ディレクトリがない場合**: 自動的に作成（`Path.mkdir(parents=True, exist_ok=True)`）
3. **D2等変性フラグと実際のモデルタイプの不一致**: 警告を出して続行

### 3.6 テストケース（検証項目）

1. D2等変性有効時の保存 → `backbone_d2.ckpt` が生成されること
2. D2等変性無効時の保存 → `backbone_no_d2.ckpt` が生成されること
3. 設定で `save_backbone: false` の場合、保存されないこと
4. 保存されたチェックポイントが正しくロードできること
5. メタデータが正しく保存されていること（特に `use_d2_equivariance` と `backbone_type`）
6. ファイル名にD2有無が反映されていること

## 4. 実装順序

1. `tacticai/modules/utils.py` に `save_backbone_checkpoint` 関数を追加
2. `tacticai/train/train_receiver.py` に `_extract_backbone_metadata` ヘルパー関数を追加
3. `tacticai/train/train_receiver.py` の best model保存処理にバックボーン保存を統合
4. 設定ファイルに `save_backbone` オプションを追加（デフォルト: `true`）
5. 動作確認

## 5. 今後の拡張

- 学習完了時にも自動保存するオプション（現在はbest modelのみ）
- 複数のエポックでのバックボーン保存（スナップショット機能）
- バックボーンのバージョン管理

