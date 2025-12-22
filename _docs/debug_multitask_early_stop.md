# マルチタスク学習が6エポックで終了する問題の調査

## 問題

D2あり・なしの両方でマルチタスク学習が6エポックで終了している。これは仕様ではない。

## 確認方法

### 1. ログファイルの確認

tmuxで実行した場合、ログファイルは以下の場所に保存される：

```bash
# 最新のログファイルを確認
ls -lht runs/receiver_shot/*.log | head -5

# 最新のログファイルの内容を確認（最後の50行）
tail -50 runs/receiver_shot/training_*.log | tail -1

# エラーメッセージを検索
grep -i "error\|exception\|traceback\|early stopping" runs/receiver_shot/*.log
```

### 2. CSVファイルの確認

```bash
# 最新のCSVファイルを確認
ls -lht runs/receiver_shot/*.csv | head -5

# CSVファイルの内容を確認（エポック数とメトリクス）
head -10 runs/receiver_shot/training_history_*.csv
```

### 3. tmuxセッションの確認

```bash
# tmuxセッション一覧
tmux ls

# 特定のセッションに入る
tmux attach -t <session-name>

# セッション内でログを確認
# Ctrl+B を押してから [ でスクロールモードに入る
```

## 考えられる原因

### 1. Early Stoppingが誤って発動

- **設定**: `patience=20`, `min_delta=0.0005`, `monitor="val_receiver_top3"`
- **問題**: 6エポックで終了するのは早すぎる（patience=20なので、最低でも20エポックは続くはず）
- **確認**: ログに "Early stopping triggered" が表示されているか

### 2. エラーが発生して終了

- **問題**: Python例外やCUDAエラーで終了している可能性
- **確認**: ログファイルにエラーメッセージやTracebackがあるか

### 3. メトリクスの問題

- **問題**: `val_receiver_top3`が正しく計算されていない、またはNaN/Infになっている
- **確認**: CSVファイルの`val_receiver_top3`列を確認

## デバッグ手順

### ステップ1: ログファイルの確認

```bash
# 最新のログファイルを見つける
LATEST_LOG=$(ls -t runs/receiver_shot/*.log 2>/dev/null | head -1)
echo "Latest log: $LATEST_LOG"

# 最後の100行を確認
tail -100 "$LATEST_LOG"
```

### ステップ2: エラーの検索

```bash
# エラー、例外、トレースバックを検索
grep -E "Error|Exception|Traceback|early stopping|stopped" "$LATEST_LOG"
```

### ステップ3: CSVファイルの確認

```bash
# 最新のCSVファイルを見つける
LATEST_CSV=$(ls -t runs/receiver_shot/*.csv 2>/dev/null | head -1)
echo "Latest CSV: $LATEST_CSV"

# エポック数とメトリクスを確認
wc -l "$LATEST_CSV"  # 行数（ヘッダー含む）
head -10 "$LATEST_CSV"  # 最初の10行
```

### ステップ4: 設定ファイルの確認

```bash
# 設定ファイルのearly_stopping設定を確認
grep -A 3 "early_stopping" configs/multitask_receiver_shot_d2.yaml
grep -A 3 "early_stopping" configs/multitask_receiver_shot_no_d2.yaml
```

## 修正案

### 修正案1: Early Stoppingの無効化（一時的）

問題を切り分けるため、一時的にEarly Stoppingを無効化：

```yaml
early_stopping:
  patience: 1000  # 実質的に無効化
  min_delta: 0.0
  monitor: "val_receiver_top3"
```

### 修正案2: ログレベルの向上

より詳細なログを出力するため、ログレベルを`DEBUG`に変更：

```yaml
log_level: "DEBUG"
```

### 修正案3: エラーハンドリングの追加

`train_multitask.py`にエラーハンドリングを追加して、エラーが発生した場合にログに記録する。

## 修正内容

### エラーハンドリングの追加

`train_multitask.py`に以下の修正を追加：

1. **エラーハンドリング**: エポックループ全体を`try-except`で囲み、エラーが発生した場合にログに記録
2. **monitor_keyの存在確認**: `monitor_key`が`val_metrics`に存在するか確認し、存在しない場合はエラーをログに記録
3. **NaN/Infチェック**: `monitor_metric`がNaN/Infでないかチェック
4. **詳細ログ**: 各フェーズ（訓練開始、訓練完了、検証開始、検証完了）にログを追加

これにより、次回実行時にエラーが発生した場合、詳細なエラーメッセージとトレースバックがログに記録されます。

## 次のステップ

1. **修正されたコードで再実行**: エラーハンドリングが追加されたコードで再実行し、ログファイルを確認
2. **エラーログの確認**: `runs/receiver_shot/training_*.log`にエラーメッセージやトレースバックが記録されているか確認
3. **問題の特定**: エラーメッセージから原因を特定し、適切な修正を実施

## 考えられる原因（修正後のコードで確認可能）

- `receiver_graphs`が0になり、`receiver_top3`がNaN/Infになる
- CUDA out of memoryエラー
- データローダーの問題（`cand_mask`が不正）
- その他の予期しないPython例外

