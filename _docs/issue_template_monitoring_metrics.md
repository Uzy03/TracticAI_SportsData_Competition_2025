# Receiver予測モデルの詳細監視指標の実装

## 目的・背景

現在、Receiver予測モデルの基本的な精度指標（Accuracy, Top-k Accuracy）は実装されていますが、モデルの詳細な動作を理解し、潜在的な問題を早期に発見するための追加の監視指標が必要です。

これらの指標は、以下の目的で使用されます：
- モデルの予測行動の詳細な分析
- データの偏りやバイアスの検出
- 候補数やデータ特性による性能の偏りの特定
- 将来の問題が発生した際の診断ツールとしての活用

## 実装すべき監視項目

### 1. ターゲットの順位（rank）分布

**説明:**
- 各サンプルにおいて、予測ロジットをソートした時に正解ターゲットが何位に来るかを記録
- 1位なら rank=1、2位なら rank=2、...、候補外なら rank > N（Nは候補数）

**計算方法:**
```python
# 候補ロジットをソート
sorted_indices = torch.argsort(logits_b, descending=True)
# ターゲットのローカルIDの順位を取得
target_rank = (sorted_indices == cand_target_idx).nonzero(as_tuple=True)[0].item() + 1
```

**ログ形式:**
- エポックごとに `rank_of_target` の分布を記録
- ヒストグラム: `{1: count1, 2: count2, ..., '>N': count_out}`
- 統計: 平均rank、中央値rank、最小rank、最大rank

**期待される効果:**
- モデルが正解をどれだけ上位に予測しているかの可視化
- ランダム予測なら平均 rank ≈ N/2、良い予測なら rank ≈ 1 が多いことを確認
- 順位分布が偏っている場合（例: 常に3位以下）の問題検出

### 2. ターゲットのcand位置ヒストグラム（ローカルIDの分布）

**説明:**
- 候補内でのターゲットのローカルID（0-indexed位置）の分布を記録
- 例: 候補が `[3, 5, 7, 10]` でターゲットが `7` の場合、ローカルIDは `2`

**計算方法:**
```python
# 既に train_epoch で計算済み: cand_target_idx
# この値をヒストグラムとして集計
```

**ログ形式:**
- エポックごとに `cand_target_local_id` のヒストグラムを記録
- 各ローカルID（0, 1, 2, ..., N-1）の出現頻度
- 統計: 一様分布からの乖離度（カイ二乗検定など）

**期待される効果:**
- 候補選択にバイアスがないかの確認
- 例: 常に候補の先頭（ローカルID=0）がターゲットになっている場合は、データ生成に問題がある可能性
- 理想的には一様分布に近い分布であることを確認

### 3. cand個数 vs 精度の相関

**説明:**
- 候補数（Ncand）ごとに精度をグループ化して記録
- 候補数が少ない場合と多い場合で精度がどう変わるかを確認

**計算方法:**
```python
# 候補数ごとにサンプルをグループ化
# cand_counts = [Ncand_b for each graph in batch]
# 各グループで精度を計算
for Ncand in range(1, max_cand_count + 1):
    mask = (cand_counts == Ncand)
    if mask.any():
        accuracy_for_Ncand = compute_accuracy(predictions[mask], targets[mask])
```

**ログ形式:**
- エポックごとに `cand_count_vs_accuracy` を記録
- 形式: `{3: 0.85, 5: 0.78, 8: 0.72, 10: 0.65, ...}` （候補数: 精度）
- グラフ: 候補数（x軸）vs 精度（y軸）のプロット

**期待される効果:**
- 候補数による性能の偏りを検出
- 例: 候補数が10以上になると精度がランダム（1/N）に近づく場合は、モデルの限界を示唆
- 候補数が3以下の場合のみ精度が高い場合は、過学習の可能性を示唆
- データセットの特性（候補数の分布）と性能の関係を理解

## 実装場所

以下のファイルに実装することを想定：

- `tacticai/train/train_receiver.py`
  - `train_epoch()` 関数内
  - `validate_epoch()` 関数内

## 実装の詳細

### データ収集

各バッチで以下のデータを収集：

```python
# train_epoch / validate_epoch 内で
rank_distribution = []  # List of ranks
cand_local_id_distribution = []  # List of local IDs
cand_count_accuracy = {}  # Dict: {Ncand: [correct, total], ...}

for b in range(B):
    # ... 既存のコード ...
    
    # 1. Rank計算
    sorted_indices = torch.argsort(logits_b, descending=True)
    target_rank = (sorted_indices == cand_target_idx).nonzero(as_tuple=True)[0].item() + 1
    rank_distribution.append(target_rank)
    
    # 2. Local ID記録（既に計算済み: cand_target_idx）
    cand_local_id_distribution.append(cand_target_idx)
    
    # 3. Cand count別の精度記録
    Ncand_b = Ncand
    if Ncand_b not in cand_count_accuracy:
        cand_count_accuracy[Ncand_b] = [0, 0]
    is_correct = (predicted_idx == cand_target_idx).item()
    cand_count_accuracy[Ncand_b][0] += is_correct
    cand_count_accuracy[Ncand_b][1] += 1
```

### ログ出力

エポック終了時に統計を計算してログ出力：

```python
# Rank分布
rank_hist = Counter(rank_distribution)
rank_stats = {
    'mean': np.mean(rank_distribution),
    'median': np.median(rank_distribution),
    'min': np.min(rank_distribution),
    'max': np.max(rank_distribution),
    'histogram': dict(rank_hist)
}

# Local ID分布
local_id_hist = Counter(cand_local_id_distribution)
local_id_stats = {
    'histogram': dict(local_id_hist),
    'uniformity_test': chi2_test(local_id_distribution)  # オプション
}

# Cand count別精度
cand_count_acc = {
    Ncand: correct / total 
    for Ncand, (correct, total) in cand_count_accuracy.items()
}
```

## 出力形式

### ログファイルへの出力

エポックごとに以下の形式でログ出力：

```
[EPOCH-STATS] Rank distribution: mean=2.3, median=1, min=1, max=15
[EPOCH-STATS] Rank histogram: {1: 450, 2: 180, 3: 120, ...}
[EPOCH-STATS] Cand local ID distribution: {0: 200, 1: 180, 2: 220, ...}
[EPOCH-STATS] Cand count vs accuracy: {3: 0.85, 5: 0.78, 8: 0.72, 10: 0.65, 12: 0.58}
```

### CSVファイルへの出力（オプション）

詳細な分析のため、CSVファイルに出力することも検討：

- `runs/training_rank_distribution_YYYYMMDD_HHMMSS.csv`
- `runs/training_cand_local_id_distribution_YYYYMMDD_HHMMSS.csv`
- `runs/training_cand_count_accuracy_YYYYMMDD_HHMMSS.csv`

## 優先度

**低（Nice to have）**

- 現在のモデルは正常に動作しており、基本的な精度指標で十分
- 将来的な問題診断や詳細分析に有用だが、緊急性は低い
- モデルの動作をより深く理解したい場合に実装を検討

## 関連issue/PR

- （過去のissueへの参照があれば）
- （関連するPRがあれば）

## 補足

これらの監視項目は、過去に「精度がランダムになる問題」の診断のために提案されましたが、その問題は既に解決されています（`cand_counts`の累積問題の修正、データ整合性チェックの追加など）。

現在は、モデルの詳細な動作を理解し、将来の問題を早期に発見するための診断ツールとしての位置づけです。

