# SportsData Competition 2025：コーナーキック戦術解析（TacticAIベース）

## 概要

CK（Corner Kick）局面をグラフとして表現し、**受け手予測**・**シュート予測**・**類似CK戦術検索**を行う研究/実装です。  
提案手法は `my_method/` に実装されています（TacticAIベースの拡張）。

## 結果（要約）

### 受け手予測・シュート予測（例）

| モデル | 受け手TOP1 | 受け手TOP3 | シュート精度 | シュートAUC |
| --- | ---: | ---: | ---: | ---: |
| CDLなし | 0.2456 | 0.4386 | 0.5263 | 0.4963 |
| TacticAI | 0.3158 | 0.5263 | 0.5263 | 0.5618 |
| 提案手法 | **0.3333** | 0.5088 | 0.5263 | **0.5778** |

### 類似CK戦術検索（定性）

- **cos類似度**: 上位が同一CK位置（同一象限）に偏りやすい  
- **提案手法（構造類似）**: 別のCK位置（別象限）でも上位に入ることがあり、**位置座標だけでなく戦術構造を捉えている可能性**が示唆される  

詳細：`_docs/類似CK検索_改善まとめ_20260104.md`

#### 結果画面
![類似CK戦術検索結果1](https://github.com/Uzy03/TracticAI_SportsData_Competition_2025/blob/dev/_images/%E9%A1%9E%E4%BC%BCCK%E6%88%A6%E8%A1%93%E6%A4%9C%E7%B4%A2%E7%B5%90%E6%9E%9C.png)

## コマンド（最短）

### 0) 前処理（CKデータ）

```bash
python SoccerData/preprocess_ck_improved.py
```

### 1) 学習（my_method：マルチタスク）

```bash
# consistency（提案手法）
python -m my_method.train.train_multitask \
  --config configs_my_method/multitask_receiver_shot_d2_consistency_stable.yaml

# baseline（比較）
python -m my_method.train.train_multitask \
  --config configs_my_method/multitask_receiver_shot_d2_baseline_stable.yaml
```

### 2) 類似CK検索（index構築→可視化）

```bash
# index構築（※cos検索はindexが必要）
python scripts/build_retrieval_index_my_method.py \
  --config configs_my_method/multitask_receiver_shot_d2_consistency_stable.yaml

# 可視化（Streamlit）
python -m streamlit run my_method/retrieval/visualization/app.py
```

## ドキュメント（詳細はここ）

- **コマンド一覧（提案手法）**: `_docs/コマンド一覧_提案手法.md`
- **類似CK検索の改善まとめ**: `_docs/類似CK検索_改善まとめ_20260104.md`
- **ポスター用（TacticAIとの差分）**: `_docs/ポスター_提案手法_変更点まとめ_TacticAIとの差分.md`

