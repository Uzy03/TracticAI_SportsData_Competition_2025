# SoccerData 前処理ガイド

このディレクトリには、実際のサッカーデータ（Jリーグトラッキングデータ）が含まれています。

## ディレクトリ構造

```
SoccerData/
├── 2023_data/          # 2023年シーズンのデータ
│   └── [試合ID]/       # 各試合ごとのディレクトリ
│       ├── play.csv    # プレイデータ（アクション情報）
│       ├── players.csv # 選手リスト
│       └── tracking.csv# トラッキングデータ
└── 2024_data/          # 2024年シーズンのデータ
    └── [試合ID]/
        ├── play.csv
        ├── players.csv
        └── tracking.csv
```

## データ形式

### play.csv
- 各プレイの詳細情報
- 主要カラム:
  - `フレーム番号`: トラッキングデータのフレーム番号
  - `ボールＸ`, `ボールＹ`: ボール位置
  - `位置座標X`, `位置座標Y`: アクション時の位置
  - `選手ID`: アクション実行選手
  - `アクションID`: アクション種別（29=ホームパス、30=アウェイパスなど）
  - `B1_選手ID`: 受け手選手ID

### players.csv
- 試合に出場した選手リスト
- 各選手の出場時間、ポジション情報

### tracking.csv
- フレーム毎のトラッキングデータ
- カラム:
  - `Frame`: フレーム番号
  - `HA`: ホーム/アウェイ（1=Home, 2=Away）
  - `SysTarget`: システムターゲット（7=ボール、それ以外=プレイヤー）
  - `No`: 背番号
  - `X`, `Y`: 位置座標（センチメートル）
  - `Speed`: 速度

## 前処理方法

### 基本的な前処理

```bash
cd SoccerData
python preprocess_soccer_data.py --task receiver
```

### オプション

```bash
python preprocess_soccer_data.py \
    --data_dir SoccerData \
    --output_dir ../data/processed \
    --task receiver \
    --split 0.7,0.15,0.15
```

### パラメータ

- `--data_dir`: データディレクトリのパス（デフォルト: `SoccerData`）
- `--output_dir`: 出力先ディレクトリ（デフォルト: `data/processed`）
- `--task`: タスクタイプ（`receiver`, `shot`, `cvae`）
- `--split`: 学習/検証/テスト分割比率（デフォルト: `0.7,0.15,0.15`）

## 出力形式

処理後、以下のディレクトリ構造で出力されます：

```
data/processed/
├── receiver_train/
│   ├── data.pickle    # 訓練データ
│   └── metadata.json  # メタデータ
├── receiver_val/
│   ├── data.pickle
│   └── metadata.json
└── receiver_test/
    ├── data.pickle
    └── metadata.json
```

### サンプル形式

各サンプルは以下の形式：

```python
{
    'x': np.array([...]),           # 22選手のX座標（正規化済み）
    'y': np.array([...]),           # 22選手のY座標（正規化済み）
    'vx': np.array([...]),          # X方向速度
    'vy': np.array([...]),          # Y方向速度
    'height': np.array([...]),      # 選手の身長
    'weight': np.array([...]),      # 選手の体重
    'ball': np.array([...]),        # ボール所持情報（0/1）
    'team': np.array([...]),        # チームID（0/1）
    'receiver_id': int,             # 受け手のインデックス
    'match_id': str,                # 試合ID
    'frame': int,                   # フレーム番号
    'action_type': int,             # アクション種別
}
```

## 注意事項

1. **座標系**: tracking.csvの座標はセンチメートル単位です
2. **フィールドサイズ**: 標準は105m × 68m
3. **プレイヤー数**: 常に22プレイヤーを想定
4. **フレーム対応**: play.csvのフレーム番号でtracking.csvを参照

## カスタマイズ

新的预处理スクリプトはモジュラー設計で、以下をカスタマイズできます：

- `extract_action_frames()`: 抽出するアクションタイプ
- `create_sample_from_action()`: サンプル作成ロジック
- 位置座標の正規化方法
- 受け手の検出ロジック

