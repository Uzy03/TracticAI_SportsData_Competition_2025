.PHONY: setup install dev test clean lint format type-check preprocess

# セットアップ
setup:
	python -m pip install --upgrade pip
	pip install -e .
	pip install -e ".[dev]"

# 開発用インストール
install:
	pip install -e ".[dev]"

# テスト実行
test:
	python -m pytest tests/ -v

# クリーンアップ
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "runs" -exec rm -rf {} +

# リンター実行
lint:
	ruff check tacticai/ tests/ scripts/

# フォーマット実行
format:
	black tacticai/ tests/ scripts/
	ruff check --fix tacticai/ tests/ scripts/

# 型チェック実行
type-check:
	mypy tacticai/

# 前処理実行
preprocess:
	python scripts/preprocess.py --input data/raw --out data/processed

# 開発用：全チェック実行
check-all: lint type-check test

# デバッグ用：最小テストデータでの学習
debug-train:
	python tacticai/train/train_receiver.py --config configs/receiver.yaml --debug_overfit
	python tacticai/train/train_shot.py --config configs/shot.yaml --debug_overfit
	python tacticai/train/train_cvae.py --config configs/cvae.yaml --debug_overfit

# サンプル実行
sample-run:
	python tacticai/train/train_receiver.py --config configs/receiver.yaml
	python tacticai/eval/eval_receiver.py --config configs/receiver.yaml
