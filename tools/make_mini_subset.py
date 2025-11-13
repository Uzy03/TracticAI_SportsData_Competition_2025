# tools/make_mini_subset.py
import argparse
import pickle
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="path to source pickle (train)")
    parser.add_argument("--dst", required=True, help="path to output pickle (mini)")
    parser.add_argument("--n", type=int, default=16, help="number of samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as f:
        data = pickle.load(f)

    # data は list[dict] or dict 内 list など想定されるが、既存pickleと同じ構造を維持したまま
    # 先頭から or ランダムに N 件を抽出
    if isinstance(data, list):
        rng = random.Random(args.seed)
        if len(data) <= args.n:
            mini = data
        else:
            mini = rng.sample(data, args.n)
    elif isinstance(data, dict):
        # 典型: {"samples": [...], "...": ...}
        key = None
        for k, v in data.items():
            if isinstance(v, list):
                key = k
                break
        assert key is not None, "No list-like field found in dict pickle"
        rng = random.Random(args.seed)
        samples = data[key]
        if len(samples) <= args.n:
            mini_samples = samples
        else:
            mini_samples = rng.sample(samples, args.n)
        mini = dict(data)
        mini[key] = mini_samples
    else:
        raise ValueError(f"Unsupported pickle top type: {type(data)}")

    with open(dst, "wb") as f:
        pickle.dump(mini, f)

    print(f"Mini subset saved: {dst} (n={args.n})")

if __name__ == "__main__":
    main()

