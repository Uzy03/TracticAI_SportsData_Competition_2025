import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect receiver pickle data")
    parser.add_argument("path", help="Path to data pickle")
    parser.add_argument("--n", type=int, default=300, help="Number of samples to inspect")
    args = parser.parse_args()

    with open(args.path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    else:
        samples = payload

    n = min(args.n, len(samples))
    if n == 0:
        print("No samples to inspect")
        return

    team_match = 0
    in_cand = 0
    cand_lens = []

    for item in samples[:n]:
        tgt_team = int(item.get("target_team", -1))
        kick_team = int(item.get("kicker_team", -1))
        tgt = int(item.get("target", -1))
        cand_ids = item.get("candidate_ids", [])

        team_match += int(tgt_team == kick_team)
        in_cand += int(tgt in cand_ids)
        cand_lens.append(len(cand_ids))

    print(f"Total checked: {n}")
    print(f"kicker_team==target_team: {team_match / n:.3f}")
    print(f"target in candidates     : {in_cand / n:.3f}")
    print(
        f"cand size: mean={np.mean(cand_lens):.2f}, min={np.min(cand_lens)}, max={np.max(cand_lens)}"
    )


if __name__ == "__main__":
    main()
