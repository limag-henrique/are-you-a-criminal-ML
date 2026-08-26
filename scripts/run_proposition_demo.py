"""Generate the synthetic witness table for Proposition 1."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from face_profile_ml.endogenous_target import EndogenousProposition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="512")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k-values", default="2,4,8")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--out-dir", default="artifacts/proposition")
    args = parser.parse_args()
    dimensions = [int(value) for value in args.dimensions.split(",")]
    k_values = [int(value) for value in args.k_values.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = [
        asdict(EndogenousProposition(d=d, n=args.n, k=k, seed=seed).run())
        for d in dimensions for k in k_values for seed in seeds
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "proposition_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
