import argparse
import pandas as pd
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/processed/google_tasks_2h.parquet")
    p.add_argument("--out", default="data/processed/google_tasks_2h_sample.parquet")
    p.add_argument("--frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = pd.read_parquet(args.inp)
    df = df.sort_values("arrival_time_s").reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    keep = rng.random(len(df)) < args.frac
    sdf = df.loc[keep].copy()

    # keep chronological order
    sdf = sdf.sort_values("arrival_time_s").reset_index(drop=True)

    sdf.to_parquet(args.out, index=False)

    print("Input rows:", len(df))
    print("Sample rows:", len(sdf))
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
