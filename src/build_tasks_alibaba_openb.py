import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inp",
        default="data/raw/alibaba_kaggle/alibaba_full/openb_pod_list_default.csv",
    )
    p.add_argument(
        "--out",
        default="data/processed/alibaba/alibaba_tasks_clean.parquet",
    )
    p.add_argument(
        "--cpu_cap_milli",
        type=float,
        required=True,
        help="VM CPU capacity in millicores for normalization (e.g., 32000)",
    )
    p.add_argument(
        "--mem_cap_mib",
        type=float,
        required=True,
        help="VM memory capacity in MiB for normalization (e.g., 65536)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.inp)

    # Basic fields
    df = df.rename(
        columns={
            "creation_time": "arrival_time_s",
            "scheduled_time": "start_time_s",
            "deletion_time": "end_time_s",
            "cpu_milli": "cpu_milli",
            "memory_mib": "mem_mib",
        }
    )

    # Convert types
    for c in ["arrival_time_s", "start_time_s", "end_time_s", "cpu_milli", "mem_mib"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fallback: if scheduled_time is missing/0, treat start as arrival
    df.loc[(df["start_time_s"].isna()) | (df["start_time_s"] <= 0), "start_time_s"] = df[
        "arrival_time_s"
    ]

    df["runtime_s"] = df["end_time_s"] - df["start_time_s"]

    # Normalize CPU/mem into [0,1] fractions of a VM
    df["cpu_req"] = df["cpu_milli"] / float(args.cpu_cap_milli)
    df["mem_req"] = df["mem_mib"] / float(args.mem_cap_mib)

    # Filters similar to Google
    df = df[df["runtime_s"] > 0]
    df = df[df["runtime_s"] <= 86400]  # 1 day cap
    df = df[df["cpu_req"] > 0]
    df = df[df["mem_req"] > 0]

    # Keep only tasks that fit within one VM under our normalization
    df = df[(df["cpu_req"] <= 1.0) & (df["mem_req"] <= 1.0)]

    # Shift time so first arrival is at 0 for simulation convenience
    t0 = float(df["arrival_time_s"].min())
    df["arrival_time_s"] = df["arrival_time_s"] - t0

    out = df[["arrival_time_s", "runtime_s", "cpu_req", "mem_req"]].dropna()
    out = out.sort_values("arrival_time_s").reset_index(drop=True)

    out.to_parquet(args.out, index=False)
    print("Wrote:", args.out)
    print("Rows:", len(out))
    print("Arrival range (s):", out["arrival_time_s"].min(), "to", out["arrival_time_s"].max())
    print("Mean cpu_req:", out["cpu_req"].mean(), "Mean mem_req:", out["mem_req"].mean())


if __name__ == "__main__":
    main()
