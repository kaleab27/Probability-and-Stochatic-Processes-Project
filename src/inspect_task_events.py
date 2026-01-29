import glob
import duckdb
import pandas as pd

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

files = sorted(glob.glob("data/raw/*.csv.gz"))
if not files:
    raise SystemExit("No .csv.gz files found in data/raw")

path = files[0]
con = duckdb.connect()

df = con.execute(
    f"SELECT * FROM read_csv_auto('{path}', header=false) LIMIT 5"
).fetchdf()

print("Sample file:", path)
print(df)
print("\nColumn names:", list(df.columns))
print("Column count:", len(df.columns))

tmin, tmax = con.execute(
    f"SELECT MIN(column00), MAX(column00) FROM read_csv_auto('{path}', header=false)"
).fetchone()
print("\nTime range raw:", tmin, "to", tmax)
