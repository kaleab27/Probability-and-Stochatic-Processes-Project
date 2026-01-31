import duckdb

INP = "data/processed/alibaba/alibaba_tasks_clean.parquet"
OUT = "data/processed/alibaba/alibaba_tasks_24h.parquet"

WINDOW_S = 24 * 3600
BIN_S = 300  # 5-minute bins (smoother for sparse traces)

con = duckdb.connect()
con.execute(f"CREATE OR REPLACE VIEW tasks AS SELECT * FROM read_parquet('{INP}')")

con.execute(f"""
CREATE OR REPLACE TABLE arrivals_per_bin AS
SELECT
  CAST(FLOOR(arrival_time_s / {BIN_S}) AS BIGINT) AS b,
  COUNT(*) AS n
FROM tasks
GROUP BY 1
ORDER BY 1;
""")

# Sliding 24h sum: WINDOW_S / BIN_S bins
win_bins = WINDOW_S // BIN_S
row = con.execute(f"""
SELECT
  b,
  SUM(n) OVER (
    ORDER BY b
    ROWS BETWEEN {win_bins - 1} PRECEDING AND CURRENT ROW
  ) AS n_in_win
FROM arrivals_per_bin
ORDER BY n_in_win DESC
LIMIT 1;
""").fetchone()

best_b = int(row[0])
t1 = (best_b + 1) * BIN_S
t0 = t1 - WINDOW_S

con.execute(f"""
COPY (
  SELECT arrival_time_s, runtime_s, cpu_req, mem_req
  FROM tasks
  WHERE arrival_time_s >= {t0}
    AND arrival_time_s < {t1}
  ORDER BY arrival_time_s
) TO '{OUT}' (FORMAT PARQUET);
""")

n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
print("Best 24h window:", t0, "to", t1, "seconds")
print("Tasks in window:", n)
print("Wrote:", OUT)
