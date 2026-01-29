import duckdb

INP = "data/processed/google_tasks_clean.parquet"
OUT = "data/processed/google_tasks_2h.parquet"

WINDOW_S = 2 * 3600
BIN_S = 60  # 1-minute bins to find a busy region

con = duckdb.connect()
con.execute(f"CREATE OR REPLACE VIEW tasks AS SELECT * FROM read_parquet('{INP}')")

# Count arrivals per minute
con.execute(f"""
CREATE OR REPLACE TABLE arrivals_per_min AS
SELECT
  CAST(FLOOR(arrival_time_s / {BIN_S}) AS BIGINT) AS minute,
  COUNT(*) AS n
FROM tasks
GROUP BY 1
ORDER BY 1;
""")

# Sliding sum over last 120 minutes -> pick max
rows = con.execute("""
SELECT
  minute,
  SUM(n) OVER (
    ORDER BY minute
    ROWS BETWEEN 119 PRECEDING AND CURRENT ROW
  ) AS n_in_2h
FROM arrivals_per_min
ORDER BY n_in_2h DESC
LIMIT 1;
""").fetchone()

best_minute = int(rows[0])
t1 = (best_minute + 1) * BIN_S
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
print("Best 2h window:", t0, "to", t1, "seconds")
print("Tasks in window:", n)
print("Wrote:", OUT)
