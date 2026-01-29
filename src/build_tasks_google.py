import duckdb

RAW_GLOB = "data/raw/*.csv.gz"
OUT = "data/processed/google_tasks_clean.parquet"

# Google 2011 task_events mapping (confirmed by your inspection)
# column00: time (microseconds)
# column02: job_id
# column03: task_index
# column05: event_type
# column09: cpu_request
# column10: mem_request
TIME_SCALE = 1_000_000.0  # microseconds -> seconds

con = duckdb.connect()
con.execute("PRAGMA threads=4;")  # you can increase if you want (e.g., 8)

con.execute(f"""
CREATE OR REPLACE VIEW task_events_raw AS
SELECT * FROM read_csv_auto('{RAW_GLOB}', header=false);
""")

con.execute(f"""
CREATE OR REPLACE VIEW task_events_s AS
SELECT
  (column00::BIGINT / {TIME_SCALE})::DOUBLE AS time_s,
  column02::BIGINT AS job_id,
  column03::BIGINT AS task_index,
  column05::INT    AS event_type,
  TRY_CAST(column09 AS DOUBLE) AS cpu_req,
  TRY_CAST(column10 AS DOUBLE) AS mem_req
FROM task_events_raw;
""")

# Event types we use: SUBMIT=0, SCHEDULE=1, FINISH=4
con.execute("""
CREATE OR REPLACE TABLE tasks_clean AS
WITH
submit AS (
  SELECT job_id, task_index, MIN(time_s) AS submit_time_s
  FROM task_events_s
  WHERE event_type = 0
  GROUP BY job_id, task_index
),
sched AS (
  SELECT e.job_id, e.task_index, MIN(e.time_s) AS start_time_s
  FROM task_events_s e
  JOIN submit s
    ON e.job_id = s.job_id AND e.task_index = s.task_index
  WHERE e.event_type = 1 AND e.time_s >= s.submit_time_s
  GROUP BY e.job_id, e.task_index
),
finish AS (
  SELECT e.job_id, e.task_index, MIN(e.time_s) AS end_time_s
  FROM task_events_s e
  JOIN sched sc
    ON e.job_id = sc.job_id AND e.task_index = sc.task_index
  WHERE e.event_type = 4 AND e.time_s >= sc.start_time_s
  GROUP BY e.job_id, e.task_index
),
req AS (
  -- Take max request observed on SUBMIT/SCHEDULE; drop tasks with missing reqs
  SELECT
    job_id, task_index,
    MAX(cpu_req) AS cpu_req,
    MAX(mem_req) AS mem_req
  FROM task_events_s
  WHERE event_type IN (0, 1)
  GROUP BY job_id, task_index
)
SELECT
  s.submit_time_s AS arrival_time_s,
  sc.start_time_s,
  f.end_time_s,
  (f.end_time_s - sc.start_time_s) AS runtime_s,
  r.cpu_req,
  r.mem_req
FROM submit s
JOIN sched sc USING (job_id, task_index)
JOIN finish f USING (job_id, task_index)
JOIN req r USING (job_id, task_index)
WHERE
  runtime_s > 0
  AND runtime_s <= 86400
  AND cpu_req IS NOT NULL AND mem_req IS NOT NULL
  AND cpu_req > 0 AND cpu_req <= 1
  AND mem_req > 0 AND mem_req <= 1;
""")

con.execute("CREATE OR REPLACE VIEW stats AS SELECT COUNT(*) AS n FROM tasks_clean;")
n = con.execute("SELECT n FROM stats").fetchone()[0]
tmin, tmax = con.execute(
    "SELECT MIN(arrival_time_s), MAX(arrival_time_s) FROM tasks_clean"
).fetchone()

con.execute(f"COPY tasks_clean TO '{OUT}' (FORMAT PARQUET);")

print("Wrote:", OUT)
print("Clean tasks:", n)
print("Arrival range (s):", tmin, "to", tmax)
