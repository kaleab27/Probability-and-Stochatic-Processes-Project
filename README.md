# COE–1: Stochastic Traffic Modeling for Cloud Resource Allocation

This repository contains a trace-driven simulation project for cloud autoscaling using:
- Poisson baselines (conceptual baseline)
- MDP/Q-learning autoscaling
- Google Cluster Trace (2011) and Alibaba OpenB pod trace (Kaggle)

## Contents
- `src/`: preprocessing + simulation + plotting scripts
- `main.tex`: LaTeX report
- `refs.bib`: references

## Data (not included)
Raw datasets are NOT included due to size/licensing.
You must download them yourself:
- Google Cluster Trace 2011 v2: https://github.com/google/cluster-data
- Alibaba OpenB pod trace (Kaggle): https://www.kaggle.com/datasets/nimishgsk17/alibaba-full

## Reproduce (high level)
1. Create venv + install requirements (duckdb, pandas, pyarrow, numpy, matplotlib)
2. Google preprocessing:
   - `python src/build_tasks_google.py`
   - `python src/pick_and_slice_2h.py`
3. Google training + experiments:
   - `python src/train_mdp.py`
   - `python src/run_experiments.py`
4. Alibaba preprocessing:
   - `python src/build_tasks_alibaba_openb.py --cpu_cap_milli 32000 --mem_cap_mib 262144`
   - `python src/pick_and_slice_24h_alibaba.py`
5. Alibaba training + experiments:
   - `python src/train_mdp_alibaba.py`
   - `python src/run_experiments_alibaba.py`
