# Intent Drift Experiment Report

Generated: 2026-02-09 08:14:15 UTC

## Primary metric: IDS (Intent Drift Score)

- Per-step divergence of output from current inferred goal. **0** = aligned, **1** = max drift. Lower is better.

## Secondary metric: Goal shift

- Task-level cumulative goal shift (initial vs final intent). Lower is better.

## Summary

| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |
|-------|-------------------|-----------------|-------------|-------------|
| overall | 0.5403 | 0.5091 | 9 | 12 |
| summarization | 0.5403 | 0.5091 | 9 | 12 |

**Intent fusion had lower mean IDS in 9/12 tasks.**

## Statistical significance

- **Paired tests** (same task under baseline vs intent, n=12 tasks).
- **Paired t-test** (H0: mean difference = 0): p = 0.0031.
- **Wilcoxon signed-rank** (non-parametric): p = 0.0068.
- **Cohen's d** (paired; negative = intent lower IDS): d = -1.086.
- Interpret: p < 0.05 suggests the mean IDS difference is unlikely due to chance; |d| ~ 0.2 small, ~0.5 medium, ~0.8+ large.

## Task-Level Comparison

| task_id | task_type | baseline_mean_ids | intent_mean_ids | delta_ids | winner |
|---------|-----------|-------------------|-----------------|-----------|---|
| summarization_0_run0 | summarization | 0.5992 | 0.5251 | -0.0740 | intent |
| summarization_0_run1 | summarization | 0.5992 | 0.5251 | -0.0740 | intent |
| summarization_0_run2 | summarization | 0.5992 | 0.5251 | -0.0740 | intent |
| summarization_1_run0 | summarization | 0.5680 | 0.5371 | -0.0310 | intent |
| summarization_1_run1 | summarization | 0.5680 | 0.5371 | -0.0310 | intent |
| summarization_1_run2 | summarization | 0.5680 | 0.5371 | -0.0310 | intent |
| summarization_2_run0 | summarization | 0.4397 | 0.4178 | -0.0219 | intent |
| summarization_2_run1 | summarization | 0.4397 | 0.4178 | -0.0219 | intent |
| summarization_2_run2 | summarization | 0.4397 | 0.4178 | -0.0219 | intent |
| summarization_3_run0 | summarization | 0.5545 | 0.5565 | +0.0020 | baseline |
| summarization_3_run1 | summarization | 0.5545 | 0.5565 | +0.0020 | baseline |
| summarization_3_run2 | summarization | 0.5545 | 0.5565 | +0.0020 | baseline |

## Graphs

![IDS by Step](ids_by_step.png)

![IDS by Task Type](ids_by_task_type.png)

![IDS per Task](ids_per_task.png)
