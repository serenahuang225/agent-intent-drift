# Intent Drift Experiment Report

Generated: 2026-02-09 08:08:06 UTC

## Primary metric: IDS (Intent Drift Score)

- Per-step divergence of output from current inferred goal. **0** = aligned, **1** = max drift. Lower is better.

## Secondary metric: Goal shift

- Task-level cumulative goal shift (initial vs final intent). Lower is better.

## Summary

| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |
|-------|-------------------|-----------------|-------------|-------------|
| overall | 0.4325 | 0.3608 | 24 | 24 |
| summarization | 0.4534 | 0.3743 | 12 | 12 |
| planning | 0.4116 | 0.3472 | 12 | 12 |

**Intent fusion had lower mean IDS in 24/24 tasks.**

## Statistical significance

- **Paired tests** (same task under baseline vs intent, n=24 tasks).
- **Paired t-test** (H0: mean difference = 0): p = 0.0000.
- **Wilcoxon signed-rank** (non-parametric): p = 0.0000.
- **Cohen's d** (paired; negative = intent lower IDS): d = -1.698.
- Interpret: p < 0.05 suggests the mean IDS difference is unlikely due to chance; |d| ~ 0.2 small, ~0.5 medium, ~0.8+ large.

## Task-Level Comparison

| task_id | task_type | baseline_mean_ids | intent_mean_ids | delta_ids | winner |
|---------|-----------|-------------------|-----------------|-----------|---|
| planning_4_run0 | planning | 0.3774 | 0.3482 | -0.0292 | intent |
| planning_4_run1 | planning | 0.3774 | 0.3482 | -0.0292 | intent |
| planning_4_run2 | planning | 0.3774 | 0.3482 | -0.0292 | intent |
| planning_5_run0 | planning | 0.4402 | 0.3918 | -0.0484 | intent |
| planning_5_run1 | planning | 0.4402 | 0.3918 | -0.0484 | intent |
| planning_5_run2 | planning | 0.4402 | 0.3918 | -0.0484 | intent |
| planning_6_run0 | planning | 0.4407 | 0.3305 | -0.1102 | intent |
| planning_6_run1 | planning | 0.4407 | 0.3305 | -0.1102 | intent |
| planning_6_run2 | planning | 0.4407 | 0.3305 | -0.1102 | intent |
| planning_7_run0 | planning | 0.3882 | 0.3184 | -0.0698 | intent |
| planning_7_run1 | planning | 0.3882 | 0.3184 | -0.0698 | intent |
| planning_7_run2 | planning | 0.3882 | 0.3184 | -0.0698 | intent |
| summarization_0_run0 | summarization | 0.4935 | 0.3786 | -0.1150 | intent |
| summarization_0_run1 | summarization | 0.4935 | 0.3786 | -0.1150 | intent |
| summarization_0_run2 | summarization | 0.4935 | 0.3786 | -0.1150 | intent |
| summarization_1_run0 | summarization | 0.4978 | 0.3584 | -0.1395 | intent |
| summarization_1_run1 | summarization | 0.4978 | 0.3584 | -0.1395 | intent |
| summarization_1_run2 | summarization | 0.4978 | 0.3584 | -0.1395 | intent |
| summarization_2_run0 | summarization | 0.3698 | 0.3452 | -0.0246 | intent |
| summarization_2_run1 | summarization | 0.3698 | 0.3452 | -0.0246 | intent |
| summarization_2_run2 | summarization | 0.3698 | 0.3452 | -0.0246 | intent |
| summarization_3_run0 | summarization | 0.4524 | 0.4150 | -0.0374 | intent |
| summarization_3_run1 | summarization | 0.4524 | 0.4150 | -0.0374 | intent |
| summarization_3_run2 | summarization | 0.4524 | 0.4150 | -0.0374 | intent |

## Graphs

![IDS by Step](ids_by_step.png)

![IDS by Task Type](ids_by_task_type.png)

![IDS per Task](ids_per_task.png)
