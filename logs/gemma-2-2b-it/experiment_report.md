# Intent Drift Experiment Report

Generated: 2026-02-09 08:11:44 UTC

## Primary metric: IDS (Intent Drift Score)

- Per-step divergence of output from current inferred goal. **0** = aligned, **1** = max drift. Lower is better.

## Secondary metric: Goal shift

- Task-level cumulative goal shift (initial vs final intent). Lower is better.

## Summary

| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |
|-------|-------------------|-----------------|-------------|-------------|
| overall | 0.4656 | 0.4371 | 15 | 24 |
| summarization | 0.5049 | 0.5372 | 3 | 12 |
| planning | 0.4262 | 0.3370 | 12 | 12 |

**Intent fusion had lower mean IDS in 15/24 tasks.**

## Statistical significance

- **Paired tests** (same task under baseline vs intent, n=24 tasks).
- **Paired t-test** (H0: mean difference = 0): p = 0.0592.
- **Wilcoxon signed-rank** (non-parametric): p = 0.0491.
- **Cohen's d** (paired; negative = intent lower IDS): d = -0.405.
- Interpret: p < 0.05 suggests the mean IDS difference is unlikely due to chance; |d| ~ 0.2 small, ~0.5 medium, ~0.8+ large.

## Task-Level Comparison

| task_id | task_type | baseline_mean_ids | intent_mean_ids | delta_ids | winner |
|---------|-----------|-------------------|-----------------|-----------|---|
| planning_4_run0 | planning | 0.3619 | 0.2753 | -0.0866 | intent |
| planning_4_run1 | planning | 0.3619 | 0.2753 | -0.0866 | intent |
| planning_4_run2 | planning | 0.3619 | 0.2753 | -0.0866 | intent |
| planning_5_run0 | planning | 0.4590 | 0.3820 | -0.0769 | intent |
| planning_5_run1 | planning | 0.4590 | 0.3820 | -0.0769 | intent |
| planning_5_run2 | planning | 0.4590 | 0.3820 | -0.0769 | intent |
| planning_6_run0 | planning | 0.4562 | 0.3902 | -0.0660 | intent |
| planning_6_run1 | planning | 0.4562 | 0.3902 | -0.0660 | intent |
| planning_6_run2 | planning | 0.4562 | 0.3902 | -0.0660 | intent |
| planning_7_run0 | planning | 0.4278 | 0.3006 | -0.1271 | intent |
| planning_7_run1 | planning | 0.4278 | 0.3006 | -0.1271 | intent |
| planning_7_run2 | planning | 0.4278 | 0.3006 | -0.1271 | intent |
| summarization_0_run0 | summarization | 0.5566 | 0.6163 | +0.0597 | baseline |
| summarization_0_run1 | summarization | 0.5566 | 0.6163 | +0.0597 | baseline |
| summarization_0_run2 | summarization | 0.5566 | 0.6163 | +0.0597 | baseline |
| summarization_1_run0 | summarization | 0.5104 | 0.4931 | -0.0173 | intent |
| summarization_1_run1 | summarization | 0.5104 | 0.4931 | -0.0173 | intent |
| summarization_1_run2 | summarization | 0.5104 | 0.4931 | -0.0173 | intent |
| summarization_2_run0 | summarization | 0.4229 | 0.4298 | +0.0069 | baseline |
| summarization_2_run1 | summarization | 0.4229 | 0.4298 | +0.0069 | baseline |
| summarization_2_run2 | summarization | 0.4229 | 0.4298 | +0.0069 | baseline |
| summarization_3_run0 | summarization | 0.5298 | 0.6097 | +0.0799 | baseline |
| summarization_3_run1 | summarization | 0.5298 | 0.6097 | +0.0799 | baseline |
| summarization_3_run2 | summarization | 0.5298 | 0.6097 | +0.0799 | baseline |

## Graphs

![IDS by Step](ids_by_step.png)

![IDS by Task Type](ids_by_task_type.png)

![IDS per Task](ids_per_task.png)
