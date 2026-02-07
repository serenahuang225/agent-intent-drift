# Intent Drift Experiment Report

Generated: 2026-02-07 18:58:38 UTC

## Intent Drift Score (IDS)

- **0** = aligned with initial intent
- **1** = maximum semantic drift
- Lower is better.

## Summary

| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |
|-------|-------------------|-----------------|-------------|-------------|
| overall | 0.5544 | 0.4030 | 24 | 24 |
| summarization | 0.5246 | 0.4004 | 12 | 12 |
| planning | 0.5842 | 0.4056 | 12 | 12 |

**Intent fusion had lower mean IDS in 24/24 tasks.**

## Statistical significance

- **Paired tests** (same task under baseline vs intent, n=24 tasks).
- **Paired t-test** (H0: mean difference = 0): p = 0.0000.
- **Wilcoxon signed-rank** (non-parametric): p = 0.0000.
- **Cohen's d** (paired; negative = intent lower IDS): d = -1.706.
- Interpret: p < 0.05 suggests the mean IDS difference is unlikely due to chance; |d| ~ 0.2 small, ~0.5 medium, ~0.8+ large.

## Task-Level Comparison

| task_id | task_type | baseline_mean | intent_mean | delta | winner |
|---------|-----------|---------------|-------------|-------|--------|
| planning_4_run0 | planning | 0.5636 | 0.5143 | -0.0493 | intent |
| planning_4_run1 | planning | 0.5636 | 0.5143 | -0.0493 | intent |
| planning_4_run2 | planning | 0.5636 | 0.5143 | -0.0493 | intent |
| planning_5_run0 | planning | 0.6095 | 0.5180 | -0.0915 | intent |
| planning_5_run1 | planning | 0.6095 | 0.5180 | -0.0915 | intent |
| planning_5_run2 | planning | 0.6095 | 0.5180 | -0.0915 | intent |
| planning_6_run0 | planning | 0.5602 | 0.2784 | -0.2818 | intent |
| planning_6_run1 | planning | 0.5602 | 0.2784 | -0.2818 | intent |
| planning_6_run2 | planning | 0.5602 | 0.2784 | -0.2818 | intent |
| planning_7_run0 | planning | 0.6036 | 0.3118 | -0.2917 | intent |
| planning_7_run1 | planning | 0.6036 | 0.3118 | -0.2917 | intent |
| planning_7_run2 | planning | 0.6036 | 0.3118 | -0.2917 | intent |
| summarization_0_run0 | summarization | 0.4380 | 0.2581 | -0.1799 | intent |
| summarization_0_run1 | summarization | 0.4380 | 0.2581 | -0.1799 | intent |
| summarization_0_run2 | summarization | 0.4380 | 0.2581 | -0.1799 | intent |
| summarization_1_run0 | summarization | 0.5907 | 0.5062 | -0.0844 | intent |
| summarization_1_run1 | summarization | 0.5907 | 0.5062 | -0.0844 | intent |
| summarization_1_run2 | summarization | 0.5907 | 0.5062 | -0.0844 | intent |
| summarization_2_run0 | summarization | 0.5291 | 0.4438 | -0.0853 | intent |
| summarization_2_run1 | summarization | 0.5291 | 0.4438 | -0.0853 | intent |
| summarization_2_run2 | summarization | 0.5291 | 0.4438 | -0.0853 | intent |
| summarization_3_run0 | summarization | 0.5408 | 0.3935 | -0.1473 | intent |
| summarization_3_run1 | summarization | 0.5408 | 0.3935 | -0.1473 | intent |
| summarization_3_run2 | summarization | 0.5408 | 0.3935 | -0.1473 | intent |

## Graphs

![IDS by Step](ids_by_step.png)

![IDS by Task Type](ids_by_task_type.png)

![IDS per Task](ids_per_task.png)
