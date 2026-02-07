# Intent Drift Experiment Report

Generated: 2026-02-06 05:50:41 UTC

## Intent Drift Score (IDS)

- **0** = aligned with initial intent
- **1** = maximum semantic drift
- Lower is better.

## Summary

| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |
|-------|-------------------|-----------------|-------------|-------------|
| overall | 0.5225 | 0.4040 | 15 | 24 |
| summarization | 0.5697 | 0.3903 | 12 | 12 |
| planning | 0.4753 | 0.4177 | 3 | 12 |

**Intent fusion had lower mean IDS in 15/24 tasks.**

## Task-Level Comparison

| task_id | task_type | baseline_mean | intent_mean | delta | winner |
|---------|-----------|---------------|-------------|-------|--------|
| planning_4_run0 | planning | 0.3855 | 0.3897 | +0.0042 | baseline |
| planning_4_run1 | planning | 0.3855 | 0.3897 | +0.0042 | baseline |
| planning_4_run2 | planning | 0.3855 | 0.3897 | +0.0042 | baseline |
| planning_5_run0 | planning | 0.5787 | 0.6003 | +0.0216 | baseline |
| planning_5_run1 | planning | 0.5787 | 0.6003 | +0.0216 | baseline |
| planning_5_run2 | planning | 0.5787 | 0.6003 | +0.0216 | baseline |
| planning_6_run0 | planning | 0.5603 | 0.2530 | -0.3073 | intent |
| planning_6_run1 | planning | 0.5603 | 0.2530 | -0.3073 | intent |
| planning_6_run2 | planning | 0.5603 | 0.2530 | -0.3073 | intent |
| planning_7_run0 | planning | 0.3769 | 0.4277 | +0.0508 | baseline |
| planning_7_run1 | planning | 0.3769 | 0.4277 | +0.0508 | baseline |
| planning_7_run2 | planning | 0.3769 | 0.4277 | +0.0508 | baseline |
| summarization_0_run0 | summarization | 0.3652 | 0.2676 | -0.0975 | intent |
| summarization_0_run1 | summarization | 0.3652 | 0.2676 | -0.0975 | intent |
| summarization_0_run2 | summarization | 0.3652 | 0.2676 | -0.0975 | intent |
| summarization_1_run0 | summarization | 0.8542 | 0.4750 | -0.3792 | intent |
| summarization_1_run1 | summarization | 0.8542 | 0.4750 | -0.3792 | intent |
| summarization_1_run2 | summarization | 0.8542 | 0.4750 | -0.3792 | intent |
| summarization_2_run0 | summarization | 0.4524 | 0.3095 | -0.1429 | intent |
| summarization_2_run1 | summarization | 0.4524 | 0.3095 | -0.1429 | intent |
| summarization_2_run2 | summarization | 0.4524 | 0.3095 | -0.1429 | intent |
| summarization_3_run0 | summarization | 0.6069 | 0.5090 | -0.0979 | intent |
| summarization_3_run1 | summarization | 0.6069 | 0.5090 | -0.0979 | intent |
| summarization_3_run2 | summarization | 0.6069 | 0.5090 | -0.0979 | intent |

## Graphs

![IDS by Step](ids_by_step.png)

![IDS by Task Type](ids_by_task_type.png)

![IDS per Task](ids_per_task.png)
