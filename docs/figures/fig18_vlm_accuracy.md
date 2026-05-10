# Figure 18 — VLM Object Identification Accuracy (VLM Clustering Stage)

**Referenced in**: Section V-E  
**Type**: Accuracy table  
**Status**: ⚠️ Replace `—` cells with actual counts.

## Recommended IEEE Format

**Diagram type**: Grouped bar chart (pgfplots) — preferred, or `booktabs` table as fallback

Once counts are available, a **grouped bar chart** (three bars per category: Correct = green, Incorrect = red, Ambiguous = yellow) is the most IEEE-effective choice for an accuracy breakdown — reviewers can see the error composition at a glance without reading individual numbers. Place the Accuracy (%) value as a text annotation above each correct bar.

If a table is preferred for precision, use an IEEE `\begin{table}` with `booktabs` rules, the **Overall** row in bold, and the Accuracy column highlighted with `\columncolor{gray!12}`. Keep the count columns narrow and right-aligned.

**LaTeX approach (chart)**: `pgfplots` `ybar` grouped — `\addplot[fill=green!60] coordinates {...}` for Correct, `\addplot[fill=red!60]` for Incorrect, `\addplot[fill=yellow!60]` for Ambiguous. `\legend{Correct, Incorrect, Ambiguous}`. x-axis = category names (abbreviated if long). Fits in one column.

| Object category | Total clusters | Correct ✓ | Incorrect ✗ | Ambiguous / empty | Accuracy (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| Vehicle (car, van, truck) | — | — | — | — | — |
| Furniture (chair, desk, bin) | — | — | — | — | — |
| Structure (pillar, bollard, wall fragment) | — | — | — | — | — |
| Other | — | — | — | — | — |
| **Overall** | — | — | — | — | — |

> **Correct**: VLM label matches ground-truth annotation.  
> **Incorrect**: VLM label does not match (e.g., "chair" predicted for a car).  
> **Ambiguous / empty**: VLM returned `"empty"` or `"unknown"` for a cluster with a ground-truth label.
