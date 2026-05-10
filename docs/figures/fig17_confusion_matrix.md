# Figure 17 — Confusion Matrix (Segmentation Stage)

**Referenced in**: Section V-D  
**Type**: Confusion matrix table  
**Status**: ⚠️ Replace `—` cells with actual counts or percentages.

## Recommended IEEE Format

**Diagram type**: Color-coded heatmap table (`colortbl` + `booktabs`)

Render the 5×5 confusion matrix as an IEEE `\begin{table}` where each cell is shaded with `\cellcolor{blue!NN}` — intensity `NN` proportional to the cell value (normalize each column by the true-class total so the diagonal reflects per-class recall directly). Diagonal cells (true positives) use the darkest shade; off-diagonal cells use progressively lighter shades. Print the raw count and percentage in each cell (e.g., `142\newline(94.7\%)`). Add Precision, Recall, F1, and IoU as appended rows below a `\midrule`.

**LaTeX approach**: `\usepackage{booktabs, xcolor, colortbl}`. Compute shading levels in advance: `\newcommand{\hc}[1]{\cellcolor{blue!#1}}`. The per-class summary block (Precision/Recall/F1/IoU) sits below a second `\midrule` as four extra rows. Use `\begin{table*}` to span both columns.

*Rows = predicted class · Columns = true class*

|  | **True: Ground** | **True: Wall** | **True: Ceiling** | **True: Objects** | **True: INIT** | **Precision** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pred: Ground** | — | — | — | — | — | — |
| **Pred: Wall** | — | — | — | — | — | — |
| **Pred: Ceiling** | — | — | — | — | — | — |
| **Pred: Objects** | — | — | — | — | — | — |
| **Pred: INIT** | — | — | — | — | — | — |
| **Recall** | — | — | — | — | — | |
| **F1** | — | — | — | — | — | |
| **IoU** | — | — | — | — | — | |

## Per-class summary

| Class | Precision | Recall | F1 | IoU |
|---|:---:|:---:|:---:|:---:|
| Ground | — | — | — | — |
| Wall | — | — | — | — |
| Ceiling | — | — | — | — |
| Objects (INIT residual) | — | — | — | — |
| **Mean** | — | — | — | — |
