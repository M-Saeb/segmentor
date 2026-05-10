# Paper Documentation

This folder contains the content for the IEEE-style academic paper on Segmentor.

## File Structure

```
docs/
├── README.md               ← This file
├── sections/
│   ├── 01_abstract.md
│   ├── 02_introduction.md
│   ├── 03_related_work.md
│   ├── 04_system_design.md      ← Primary contribution (architecture)
│   ├── 05_implementation.md     ← Algorithms and data structures
│   ├── 06_evaluation.md
│   └── 07_conclusion.md
└── figures/
    └── (placeholder descriptions for all diagrams)
```

## Processing Stages (as named in the paper)

| Stage | Old name | New name |
|---|---|---|
| 1 | Pre-processing | Pre-processing (unchanged) |
| 2 | Pipeline | Segmentation Stage |
| 3 | Post-processing | VLM Clustering Stage |
| 4 | *(new)* | Exporting Stage |

## Status

| Section | Status | Notes |
|---|---|---|
| Abstract | Draft | Needs evaluation numbers |
| Introduction | Draft | Needs domain focus clarification |
| Related Work | Skeleton | Needs citation input from author |
| System Design | Complete Draft | Core contribution; 4 stages |
| Implementation | Complete Draft | Algorithms covered in full |
| Evaluation | Skeleton | Needs benchmark data |
| Conclusion | Skeleton | Needs final contributions summary |
