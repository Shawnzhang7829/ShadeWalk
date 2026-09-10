# Geometry-first detection route - comparison report

Date of the experiment: 2026-06-11. Controlled variables: the facade-facing test calls `step2_project.py` of
Module 3b directly (no code change); the CLIP probes and thresholds are those of production
(SG `sg_colonnade_final` 0.45 / BO `bo_probe` 0.40); the vector chain is the same project -> merge -> buffer code,
only the input and output folders differ.

## 1. The two orderings

| | CLIP-first (production) | Geometry-first (`geometry_first/`) |
|---|---|---|
| stages | broad screen (all views, plus backfill of views 1/3) -> candidate panoramas -> colonnade probe -> Module 3b facade test + projection | remove panorama points inside buildings -> **exact facade-facing geometry on all views** -> colonnade probe only on facing views -> same projection |
| CLIP encodings | 1,039,104 images (two broad passes + candidate re-encoding) | **399,631 images (-62 %)** |
| number of stages | 4 | 3 |

## 2. View-level agreement (restricted to the candidate panoramas of the production run)

| City | NEW | OLD | agree | only NEW | only OLD | Jaccard |
|---|---|---|---|---|---|---|
| SG | 18,042 | 18,032 | 18,032 | 10 (*) | **0** | 0.9994 |
| BO | 24,999 | 24,999 | 24,999 | 0 | **0** | **1.0000** |

(*) panoramas that the production version skipped because it required all four views to be present.

Conclusion: the production result is reproduced completely (only-OLD = 0); swapping the order does not change
the result. In addition the geometry-first route detects arcade views outside the candidate domain (views the
broad screen had rejected): **+883 in Singapore and +3,568 in Bologna**.

## 3. Vector-level comparison (same vector rules)

| Metric | SG old | SG new | change | BO old | BO new | change |
|---|---|---|---|---|---|---|
| projected segments (views) | 18,032 | 18,925 | +5.0 % | 24,999 | 28,567 | +14.3 % |
| arcade runs | 2,134 | 2,205 | +3.3 % | 1,397 | 1,661 | +18.9 % |
| total length (km) | 146.3 | 150.1 | +2.6 % | 169.8 | 188.8 | **+11.2 %** |
| arcade strip polygons | 7,081 | 7,297 | +3.1 % | 5,943 | 6,643 | +11.8 % |
| strip area (m2) | 176,402 | 181,034 | +2.6 % | 411,995 | 455,337 | **+10.5 %** |
| mean arcade height (m) | 3.60 | 3.60 | - | 3.59 | 3.58 | - |

All additions come from the recovered broad-screen misses (validated by the same probe and threshold);
Bologna benefits more because its Tier-3 threshold was stricter.

(The numbers in this table refer to the gap-bridging merge of June 2026, `step2_merge.py`; the final Singapore
vectors of the paper were produced later with the edge-based method `step2_edge_ffw.py`, see Module 3b.)

## 4. Run time (both cities, same machine)

| Stage | CLIP-first | geometry-first |
|---|---|---|
| CLIP broad screen + backfill | 228 min | - |
| CLIP colonnade probe | 66 min | 163 min (facing views only, 41 img/s, serial image reading) |
| facade-facing geometry | about 2 min (64 k positive views, inside the projection) | 16.8 min (883 k views, up front) |
| projection -> merge -> buffer | about 11 min | 16 min |
| **total** | **about 307 min** | **about 196 min (-36 %)** |

With the production image prefetching (62-68 img/s) the probe stage drops to about 103 min and the whole chain to
about 136 min (-56 %), in line with the -62 % encoding count.

## 5. Conclusions

1. Correctness: the two routes agree view by view on the common domain (Bologna identical; the Singapore
   production set is reproduced 100 %). The earlier -30 % discrepancy of a crude pre-filter came entirely from
   using a different facing criterion, not from the ordering.
2. Completeness: geometry-first is a strict superset (+883 / +3,568 views -> +2.6 % / +11 % km).
3. Efficiency: measured -36 % (up to -56 % with prefetching); the CLIP encoding count is -62 %.
4. Lesson: a pre-filter must use the same criterion as the final test; a different proxy costs about one third of
   the recall.

## Files

- `run_facing_precise.py` - facade-facing test on all views (`facing_views_<city>.csv`)
- `detect_precise.py` - colonnade probe on the facing views (`detect_scores_<city>.csv`, `arcade_precise_<city>_positive.csv`)
- `project_precise.py` - runs the Module 3b chain on the precise positives into a separate folder
- `compare_match.py` - view-level match verdict against production
- `viz_added_map.py`, `make_figs_v2.py` - figures (where the recovered runs are; standard figure set)
