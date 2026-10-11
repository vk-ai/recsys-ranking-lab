# Multi-seed results (5 model seeds: 0, 1, 2, 3, 4)

Dataset `data/ml-latest-small`, features `spark`. Mean ± sd across seeds (sample sd).
Data split, simulated availability and the simulated click model are fixed; only model training randomness varies.

| policy | ndcg@10 | recall@10 | map@10 | ild@10 | coverage@10 | novelty@10 | aplt@10 | gini@10 |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.0573 ± 0.0000 | 0.0628 ± 0.0000 | 0.0295 ± 0.0000 | 0.7793 ± 0.0000 | 0.0104 ± 0.0000 | 1.92 ± 0.00 | 0.0080 ± 0.0000 | 0.9975 ± 0.0000 |
| two_tower | 0.0827 ± 0.0066 | 0.1015 ± 0.0079 | 0.0415 ± 0.0042 | 0.7692 ± 0.0013 | 0.0851 ± 0.0023 | 3.54 ± 0.06 | 0.5809 ± 0.0141 | 0.9656 ± 0.0015 |
| two_tower+ranker | 0.0853 ± 0.0058 | 0.1021 ± 0.0071 | 0.0427 ± 0.0039 | 0.7225 ± 0.0088 | 0.0589 ± 0.0030 | 3.13 ± 0.11 | 0.4562 ± 0.0355 | 0.9800 ± 0.0019 |
| two_tower+ranker+mmr | 0.0857 ± 0.0050 | 0.1026 ± 0.0053 | 0.0426 ± 0.0034 | 0.7812 ± 0.0070 | 0.0578 ± 0.0027 | 3.10 ± 0.10 | 0.4494 ± 0.0311 | 0.9804 ± 0.0017 |
| two_tower+ranker+genre_cap | 0.0815 ± 0.0051 | 0.0955 ± 0.0061 | 0.0411 ± 0.0032 | 0.8411 ± 0.0015 | 0.0586 ± 0.0038 | 3.06 ± 0.11 | 0.4389 ± 0.0344 | 0.9808 ± 0.0020 |

Ranker vs two-tower NDCG@10: mean delta +0.0026 ± 0.0041; ranker better in 4/5 seeds; per-seed paired t significant (p < 0.05) in 1/5.

Per-seed deltas: +0.0040, +0.0074, +0.0015, -0.0036, +0.0038
