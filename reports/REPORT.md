# Offline evaluation report

Dataset: `data/ml-latest-small`; feature engine: `spark`; split: 90816 train / 5010 val / 5010 test ratings, 501 eval users, 9742 items.

Relevant = test-period ratings >= 4 on items that are (SIMULATED) available. k = 10.

| policy | recall@10 | NDCG@10 | MAP@10 | precision@10 | ILD@10 | coverage@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0628 | 0.0573 | 0.0295 | 0.0333 | 0.7793 | 0.0104 |
| two_tower | 0.1049 | 0.0813 | 0.0397 | 0.0508 | 0.7680 | 0.0852 |
| two_tower+ranker | 0.1017 | 0.0852 | 0.0432 | 0.0499 | 0.7231 | 0.0590 |
| two_tower+ranker+mmr | 0.1057 | 0.0873 | 0.0436 | 0.0518 | 0.7844 | 0.0585 |
| two_tower+ranker+genre_cap | 0.0946 | 0.0815 | 0.0419 | 0.0456 | 0.8400 | 0.0597 |

Two-tower candidate recall (all relevant, before availability): recall@50 = 0.2592, recall@100 = 0.3652.
Without the availability filter, 12.3% of the ranker's top-10 would be unavailable (SIMULATED, 84.5% of items available).

## Offline significance (paired, same users)

two_tower+ranker vs two_tower, per-user NDCG@10: delta = +0.0040 (paired t p = 0.361, permutation p = 0.368, 95% paired bootstrap CI [-0.0044, +0.0125]).

## SIMULATED A/B test (click model, not real users)

Control `two_tower` n=10003, treatment `two_tower+ranker` n=9997; SRM p = 0.966.

- Clicks per session: 0.1756 -> 0.1812 (+3.2%), Welch t = 1.32, p = 0.186, 95% CI [-0.0027, +0.0141], bootstrap CI [-0.0029, +0.0138].
- Any-click rate: 0.3604 -> 0.3653, z = 0.72, p = 0.47.
- Users per arm for 80% power at the observed any-click lift: 150065.
- A/A calibration: 200 A/A tests of 2000 users, false-positive rate 0.055 at alpha 0.05.
- Team-draft interleaving (5000 users): treatment wins 392, control wins 376, ties 4232; sign test p = 0.588.

Run: python 3.12.14, torch 2.14.1+cpu, 49.9 s on CPU.
