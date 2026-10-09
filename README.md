# recsys-ranking-lab

**Learning project.** A small, end-to-end two-stage recommender on the public
[MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset, built to practise
retrieval, learning-to-rank, constraint-aware re-ranking, offline evaluation and experiment
analysis. It is an OSS/learning repo. It is not used in production anywhere and is not based on
any employer's system.

```
ratings ──► per-user temporal split ──► PySpark feature job (user/item features, local mode)
                                             │
                    two-tower retrieval (PyTorch, in-batch softmax + logQ) ──► top-100 candidates
                                             │
              MLP re-ranker (PyTorch, listwise softmax loss, early stopping) ──► scores
                                             │
   SIMULATED availability filter ──► diversity (MMR or per-genre caps) ──► top-10 slate
                                             │
     offline metrics (recall/NDCG/MAP@k, ILD, coverage) + paired significance tests
                                             │
     SIMULATED online experiment: A/B (Welch t, z-test, bootstrap CIs, SRM, power), A/A, interleaving
```

## What's in it

| Piece | Where | Notes |
|---|---|---|
| Dataset download with a pinned SHA-256 | `data.py`, `recsys-lab download` | ~1 MB zip from GroupLens, never committed |
| Per-user temporal split | `data.temporal_split` | last 10 ratings per user = test, previous 10 = val, users need ≥ 30 ratings to be evaluated |
| PySpark feature job | `spark_job.py` | user activity, mean rating, last-active year, genre affinity; item counts, smoothed mean, positive rate, year, genre flags. A pandas reference (`features.py`) must match it exactly (parity test in CI) |
| Two-tower retrieval | `two_tower.py` | id + side-feature towers, L2-normalised, in-batch sampled softmax with logQ correction, brute-force top-k |
| Re-ranker | `ranker.py` | MLP over two-tower score/rank, popularity, smoothed rating, release-year-vs-user-time, genre affinity × genre; ListNet-style loss; early stopping on held-out lists |
| Rolling training windows | `data.ranker_window`, `pipeline.py` | the ranker learns from 3 label windows (val + 2 earlier), each with its own feature job + two-tower fit, and at test time everything is refit on train+val, which reduces train/serve skew |
| Constraints | `constraints.py` | **simulated** availability filter, MMR (genre Jaccard), per-genre caps with back-fill |
| Metrics | `metrics.py` | recall@k, precision@k, NDCG@k, MAP@k (AP normalised by min(|rel|, k)), intra-list diversity, catalog coverage |
| Statistics | `stats.py` | Welch and paired t-tests (exact Student-t tail via the incomplete beta), two-proportion z-test, unpaired and paired bootstrap CIs, sign-flip permutation test, sign test, sample size, SRM check. numpy + stdlib only, checked against scipy reference values in tests. Mirrors the approach in [llm-eval-lab](https://github.com/vk-ai/llm-eval-lab) |
| Simulated experiment | `abtest.py` | A/B test on a position-based click model, 200 A/A tests to check the false-positive rate, team-draft interleaving |

## Results (real run, `recsys-lab run --engine spark`)

MovieLens ml-latest-small: 90,816 train / 5,010 val / 5,010 test ratings, 501 evaluated users,
9,742 items. Relevant = test-period rating ≥ 4 on an item that is (simulated) available. k = 10.
CPU only, Python 3.12, torch 2.14.1+cpu, PySpark 4.2.0 in local mode, 50 s end to end.
The pandas feature engine gives identical numbers. Full output: [`reports/REPORT.md`](reports/REPORT.md),
[`reports/metrics.json`](reports/metrics.json).

| policy | recall@10 | NDCG@10 | MAP@10 | precision@10 | ILD@10 | coverage@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0628 | 0.0573 | 0.0295 | 0.0333 | 0.7793 | 0.0104 |
| two_tower | 0.1049 | 0.0813 | 0.0397 | 0.0508 | 0.7680 | 0.0852 |
| two_tower+ranker | 0.1017 | 0.0852 | 0.0432 | 0.0499 | 0.7231 | 0.0590 |
| two_tower+ranker+mmr (λ = 0.7) | 0.1057 | 0.0873 | 0.0436 | 0.0518 | 0.7844 | 0.0585 |
| two_tower+ranker+genre_cap (≤ 4 per genre) | 0.0946 | 0.0815 | 0.0419 | 0.0456 | 0.8400 | 0.0597 |

- Two-tower candidate recall (before availability filtering): recall@50 = 0.259, recall@100 = 0.365.
  This caps what the re-ranker can do.
- Without the availability filter, 12.3% of the ranker's top 10 would be unavailable items
  (simulated, 84.5% of items available).
- **The ranker's offline lift is not significant.** Two-tower+ranker vs two-tower on per-user
  NDCG@10: +0.0040, paired t p = 0.36, permutation p = 0.37, 95% paired bootstrap CI [−0.0044, +0.0125].
  With 501 users, an effect this small can't be separated from noise.
- Diversity trade-off: MMR raises ILD (0.723 → 0.784) and here also NDCG. Genre caps raise ILD
  further (0.840) at a cost in recall.

**Simulated A/B test** (click model, *not* real users): two_tower (control, n = 10,003) vs
two_tower+ranker (treatment, n = 9,997), SRM p = 0.97.

- Clicks per session: 0.1756 → 0.1812 (+3.2%), Welch t = 1.32, p = 0.19, 95% CI [−0.0027, +0.0141],
  bootstrap CI [−0.0029, +0.0138]. Any-click rate: 0.360 → 0.365, z = 0.72, p = 0.47.
- Detecting the observed any-click lift at 80% power would need about 150k users per arm.
- A/A check: 200 A/A tests of 2,000 users each, false-positive rate 0.055 at α = 0.05 (as expected).
- Team-draft interleaving (5,000 users): treatment 392 wins, control 376 wins, 4,232 ties; sign test p = 0.59.

**How these numbers were reached, honestly.** I changed the pipeline four times while looking at
test-period results. All four earlier variants trained the ranker on a single val window and scored
test with train-only features, and the ranker came out *worse* than two-tower alone, with NDCG@10
deltas between −0.002 and −0.006. The changes were:
(1) a release-year-vs-user-time feature;
(2) early stopping, with the ranker size chosen by 5-fold CV on the **val** lists only;
(3) refitting features and two-tower on train+val before scoring test;
(4) two earlier label windows for ranker training.
Those are standard fixes for overfitting and train/serve skew, not a test-set hyper-parameter search.
Still, the test period was looked at repeatedly, so treat the small ranker lift as optimistic. It
isn't significant anyway.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheel
pip install -e ".[dev,spark]"        # drop ",spark" to skip PySpark (needs Java 17+)

pytest -q                            # offline, ~30 s; Spark tests skip without pyspark/Java
recsys-lab download                  # MovieLens ml-latest-small, SHA-256 verified, into data/
recsys-lab run --engine spark        # or --engine pandas; writes reports/ and artifacts/

# the feature job on its own
python -m recsys_lab.spark_job --train artifacts/splits/train.csv \
  --movies data/ml-latest-small/movies.csv --out artifacts/features_manual
```

Tests use a tiny **synthetic** fixture in MovieLens format (`tests/fixtures/mini`, generated by
`tests/fixtures/make_fixture.py`), so they never download anything. MovieLens itself is not
redistributed here. See the GroupLens usage terms, and cite F. Maxwell Harper and Joseph A. Konstan, 2015,
*The MovieLens Datasets: History and Context*, ACM TiiS 5(4).

CI (GitHub Actions) has three jobs:
- lint plus unit and pipeline tests on Python 3.11 and 3.12, with CPU torch;
- a Spark job on Java 17 for pandas/Spark feature parity, the pipeline on the Spark engine, and the CLI job;
- a `movielens-e2e` job that downloads the real dataset and runs the full pipeline with PySpark, then posts the report to the job summary.

## Real vs stubbed / simulated

| Real | Simulated or stubbed |
|---|---|
| MovieLens ratings, downloaded and checksum-verified | **Item availability is random** (seeded, 85% available). MovieLens has no stock/opening-hours signal; it only shows where the filter sits and what it costs |
| Two-tower and ranker training in PyTorch on CPU, real metrics from a real run | **The A/B test, A/A tests and interleaving are simulated**: a position-based click model with made-up parameters (0.35 / 0.10 / 0.02 attractiveness, 1/log2 position decay), applied to offline slates. No real users or traffic |
| PySpark feature job, actually executed in local mode and parity-tested against pandas | Simulated users are resamples of the same 501 evaluation users, so the A/B "n" overstates how much independent preference data there is. The significance results describe the simulator, not a population |
| Offline metrics and significance tests (checked against scipy) | Spark only ever runs in `local[2]` mode; it has not been run on a cluster (Dataproc/EMR/Databricks) |
| Diversity re-ranking (MMR, genre caps) | Movies stand in for restaurants/dishes, and genres stand in for cuisines. No delivery-specific signals (location, ETA, prep time, price) |
| | The ranker is an MLP, not LightGBM/GBDT LambdaMART (kept dependency-free), and retrieval is brute-force dot product, not an ANN index (FAISS/ScaNN) |
| | No feature store, model registry, serving API or online inference; everything is a batch script |
| | Per-user leave-last-out split, not a global time cut, so some cross-user future information exists in training (standard MovieLens practice, but optimistic) |
| | Single dataset, single seed; no hyper-parameter sweep or variance across seeds is reported |

## Layout

```
src/recsys_lab/   data, features, spark_job, two_tower, ranker, constraints, metrics, stats, abtest, pipeline, cli
tests/            33 offline tests (2 need Spark), synthetic fixture
reports/          REPORT.md and metrics.json from the run above
```

License: MIT (code only; the MovieLens data has its own terms).
