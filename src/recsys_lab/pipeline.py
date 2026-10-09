"""End-to-end offline pipeline.

split -> features (PySpark or pandas) -> two-tower retrieval -> ranker trained on
the val period -> constraints (SIMULATED availability, MMR / genre caps) ->
offline metrics on the test period -> paired significance -> SIMULATED A/B test,
A/A calibration and interleaving -> reports/metrics.json + reports/REPORT.md
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import abtest, constraints, data, features, metrics, ranker, stats, two_tower


@dataclass
class PipelineConfig:
    k: int = 10
    n_candidates: int = 100
    n_test: int = 10
    n_val: int = 10
    min_train: int = 10
    ranker_windows: int = 3  # rolling label windows for the re-ranker (0 = val period)
    available_rate: float = 0.85
    mmr_lambda: float = 0.7
    max_per_genre: int = 4
    ab_users: int = 20000
    aa_experiments: int = 200
    aa_users: int = 2000
    interleave_users: int = 5000
    bootstrap: int = 2000
    seed: int = 0
    two_tower: two_tower.TwoTowerConfig = field(default_factory=two_tower.TwoTowerConfig)
    ranker: ranker.RankerConfig = field(default_factory=ranker.RankerConfig)


def _numeric_tables(users_f: pd.DataFrame, items_f: pd.DataFrame):
    user_num = np.stack(
        [np.log1p(users_f["user_n"]), users_f["user_mean"], np.log1p(users_f["user_n_pos"]), users_f["user_last_year"]],
        1,
    ).astype(np.float32)
    user_aff = users_f[features.AFF_COLS].to_numpy(np.float32)
    year = items_f["year"].to_numpy(np.float64)
    med = np.nanmedian(year) if np.isfinite(year).any() else 0.0
    year = np.where(np.isfinite(year), year, med)
    item_num = np.stack(
        [np.log1p(items_f["item_n"]), items_f["item_mean_smoothed"], items_f["item_pos_rate"], year], 1
    ).astype(np.float32)
    item_genres = items_f[features.GENRE_COLS].to_numpy(np.float32)
    return user_num, user_aff, item_num, item_genres


def _fit_stage(
    tag: str,
    history: pd.DataFrame,
    history_csv: Path,
    movies: pd.DataFrame,
    data_dir: Path,
    out_dir: Path,
    engine: str,
    cfg: PipelineConfig,
    log,
) -> dict:
    """Feature job + two-tower fit on one history window."""
    feat_dir = out_dir / f"features_{tag}"
    if engine == "spark":
        from . import spark_job

        spark_job.run(str(history_csv), str(data_dir / "movies.csv"), str(feat_dir))
    elif engine == "pandas":
        features.write_features(*features.build_features_pandas(history, movies), feat_dir)
    else:
        raise ValueError(f"unknown engine {engine!r}")
    users_f, items_f = features.read_features(feat_dir)
    uid2row = {int(u): i for i, u in enumerate(users_f["userId"])}
    mid2row = {int(m): i for i, m in enumerate(items_f["movieId"])}
    user_num, user_aff, item_num, item_genres = _numeric_tables(users_f, items_f)
    pos = history[history["rating"] >= data.POSITIVE_RATING]
    model, losses = two_tower.train_two_tower(
        pos["userId"].map(uid2row).to_numpy(),
        pos["movieId"].map(mid2row).to_numpy(),
        user_aff,
        item_genres,
        cfg.two_tower,
    )
    uvec, ivec = two_tower.embed_all(model, user_aff, item_genres)
    log(
        f"stage {tag}: features ({engine}) users={len(users_f)} items={len(items_f)}; two-tower loss {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    return {
        "uid2row": uid2row,
        "mid2row": mid2row,
        "user_num": user_num,
        "user_aff": user_aff,
        "item_num": item_num,
        "item_genres": item_genres,
        "n_items": len(items_f),
        "uvec": uvec,
        "ivec": ivec,
        "tt_losses": losses,
    }


def run(
    data_dir: Path,
    out_dir: Path = Path("artifacts"),
    reports_dir: Path | None = Path("reports"),
    engine: str = "pandas",
    cfg: PipelineConfig = PipelineConfig(),
    log=print,
) -> dict:
    t0 = time.time()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out_dir = Path(out_dir)
    ratings, movies = data.load(data_dir)
    sp = data.temporal_split(ratings, cfg.n_test, cfg.n_val, cfg.min_train)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        getattr(sp, name).to_csv(split_dir / f"{name}.csv", index=False)
    log(f"split: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)} eval_users={len(sp.eval_users)}")

    # Two time-sliced "stages", like a daily retrain:
    #   stage "val":  features + two-tower fit on train only; its candidates on
    #                 the val period are the ranker's training lists.
    #   stage "test": features + two-tower refit on train+val, then the ranker
    #                 (trained in stage "val") re-scores its candidates on test.
    # Using train-only features at test time would put a whole period of
    # train/serve skew between the ranker's training data and its inputs.
    hist_test = pd.concat([sp.train, sp.val], ignore_index=True)
    hist_test.to_csv(split_dir / "train_val.csv", index=False)
    stages_v = []
    for w in range(cfg.ranker_windows):
        hist_w, labels_w = data.ranker_window(ratings, w, cfg.n_test, cfg.n_val, cfg.min_train)
        hist_w.to_csv(split_dir / f"history_w{w}.csv", index=False)
        stages_v.append(
            (
                _fit_stage(
                    f"w{w}", hist_w, split_dir / f"history_w{w}.csv", movies, Path(data_dir), out_dir, engine, cfg, log
                ),
                hist_w,
                labels_w,
            )
        )
    stage_v = stages_v[0][0]
    stage_t = _fit_stage(
        "test", hist_test, split_dir / "train_val.csv", movies, Path(data_dir), out_dir, engine, cfg, log
    )
    mid2row, item_genres, n_items = stage_t["mid2row"], stage_t["item_genres"], stage_t["n_items"]
    if stage_v["mid2row"] != mid2row:
        raise RuntimeError("item rows differ between stages")

    def to_rows(d: dict[int, set[int]]) -> dict[int, set[int]]:
        return {u: {mid2row[m] for m in s if m in mid2row} for u, s in d.items()}

    eval_users = sp.eval_users
    seen_train = to_rows(data.seen(sp.train))
    seen_val = to_rows(data.seen(sp.val))
    test_pos = to_rows(data.positives(sp.test))
    test_seen = to_rows(data.seen(sp.test))
    test_watched = {u: test_seen.get(u, set()) - test_pos.get(u, set()) for u in test_seen}

    def candidates(stage: dict, exclude: list[set[int]], users=None):
        users = eval_users if users is None else users
        urows = np.array([stage["uid2row"][int(u)] for u in users])
        cand, sc = two_tower.retrieve(stage["uvec"], stage["ivec"], urows, exclude, cfg.n_candidates)
        x = ranker.build_ranker_features(
            sc, cand, urows, stage["user_num"], stage["user_aff"], stage["item_num"], stage["item_genres"]
        )
        return cand, sc, x

    # --- ranker on the val period ------------------------------------------------
    xs, ys = [], []
    for w, (stage_w, hist_w, labels_w) in enumerate(stages_v):
        users_w = np.sort(labels_w["userId"].unique())
        seen_w = to_rows(data.seen(hist_w))
        pos_w = to_rows(data.positives(labels_w))
        cand_w, _, x_w = candidates(stage_w, [seen_w.get(int(u), set()) for u in users_w], users_w)
        y_w = np.array(
            [[1.0 if c in pos_w.get(int(u), set()) else 0.0 for c in row] for u, row in zip(users_w, cand_w)],
            dtype=np.float32,
        )
        xs.append(x_w)
        ys.append(y_w)
        if w == 0:
            cand_v, y_v = cand_w, y_w
    rk_model, rk_log = ranker.train_ranker(np.concatenate(xs), np.concatenate(ys), cfg.ranker)
    tt_val_ndcg = float(
        np.mean(
            [
                metrics.ndcg_at_k(list(cand_v[i]), set(cand_v[i][y_v[i] > 0]), 10)
                for i in range(len(y_v))
                if y_v[i].sum() > 0
            ]
        )
    )
    log(
        f"ranker: {rk_log['fit_lists']} fit / {rk_log['holdout_lists']} early-stopping lists, best epoch {rk_log['best_epoch']}, "
        f"holdout NDCG@10 {rk_log['best_holdout_ndcg@10']}"
    )

    # --- test period ---------------------------------------------------------------
    excl_test = [seen_train.get(int(u), set()) | seen_val.get(int(u), set()) for u in eval_users]
    cand_t, sc_t, x_t = candidates(stage_t, excl_test)
    rk_t = ranker.score(rk_model, x_t)
    tt_losses = stage_t["tt_losses"]
    available = constraints.simulated_availability(n_items, cfg.available_rate, seed=cfg.seed + 7)

    k = cfg.k
    hist_pos = hist_test[hist_test["rating"] >= data.POSITIVE_RATING]
    pop_counts = np.bincount(hist_pos["movieId"].map(mid2row).to_numpy(), minlength=n_items)
    pop_order = np.argsort(-pop_counts, kind="stable")
    policies: dict[str, dict[int, list[int]]] = {
        name: {}
        for name in (
            "popularity",
            "two_tower",
            "two_tower+ranker",
            "two_tower+ranker+mmr",
            "two_tower+ranker+genre_cap",
        )
    }
    unfiltered_ranker: dict[int, list[int]] = {}
    for j, u in enumerate(eval_users):
        u = int(u)
        ex = excl_test[j]
        policies["popularity"][u] = [int(i) for i in pop_order[: k + len(ex) + 200] if i not in ex and available[i]][:k]
        c_tt, s_tt = constraints.filter_available(cand_t[j : j + 1], sc_t[j : j + 1], available)
        policies["two_tower"][u] = c_tt[0][:k].tolist()
        order = np.argsort(-rk_t[j], kind="stable")
        unfiltered_ranker[u] = cand_t[j][order][:k].tolist()
        rc, rs = cand_t[j][order], rk_t[j][order]
        m = available[rc]
        rc, rs = rc[m], rs[m]
        policies["two_tower+ranker"][u] = rc[:k].tolist()
        policies["two_tower+ranker+mmr"][u] = constraints.mmr(
            rc[: 5 * k], rs[: 5 * k], item_genres, k, cfg.mmr_lambda
        ).tolist()
        policies["two_tower+ranker+genre_cap"][u] = constraints.genre_cap(
            rc, item_genres, k, cfg.max_per_genre
        ).tolist()

    rel_avail = {int(u): {i for i in test_pos.get(int(u), set()) if available[i]} for u in eval_users}
    rel_all = {int(u): test_pos.get(int(u), set()) for u in eval_users}
    results = {name: metrics.evaluate(lists, rel_avail, k, item_genres, n_items) for name, lists in policies.items()}
    cand_lists = {int(u): cand_t[j].tolist() for j, u in enumerate(eval_users)}
    retrieval = {
        f"recall@{cfg.n_candidates}": metrics.evaluate(cand_lists, rel_all, cfg.n_candidates)[
            f"recall@{cfg.n_candidates}"
        ],
        "recall@50": metrics.evaluate(cand_lists, rel_all, 50)["recall@50"] if cfg.n_candidates >= 50 else None,
    }
    unavailable_shown = float(np.mean([np.mean([not available[i] for i in lst]) for lst in unfiltered_ranker.values()]))

    users_eval = [u for u in policies["two_tower"] if rel_avail.get(u)]
    a_nd = metrics.per_user(metrics.ndcg_at_k, policies["two_tower"], rel_avail, k, users_eval)
    b_nd = metrics.per_user(metrics.ndcg_at_k, policies["two_tower+ranker"], rel_avail, k, users_eval)
    pt = stats.paired_t_test(a_nd, b_nd)
    pb = stats.paired_bootstrap_ci(a_nd, b_nd, cfg.bootstrap, cfg.seed)
    offline_sig = {
        "comparison": f"two_tower+ranker vs two_tower, per-user NDCG@{k}",
        **pt.to_dict(),
        "paired_bootstrap_ci_low": float(f"{pb[0]:.6g}"),
        "paired_bootstrap_ci_high": float(f"{pb[1]:.6g}"),
        "permutation_p_value": float(f"{stats.paired_permutation_test(a_nd, b_nd, 5000, cfg.seed):.6g}"),
    }

    # --- SIMULATED online experiment --------------------------------------------------
    rel_mid = rel_avail
    ab = abtest.ab_test(
        policies["two_tower"],
        policies["two_tower+ranker"],
        np.array(users_eval),
        rel_mid,
        test_watched,
        cfg.ab_users,
        seed=cfg.seed,
        n_bootstrap=cfg.bootstrap,
    )
    aa = abtest.aa_false_positive_rate(
        policies["two_tower"],
        np.array(users_eval),
        rel_mid,
        test_watched,
        cfg.aa_experiments,
        cfg.aa_users,
        seed=cfg.seed + 1,
    )
    il = abtest.interleaving_test(
        policies["two_tower"],
        policies["two_tower+ranker"],
        np.array(users_eval),
        rel_mid,
        test_watched,
        k,
        cfg.interleave_users,
        seed=cfg.seed + 2,
    )

    report = {
        "dataset": str(data_dir),
        "feature_engine": engine,
        "config": asdict(cfg),
        "split": {
            "train": len(sp.train),
            "val": len(sp.val),
            "test": len(sp.test),
            "eval_users": int(len(eval_users)),
            "items": n_items,
        },
        "training": {
            "two_tower_loss_first": tt_losses[0],
            "two_tower_loss_last": tt_losses[-1],
            "ranker_fit_lists": rk_log["fit_lists"],
            "ranker_holdout_lists": rk_log["holdout_lists"],
            "ranker_best_epoch": rk_log["best_epoch"],
            "ranker_best_holdout_ndcg@10": rk_log["best_holdout_ndcg@10"],
            "two_tower_val_ndcg@10_all_lists": tt_val_ndcg,
        },
        "retrieval": retrieval,
        "offline": results,
        "availability": {
            "simulated": True,
            "available_rate": float(available.mean()),
            "unavailable_share_in_unfiltered_top_k": unavailable_shown,
        },
        "offline_significance": offline_sig,
        "simulated_ab_test": {"control": "two_tower", "treatment": "two_tower+ranker", **ab},
        "simulated_aa_calibration": aa,
        "simulated_interleaving": {"control": "two_tower", "treatment": "two_tower+ranker", **il},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "seconds": round(time.time() - t0, 1)},
    }
    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
        (reports_dir / "REPORT.md").write_text(render_markdown(report))
    log(f"done in {report['env']['seconds']}s")
    return report


def _f(x: float) -> str:
    return f"{x:.4f}"


def render_markdown(r: dict) -> str:
    k = r["config"]["k"]
    lines = [
        "# Offline evaluation report",
        "",
        f"Dataset: `{r['dataset']}`; feature engine: `{r['feature_engine']}`; "
        f"split: {r['split']['train']} train / {r['split']['val']} val / {r['split']['test']} test ratings, {r['split']['eval_users']} eval users, {r['split']['items']} items.",
        "",
        f"Relevant = test-period ratings >= 4 on items that are (SIMULATED) available. k = {k}.",
        "",
        f"| policy | recall@{k} | NDCG@{k} | MAP@{k} | precision@{k} | ILD@{k} | coverage@{k} |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in r["offline"].items():
        lines.append(
            f"| {name} | {_f(m[f'recall@{k}'])} | {_f(m[f'ndcg@{k}'])} | {_f(m[f'map@{k}'])} | {_f(m[f'precision@{k}'])} | {_f(m[f'ild@{k}'])} | {_f(m[f'coverage@{k}'])} |"
        )
    ret = r["retrieval"]
    n = r["config"]["n_candidates"]
    s = r["offline_significance"]
    ab = r["simulated_ab_test"]
    cps = ab["clicks_per_session"]
    z = ab["any_click_rate"]
    il = r["simulated_interleaving"]
    aa = r["simulated_aa_calibration"]
    lines += [
        "",
        f"Two-tower candidate recall (all relevant, before availability): recall@50 = {_f(ret['recall@50'] or 0)}, recall@{n} = {_f(ret[f'recall@{n}'])}.",
        f"Without the availability filter, {r['availability']['unavailable_share_in_unfiltered_top_k']:.1%} of the ranker's top-{k} would be unavailable (SIMULATED, {r['availability']['available_rate']:.1%} of items available).",
        "",
        "## Offline significance (paired, same users)",
        "",
        f"{s['comparison']}: delta = {s['delta']:+.4f} (paired t p = {s['p_value']:.3g}, permutation p = {s['permutation_p_value']:.3g}, "
        f"95% paired bootstrap CI [{s['paired_bootstrap_ci_low']:+.4f}, {s['paired_bootstrap_ci_high']:+.4f}]).",
        "",
        "## SIMULATED A/B test (click model, not real users)",
        "",
        f"Control `{ab['control']}` n={ab['n_control']}, treatment `{ab['treatment']}` n={ab['n_treatment']}; SRM p = {ab['srm_p_value']:.3g}.",
        "",
        f"- Clicks per session: {cps['mean_a']:.4f} -> {cps['mean_b']:.4f} ({cps['relative_lift']:+.1%}), Welch t = {cps['statistic']:.2f}, p = {cps['p_value']:.3g}, "
        f"95% CI [{cps['ci_low']:+.4f}, {cps['ci_high']:+.4f}], bootstrap CI [{cps['bootstrap_ci_low']:+.4f}, {cps['bootstrap_ci_high']:+.4f}].",
        f"- Any-click rate: {z['mean_a']:.4f} -> {z['mean_b']:.4f}, z = {z['statistic']:.2f}, p = {z['p_value']:.3g}.",
        f"- Users per arm for 80% power at the observed any-click lift: {ab['users_per_arm_for_80pct_power_at_observed_lift']}.",
        f"- A/A calibration: {aa['experiments']} A/A tests of {aa['users_per_experiment']} users, false-positive rate {aa['false_positive_rate']:.3f} at alpha {aa['alpha']}.",
        f"- Team-draft interleaving ({il['users']} users): treatment wins {il['wins_treatment']}, control wins {il['wins_control']}, ties {il['ties']}; sign test p = {il['sign_test_p_value']:.3g}.",
        "",
        f"Run: python {r['env']['python']}, torch {r['env']['torch']}, {r['env']['seconds']} s on CPU.",
        "",
    ]
    return "\n".join(lines)
