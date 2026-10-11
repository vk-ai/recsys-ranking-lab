"""Run the pipeline under several model seeds and report mean, sd and min/max.

Only the *model* seeds change (two-tower init + batch order, ranker init + holdout
split). The data split, simulated availability and simulated A/B click model keep
``cfg.seed``, so differences between runs come from training randomness alone.
A single-seed NDCG gap smaller than the seed-to-seed spread is not a finding.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .pipeline import PipelineConfig, run

KEY_METRICS = ("ndcg@{k}", "recall@{k}", "map@{k}", "ild@{k}", "coverage@{k}", "novelty@{k}", "aplt@{k}", "gini@{k}")


def summarize(reports: list[dict], seeds: list[int]) -> dict:
    k = reports[0]["config"]["k"]
    keys = [m.format(k=k) for m in KEY_METRICS]
    policies = list(reports[0]["offline"])
    table: dict[str, dict[str, dict]] = {}
    for p in policies:
        table[p] = {}
        for key in keys:
            vals = np.array([r["offline"][p][key] for r in reports], dtype=float)
            table[p][key] = {
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "per_seed": [float(v) for v in vals],
            }
    nd = f"ndcg@{k}"
    deltas = np.array([r["offline"]["two_tower+ranker"][nd] - r["offline"]["two_tower"][nd] for r in reports])
    p_vals = [r["offline_significance"]["p_value"] for r in reports]
    return {
        "seeds": seeds,
        "k": k,
        "dataset": reports[0]["dataset"],
        "feature_engine": reports[0]["feature_engine"],
        "metrics": table,
        "ranker_vs_two_tower_ndcg": {
            "per_seed_delta": [float(d) for d in deltas],
            "mean_delta": float(deltas.mean()),
            "sd_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "seeds_ranker_better": int((deltas > 0).sum()),
            "per_seed_paired_t_p": p_vals,
            "seeds_significant_at_0.05": int(sum(p < 0.05 for p in p_vals)),
        },
        "seconds": [r["env"]["seconds"] for r in reports],
    }


def render(s: dict) -> str:
    k = s["k"]
    keys = [m.format(k=k) for m in KEY_METRICS]
    lines = [
        f"# Multi-seed results ({len(s['seeds'])} model seeds: {', '.join(map(str, s['seeds']))})",
        "",
        f"Dataset `{s['dataset']}`, features `{s['feature_engine']}`. Mean ± sd across seeds (sample sd).",
        "Data split, simulated availability and the simulated click model are fixed; only model training randomness varies.",
        "",
        "| policy | " + " | ".join(keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for p, row in s["metrics"].items():
        cells = []
        for key in keys:
            m = row[key]
            cells.append(
                f"{m['mean']:.2f} ± {m['sd']:.2f}" if key.startswith("novelty") else f"{m['mean']:.4f} ± {m['sd']:.4f}"
            )
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    d = s["ranker_vs_two_tower_ndcg"]
    lines += [
        "",
        f"Ranker vs two-tower NDCG@{k}: mean delta {d['mean_delta']:+.4f} ± {d['sd_delta']:.4f}; "
        f"ranker better in {d['seeds_ranker_better']}/{len(s['seeds'])} seeds; "
        f"per-seed paired t significant (p < 0.05) in {d['seeds_significant_at_0.05']}/{len(s['seeds'])}.",
        "",
        "Per-seed deltas: " + ", ".join(f"{x:+.4f}" for x in d["per_seed_delta"]),
        "",
    ]
    return "\n".join(lines)


def run_seeds(
    data_dir: Path,
    seeds: list[int],
    out_dir: Path = Path("artifacts"),
    reports_dir: Path | None = Path("reports"),
    engine: str = "pandas",
    base: PipelineConfig | None = None,
    log=print,
) -> dict:
    base = base or PipelineConfig()
    reports = []
    for s in seeds:
        cfg = copy.deepcopy(base)
        cfg.two_tower.seed = s
        cfg.ranker.seed = s
        log(f"== model seed {s}")
        reports.append(run(data_dir, Path(out_dir) / f"seed{s}", None, engine, cfg, log=log))
    summary = summarize(reports, seeds)
    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "multiseed.json").write_text(json.dumps(summary, indent=2) + "\n")
        (reports_dir / "MULTISEED.md").write_text(render(summary))
    return summary
