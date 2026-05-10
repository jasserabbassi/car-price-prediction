"""Wilcoxon signed-rank test on per-fold R² values.

Loads `models/per_fold_r2.json` (produced by `train_model.py`) and tests
whether the Weighted Ensemble's per-fold R² is significantly greater than
each individual base model's per-fold R² (one-sided test).

Outputs:
    models/wilcoxon_results.md        markdown summary table
    models/wilcoxon_results.json      raw numeric results
    stdout                             pretty-printed summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from config import MODELS_DIR

ALPHA = 0.05
ENSEMBLE_KEY = "Weighted Ensemble"
PER_FOLD_PATH = MODELS_DIR / "per_fold_r2.json"
MD_PATH = MODELS_DIR / "wilcoxon_results.md"
JSON_PATH = MODELS_DIR / "wilcoxon_results.json"


def main() -> int:
    if not PER_FOLD_PATH.exists():
        print(f"ERROR: {PER_FOLD_PATH} not found. Run train_model.py first.")
        return 1

    per_fold = json.loads(PER_FOLD_PATH.read_text())
    if ENSEMBLE_KEY not in per_fold:
        print(f"ERROR: '{ENSEMBLE_KEY}' missing from {PER_FOLD_PATH}.")
        return 2

    ensemble = np.asarray(per_fold[ENSEMBLE_KEY], dtype=float)
    print(f"Ensemble per-fold R²: {ensemble.tolist()}")
    print(f"Ensemble mean ± std: {ensemble.mean():.4f} ± {ensemble.std():.4f}\n")

    rows = []
    for name, scores in per_fold.items():
        if name == ENSEMBLE_KEY:
            continue
        base = np.asarray(scores, dtype=float)
        diff = ensemble - base
        if np.allclose(diff, 0):
            stat, pval = float("nan"), 1.0
            interp = "ties everywhere"
        else:
            try:
                res = wilcoxon(ensemble, base, alternative="greater", zero_method="zsplit")
                stat = float(res.statistic)
                pval = float(res.pvalue)
                interp = "ensemble > base" if pval < ALPHA else "not significant"
            except Exception as exc:  # pragma: no cover
                stat, pval = float("nan"), float("nan")
                interp = f"error: {exc}"

        rows.append(
            {
                "Comparison": f"Ensemble vs {name}",
                "Ensemble mean R²": float(ensemble.mean()),
                "Base mean R²": float(base.mean()),
                "Mean Δ": float(np.mean(diff)),
                "W": stat,
                "p-value (one-sided)": pval,
                f"Significant @ α={ALPHA}": pval < ALPHA if not np.isnan(pval) else False,
                "Interpretation": interp,
            }
        )

    # Markdown table.
    md_lines = [
        "# Wilcoxon Signed-Rank Test — Weighted Ensemble vs Base Models",
        "",
        f"- Source: `{PER_FOLD_PATH.relative_to(MODELS_DIR.parent)}`",
        f"- Alternative hypothesis: ensemble per-fold R² > base per-fold R²",
        f"- Significance threshold: α = {ALPHA}",
        "",
        "| Comparison | Ensemble R² | Base R² | Mean Δ | W | p (one-sided) | Significant? |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {Comparison} | {ER:.4f} | {BR:.4f} | {D:+.4f} | {W} | {P} | {S} |".format(
                Comparison=row["Comparison"],
                ER=row["Ensemble mean R²"],
                BR=row["Base mean R²"],
                D=row["Mean Δ"],
                W=f"{row['W']:.2f}" if not np.isnan(row["W"]) else "—",
                P=f"{row['p-value (one-sided)']:.4f}" if not np.isnan(row["p-value (one-sided)"]) else "—",
                S="**yes**" if row[f"Significant @ α={ALPHA}"] else "no",
            )
        )

    MD_PATH.write_text("\n".join(md_lines) + "\n")
    JSON_PATH.write_text(json.dumps(rows, indent=2, default=str))

    # Pretty-print to stdout.
    for row in rows:
        print(
            f"{row['Comparison']:38s}"
            f"  Δ={row['Mean Δ']:+.4f}  "
            f"W={row['W'] if not np.isnan(row['W']) else 'NA':>5}  "
            f"p={row['p-value (one-sided)']:.4f}  "
            f"{'***' if row[f'Significant @ α={ALPHA}'] else '   '}"
        )
    print(f"\nWrote {MD_PATH}")
    print(f"Wrote {JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
