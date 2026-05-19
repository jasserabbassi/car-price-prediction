# Wilcoxon Signed-Rank Test — Weighted Ensemble vs Base Models

- Source: `models/per_fold_r2.json`
- Alternative hypothesis: ensemble per-fold R² > base per-fold R²
- Significance threshold: α = 0.05

| Comparison | Ensemble R² | Base R² | Mean Δ | W | p (one-sided) | Significant? |
|---|---|---|---|---|---|---|
| Ensemble vs Linear Regression | 0.6887 | 0.2579 | +0.4308 | 15.00 | 0.0312 | **yes** |
| Ensemble vs Ridge Regression | 0.6887 | 0.2579 | +0.4307 | 15.00 | 0.0312 | **yes** |
| Ensemble vs Lasso Regression | 0.6887 | 0.2579 | +0.4307 | 15.00 | 0.0312 | **yes** |
| Ensemble vs Support Vector Regression | 0.6887 | 0.0976 | +0.5911 | 15.00 | 0.0312 | **yes** |
| Ensemble vs Random Forest | 0.6887 | 0.6978 | -0.0092 | 1.00 | 0.9688 | no |
| Ensemble vs Gradient Boosting | 0.6887 | 0.6385 | +0.0502 | 15.00 | 0.0312 | **yes** |
| Ensemble vs XGBoost | 0.6887 | 0.6851 | +0.0035 | 12.00 | 0.1562 | no |
| Ensemble vs AdaBoost | 0.6887 | 0.3795 | +0.3092 | 15.00 | 0.0312 | **yes** |
| Ensemble vs MLP | 0.6887 | 0.5574 | +0.1312 | 15.00 | 0.0312 | **yes** |
