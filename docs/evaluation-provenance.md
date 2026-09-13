# Evaluation provenance

The reported metrics are generated from synthetic transaction data created by the repository scripts with fixed seeds. The supervised evaluation sorts rows chronologically by person and uses a later time-based holdout; the cross-validation scores are produced by `TimeSeriesSplit`. This is an offline synthetic benchmark and should not be interpreted as evidence about real users or financial outcomes.

The feature-set comparison includes simple historical baselines. The realistic/stress scenarios are intentionally harder and their negative results are retained rather than replaced with a more favorable run.
