"""trajectory_eval -- a small, codebase-agnostic toolkit for critiquing trajectory-prediction evaluation.

The whole point: *the toolkit*, not any individual model repo, is the single source of
truth for metrics. Every model run dumps raw per-track predictions (or at least per-track
errors) in one standard format; every experiment (A--G in the paper) is then a script over
those files, computed identically across models. That is what makes the numbers comparable
by construction -- the opposite of the incomparable-protocol problem the paper critiques.

Modules
-------
schema   : standardized per-track dump (.npz) + uniform metric computation
io       : loaders (new .npz dumps AND the legacy SenSys .pkl files)
stats    : bootstrap CIs, seed aggregation, paired/unpaired tests, Holm-Bonferroni  (Exp A)
tails    : quantiles, CVaR, tail-share, rank-flip across metrics                      (Exp C)
bestofk  : minADE_K sweep + oracle-vs-deployable selection gap                        (Exp B)
safety   : safety-critical subset splitting (PIE crossing labels)                     (Exp F)
plots    : forest plot (overlapping CIs) + Lorenz/CDF of error mass
"""

from . import schema, io, stats, tails, bestofk, safety, plots  # noqa: F401

__all__ = ["schema", "io", "stats", "tails", "bestofk", "safety", "plots"]
__version__ = "0.1.0"
