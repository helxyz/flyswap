"""Ranking cell types by gradient, then checking the ranking against grafts.

Grafting every cell type is the honest way to find which ones carry a sex
difference, and on a real connectome it means thousands of simulations. The
gradient is the cheap alternative: because the rate model is differentiable in
the per-cell-type gains, one backward pass yields d(readout)/d(gain) for every
type at once.

Gradients and grafts answer slightly different questions, which is exactly why
both belong here. The gradient is a local derivative: how the readout responds
to nudging a type's influence from its current value. A graft is a finite,
structured change: replacing that type's wiring wholesale. They agree when the
readout responds smoothly, and diverge when it does not.

So the deliverable of this module is not the ranking alone, it is the
*agreement* between the ranking and the grafts it is meant to replace —
reported as precision@k and rank correlation. A low number is a real finding
about when cheap attribution can be trusted, not a failure to hide.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from flyswap.sim.rate import RateSimulator


@dataclass
class AttributionResult:
    """Per-type scores, highest influence first."""

    types: list[str]
    scores: np.ndarray
    method: str
    readout_type: str
    config: dict

    def ranking(self, descending: bool = True) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.scores) if descending else np.abs(self.scores))
        return [(self.types[i], float(self.scores[i])) for i in order]

    def top(self, k: int) -> list[str]:
        return [t for t, _ in self.ranking()[:k]]

    def score_for(self, cell_type: str) -> float:
        try:
            return float(self.scores[self.types.index(cell_type)])
        except ValueError:
            return 0.0


def _readout_fn(
    sim: RateSimulator,
    stimulus: jnp.ndarray,
    readout_rows: jnp.ndarray,
    duration_ms: float,
    window_ms: float,
):
    """Scalar readout as a differentiable function of the type gains."""
    n_steps = int(round(duration_ms / sim.params.dt_ms))
    n_window = max(1, int(round(window_ms / sim.params.dt_ms)))

    def f(type_gains: jnp.ndarray) -> jnp.ndarray:
        trace = sim._run(sim.W.data, stimulus, type_gains, n_steps)
        steady = trace[-n_window:].mean(axis=0)
        return steady[readout_rows].mean()

    return f


def gain_gradient(
    sim: RateSimulator,
    stimulus_type: str,
    readout_type: str,
    amplitude: float = 50.0,
    duration_ms: float = 200.0,
    window_ms: float = 50.0,
    at_gain: float = 1.0,
) -> AttributionResult:
    """d(readout)/d(gain) for every cell type, in one backward pass."""
    rows = sim.graph.rows_for_type(readout_type)
    if len(rows) == 0:
        raise KeyError(f"readout type {readout_type!r} has no neurons")

    stim = sim.stimulus_for_type(stimulus_type, amplitude)
    f = _readout_fn(sim, stim, jnp.asarray(rows), duration_ms, window_ms)
    gains = jnp.full(sim.n_types, at_gain, dtype=jnp.float32)
    grads = np.asarray(jax.grad(f)(gains))

    return AttributionResult(
        types=list(sim.type_names),
        scores=grads,
        method="gain_gradient",
        readout_type=readout_type,
        config={
            "stimulus_type": stimulus_type,
            "amplitude": amplitude,
            "duration_ms": duration_ms,
            "window_ms": window_ms,
            "at_gain": at_gain,
            **sim.params.describe(),
        },
    )


def integrated_gradients(
    sim: RateSimulator,
    stimulus_type: str,
    readout_type: str,
    steps: int = 32,
    baseline_gain: float = 0.0,
    target_gain: float = 1.0,
    amplitude: float = 50.0,
    duration_ms: float = 200.0,
    window_ms: float = 50.0,
) -> AttributionResult:
    """Integrated gradients along the path from silenced to intact.

    A plain gradient is a derivative at one point, which can be misleading
    when a readout saturates: a type that fully drives the readout may show a
    near-zero local slope precisely because it is already doing all the work.
    Integrating the gradient from "gain 0" (type silenced) to "gain 1" (type
    intact) measures the type's total contribution over that path instead, and
    is the better match to what a graft does.

    Computed by averaging gradients at `steps` points along the path, which is
    the standard Riemann approximation.
    """
    rows = sim.graph.rows_for_type(readout_type)
    if len(rows) == 0:
        raise KeyError(f"readout type {readout_type!r} has no neurons")

    stim = sim.stimulus_for_type(stimulus_type, amplitude)
    f = _readout_fn(sim, stim, jnp.asarray(rows), duration_ms, window_ms)
    grad_f = jax.jit(jax.grad(f))

    alphas = (np.arange(steps) + 0.5) / steps
    total = np.zeros(sim.n_types, dtype=np.float64)
    for a in alphas:
        g = jnp.full(
            sim.n_types,
            baseline_gain + a * (target_gain - baseline_gain),
            dtype=jnp.float32,
        )
        total += np.asarray(grad_f(g), dtype=np.float64)

    scores = (total / steps) * (target_gain - baseline_gain)

    return AttributionResult(
        types=list(sim.type_names),
        scores=scores,
        method="integrated_gradients",
        readout_type=readout_type,
        config={
            "stimulus_type": stimulus_type,
            "steps": steps,
            "baseline_gain": baseline_gain,
            "target_gain": target_gain,
            "amplitude": amplitude,
            "duration_ms": duration_ms,
            "window_ms": window_ms,
            **sim.params.describe(),
        },
    )


# -- agreement with interventions ----------------------------------------


def precision_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    """Fraction of the top k that are genuinely in `truth`."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if t in truth) / len(top)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in scipy.stats.

    Ties are averaged, matching the standard definition.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # Average ranks within tied groups.
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


@dataclass
class AgreementReport:
    """How well a gradient ranking predicts measured graft effects.

    Note on reading precision@k: when k exceeds the number of "true" types,
    precision is capped at n_true/k however perfect the ranking. The
    `ceiling_at_k` field records that bound so a low score is not mistaken
    for a bad ranking. On the real connectome, with thousands of types, k=20
    sits well below n_true and the ceiling is 1.0.
    """

    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    spearman_rho: float
    n_types: int
    n_true: int
    gradient_method: str
    top_by_gradient: list[str]
    top_by_graft: list[str]

    def ceiling_at_k(self, k: int) -> float:
        return min(1.0, self.n_true / k) if k > 0 else 0.0

    def describe(self) -> dict:
        return {
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "precision_at_20": round(self.precision_at_20, 4),
            "ceiling_at_5": round(self.ceiling_at_k(5), 4),
            "ceiling_at_10": round(self.ceiling_at_k(10), 4),
            "ceiling_at_20": round(self.ceiling_at_k(20), 4),
            "spearman_rho": None
            if np.isnan(self.spearman_rho)
            else round(self.spearman_rho, 4),
            "n_types": self.n_types,
            "n_true": self.n_true,
            "gradient_method": self.gradient_method,
            "top_by_gradient": self.top_by_gradient,
            "top_by_graft": self.top_by_graft,
        }


def compare_to_grafts(
    attribution: AttributionResult,
    graft_effects: dict[str, float],
    top_fraction: float = 0.1,
    min_true: int = 1,
) -> AgreementReport:
    """Score a gradient ranking against measured graft effects.

    `graft_effects` maps cell type to the magnitude of readout change its
    graft produced. The "true" set is the top `top_fraction` of types by that
    measured effect — the ones a cheap method would need to find.
    """
    shared = [t for t in attribution.types if t in graft_effects]
    if not shared:
        raise ValueError("no overlap between attribution types and graft effects")

    measured = np.array([abs(graft_effects[t]) for t in shared])
    predicted = np.array([abs(attribution.score_for(t)) for t in shared])

    n_true = max(min_true, int(round(top_fraction * len(shared))))
    truth_order = np.argsort(-measured)
    truth = {shared[i] for i in truth_order[:n_true]}

    ranked = [t for t in attribution.ranking() if t[0] in set(shared)]
    ranked_types = [t for t, _ in ranked]

    return AgreementReport(
        precision_at_5=precision_at_k(ranked_types, truth, 5),
        precision_at_10=precision_at_k(ranked_types, truth, 10),
        precision_at_20=precision_at_k(ranked_types, truth, 20),
        spearman_rho=spearman(predicted, measured),
        n_types=len(shared),
        n_true=n_true,
        gradient_method=attribution.method,
        top_by_gradient=ranked_types[:10],
        top_by_graft=[shared[i] for i in truth_order[:10]],
    )
