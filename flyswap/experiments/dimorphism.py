"""E1: is a male-female difference bigger than a female-female difference?

This is the experiment the whole project is arranged around, and the reason
is a limitation published fly-brain simulations state plainly: there is one
animal per connectome, so a male-female difference cannot be separated from
the difference between any two individuals, or between two reconstructions.

The remedy is to measure both. Two female connectomes exist, so the
female-female contrast estimates everything that is *not* sex: individual
variation, reconstruction differences, annotation differences. A male-female
contrast estimates that plus sex. The difference between the two contrasts is
the only part attributable to sex, and it is reported with a bootstrap
interval over seeds so it comes with an honest error bar.

A readout only counts as dimorphic if that interval excludes zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyswap.graph.build import SignedGraph
from flyswap.sim.rate import RateParams, RateSimulator


@dataclass
class StimulusCondition:
    """One sensory population driven at one amplitude."""

    name: str
    stimulus_type: str
    readout_types: tuple[str, ...]
    amplitude: float = 50.0

    def describe(self) -> dict:
        return {
            "name": self.name,
            "stimulus_type": self.stimulus_type,
            "readout_types": list(self.readout_types),
            "amplitude": self.amplitude,
        }


@dataclass
class DimorphismScore:
    """The headline number for one stimulus/readout pair."""

    condition: str
    readout_type: str
    male_response: float
    female_responses: dict[str, float]
    cross_sex_distance: float
    within_sex_distance: float
    score: float
    ci_low: float
    ci_high: float
    effect_ratio: float
    ci_excludes_zero: bool
    significant: bool

    def describe(self) -> dict:
        return {
            "condition": self.condition,
            "readout_type": self.readout_type,
            "male_response": round(self.male_response, 4),
            "female_responses": {
                k: round(v, 4) for k, v in self.female_responses.items()
            },
            "cross_sex_distance": round(self.cross_sex_distance, 4),
            "within_sex_distance": round(self.within_sex_distance, 4),
            "score": round(self.score, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "effect_ratio": round(self.effect_ratio, 4),
            "ci_excludes_zero": self.ci_excludes_zero,
            "significant": self.significant,
        }


@dataclass
class DimorphismReport:
    scores: list[DimorphismScore]
    config: dict = field(default_factory=dict)

    def significant(self) -> list[DimorphismScore]:
        return [s for s in self.scores if s.significant]

    def describe(self) -> dict:
        return {
            "config": self.config,
            "n_scores": len(self.scores),
            "n_significant": len(self.significant()),
            "scores": [s.describe() for s in self.scores],
        }


def _responses_across_seeds(
    graph: SignedGraph,
    condition: StimulusCondition,
    readout_type: str,
    params: RateParams,
    seeds: list[int],
    jitter: float,
    duration_ms: float,
) -> np.ndarray:
    """Readout under small stimulus jitter, one value per seed.

    The rate model is deterministic, so repeated runs of one condition give an
    identical number and no error bar. Jittering the stimulus amplitude across
    seeds provides the variation a bootstrap needs, and mirrors the fact that
    a real stimulus is never delivered at exactly one intensity. The *same*
    seeds are used for every animal, so a paired comparison is possible.
    """
    sim = RateSimulator(graph, params)
    rows = graph.rows_for_type(condition.stimulus_type)
    if len(rows) == 0:
        return np.full(len(seeds), np.nan)

    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        amp = condition.amplitude * (1.0 + jitter * rng.standard_normal())
        stim = sim.make_stimulus(rows, float(max(amp, 0.0)))
        steady = sim.steady_state(stim, duration_ms=duration_ms)
        out.append(float(sim.readout(steady, readout_type)))
    return np.array(out)


def _relative_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference between two response vectors.

    Dividing by the mean magnitude makes the measure scale-free, so a readout
    with large absolute rates does not dominate one with small rates. The
    small constant keeps a pair of near-silent readouts from producing a huge
    ratio out of numerical noise.
    """
    denom = (np.abs(a) + np.abs(b)) / 2.0 + 1.0
    return np.abs(a - b) / denom


def run_dimorphism_experiment(
    male: SignedGraph,
    females: dict[str, SignedGraph],
    conditions: list[StimulusCondition],
    params: RateParams | None = None,
    seeds: tuple[int, ...] = tuple(range(10)),
    jitter: float = 0.08,
    duration_ms: float = 200.0,
    n_bootstrap: int = 2000,
    ci: float = 95.0,
    bootstrap_seed: int = 7,
    min_effect_ratio: float = 1.0,
) -> DimorphismReport:
    """Score every condition/readout pair for sex difference.

    Requires at least two female graphs: with one, there is no within-sex
    baseline and the experiment cannot be run as designed.

    A readout is called dimorphic only if it clears *two* bars:

    1. the bootstrap interval excludes zero, and
    2. the score is at least `min_effect_ratio` times the within-sex distance.

    The second bar exists because the first is too easy to clear. The rate
    model is deterministic, so the only variation between seeds comes from
    stimulus jitter, and that variation is highly correlated across animals.
    The resulting intervals are extremely narrow, and a difference far too
    small to mean anything biologically will still exclude zero. Requiring the
    sex effect to be at least as large as the individual-variation baseline is
    the statement the project actually wants to make: *this difference is
    bigger than the difference between two individuals*.
    """
    if len(females) < 2:
        raise ValueError(
            "need at least two female connectomes for a within-sex baseline; "
            f"got {sorted(females)}"
        )
    params = params or RateParams()
    female_names = sorted(females)
    seed_list = list(seeds)
    rng = np.random.default_rng(bootstrap_seed)
    alpha = (100.0 - ci) / 2.0

    scores: list[DimorphismScore] = []
    for condition in conditions:
        for readout_type in condition.readout_types:
            male_r = _responses_across_seeds(
                male, condition, readout_type, params, seed_list, jitter, duration_ms
            )
            female_r = {
                name: _responses_across_seeds(
                    females[name],
                    condition,
                    readout_type,
                    params,
                    seed_list,
                    jitter,
                    duration_ms,
                )
                for name in female_names
            }
            if np.isnan(male_r).any() or any(
                np.isnan(v).any() for v in female_r.values()
            ):
                continue

            # Cross-sex: male against each female, averaged.
            cross = np.mean(
                [_relative_distance(male_r, female_r[n]) for n in female_names], axis=0
            )
            # Within-sex: every female pair.
            within_pairs = [
                _relative_distance(female_r[a], female_r[b])
                for i, a in enumerate(female_names)
                for b in female_names[i + 1 :]
            ]
            within = np.mean(within_pairs, axis=0)

            per_seed = cross - within
            point = float(per_seed.mean())

            # Bootstrap over seeds: resample which seeds contribute.
            idx = rng.integers(0, len(seed_list), size=(n_bootstrap, len(seed_list)))
            boot = per_seed[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [alpha, 100.0 - alpha])

            within_mean = float(within.mean())
            effect_ratio = point / within_mean if within_mean > 1e-9 else np.inf
            excludes_zero = bool(lo > 0.0)

            scores.append(
                DimorphismScore(
                    condition=condition.name,
                    readout_type=readout_type,
                    male_response=float(male_r.mean()),
                    female_responses={
                        n: float(female_r[n].mean()) for n in female_names
                    },
                    cross_sex_distance=float(cross.mean()),
                    within_sex_distance=within_mean,
                    score=point,
                    ci_low=float(lo),
                    ci_high=float(hi),
                    effect_ratio=float(effect_ratio),
                    ci_excludes_zero=excludes_zero,
                    significant=bool(
                        excludes_zero and effect_ratio >= min_effect_ratio
                    ),
                )
            )

    return DimorphismReport(
        scores=scores,
        config={
            "seeds": seed_list,
            "jitter": jitter,
            "duration_ms": duration_ms,
            "n_bootstrap": n_bootstrap,
            "ci": ci,
            "bootstrap_seed": bootstrap_seed,
            "min_effect_ratio": min_effect_ratio,
            "female_datasets": female_names,
            "male_dataset": male.dataset,
            "conditions": [c.describe() for c in conditions],
            **params.describe(),
        },
    )
