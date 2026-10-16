"""Choosing the one free parameter, once, by rule.

The rate model has a single global gain. With input-fraction weights a
neuron's total input is bounded by 1, so some amplification is needed for
activity to cross more than one synapse, but too much drives the network into
saturation where every neuron sits at its ceiling and all differences vanish.

Calibration picks that gain from stated criteria rather than by eye, on
control stimuli only, *before* any experimental comparison runs. Doing it per
experiment would mean tuning until the hypothesis came out right.

The criteria, in the order they are applied:

1. **Not saturated** — under a fraction of neurons at the rate ceiling.
2. **Propagates** — a stimulus reaches neurons several synapses downstream.
3. **Decays** — activity dies after the stimulus ends, rather than locking on.

Among gains that pass, the largest is chosen: it gives the strongest signal
without violating any criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyswap.graph.build import SignedGraph
from flyswap.sim.rate import RateParams, RateSimulator


@dataclass
class CalibrationCriteria:
    max_saturated_fraction: float = 0.02
    min_downstream_rate: float = 0.5
    max_residual_fraction: float = 0.05
    gains: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0)

    def describe(self) -> dict:
        return {
            "max_saturated_fraction": self.max_saturated_fraction,
            "min_downstream_rate": self.min_downstream_rate,
            "max_residual_fraction": self.max_residual_fraction,
            "gains_tested": list(self.gains),
        }


@dataclass
class CalibrationResult:
    gain: float | None
    table: list[dict]
    criteria: dict
    stimulus_type: str
    downstream_type: str

    @property
    def passed(self) -> bool:
        return self.gain is not None

    def describe(self) -> dict:
        return {
            "chosen_gain": self.gain,
            "stimulus_type": self.stimulus_type,
            "downstream_type": self.downstream_type,
            "criteria": self.criteria,
            "table": self.table,
        }


def calibrate_gain(
    graph: SignedGraph,
    stimulus_type: str,
    downstream_type: str,
    criteria: CalibrationCriteria | None = None,
    amplitude: float = 50.0,
    duration_ms: float = 200.0,
    decay_ms: float = 150.0,
    base_params: RateParams | None = None,
) -> CalibrationResult:
    """Sweep gains and return the largest that passes every criterion.

    `stimulus_type` and `downstream_type` should be a control pathway known to
    transmit — not the circuit under study.
    """
    criteria = criteria or CalibrationCriteria()
    base = base_params or RateParams()
    table: list[dict] = []
    passing: list[float] = []

    for gain in criteria.gains:
        params = RateParams(
            tau_ms=base.tau_ms,
            dt_ms=base.dt_ms,
            theta=base.theta,
            r_max=base.r_max,
            gain=gain,
        )
        sim = RateSimulator(graph, params)
        stim = sim.stimulus_for_type(stimulus_type, amplitude)

        trace = np.asarray(sim.run(stim, duration_ms))
        n_window = max(1, int(round(50.0 / params.dt_ms)))
        steady = trace[-n_window:].mean(axis=0)

        saturated = float((steady >= 0.99 * params.r_max).mean())
        downstream = float(
            np.asarray(sim.readout(np.asarray(steady), downstream_type))
        )

        # Decay: continue from the driven state with no stimulus.
        zero = np.zeros(sim.n, dtype=np.float32)
        decay_trace = np.asarray(sim.run(zero, decay_ms))
        peak = max(float(steady.mean()), 1e-9)
        residual = float(decay_trace[-1].mean() / peak)

        ok = (
            saturated <= criteria.max_saturated_fraction
            and downstream >= criteria.min_downstream_rate
            and residual <= criteria.max_residual_fraction
        )
        table.append(
            {
                "gain": gain,
                "saturated_fraction": round(saturated, 4),
                "downstream_rate": round(downstream, 4),
                "residual_fraction": round(residual, 4),
                "passes": ok,
            }
        )
        if ok:
            passing.append(gain)

    return CalibrationResult(
        gain=max(passing) if passing else None,
        table=table,
        criteria=criteria.describe(),
        stimulus_type=stimulus_type,
        downstream_type=downstream_type,
    )
