"""Rate model dynamics, checked against hand-computed values where possible."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flyswap.data.schema import Connectome, DatasetSpec
from flyswap.graph import build_graph
from flyswap.sim import RateParams, RateSimulator, calibrate_gain


def single_neuron_graph():
    """One neuron, no connections: dynamics reduce to a leak toward relu(I)."""
    neurons = pd.DataFrame(
        {
            "body_id": [1, 2],
            "cell_type": ["A", "B"],
            "side": ["M", "M"],
            "superclass": ["sensory", "motor"],
            "nt": ["acetylcholine", "acetylcholine"],
            "nt_conf": [1.0, 1.0],
            "fru_dsx": ["none", "none"],
        }
    )
    edges = pd.DataFrame([(1, 2, 10)], columns=["pre", "post", "count"])
    spec = DatasetSpec("t", "male", ("brain",), "", "")
    c = Connectome(spec=spec, neurons=neurons, edges=edges)
    return build_graph(c)


class TestDynamics:
    def test_zero_stimulus_stays_silent(self, graphs, params):
        """No input, no activity. Catches spurious self-excitation."""
        sim = RateSimulator(graphs["male"], params)
        steady = sim.steady_state(jnp.zeros(sim.n))
        assert float(jnp.max(steady)) == pytest.approx(0.0, abs=1e-6)

    def test_relaxes_toward_target(self):
        """An unconnected driven neuron approaches relu(I) exponentially.

        After t = 3*tau it should be within ~5% of its target.
        """
        g = single_neuron_graph()
        p = RateParams(tau_ms=20.0, dt_ms=1.0, gain=1.0)
        sim = RateSimulator(g, p)
        stim = sim.make_stimulus(np.array([0]), 10.0)
        trace = np.asarray(sim.run(stim, duration_ms=60.0))
        assert trace[-1, 0] == pytest.approx(10.0, rel=0.06)

    def test_euler_step_matches_closed_form(self):
        """One step of the integrator equals the formula it implements."""
        g = single_neuron_graph()
        p = RateParams(tau_ms=20.0, dt_ms=1.0, gain=1.0)
        sim = RateSimulator(g, p)
        stim = sim.make_stimulus(np.array([0]), 10.0)
        trace = np.asarray(sim.run(stim, duration_ms=1.0))
        # r starts at 0, so r1 = 0 + (dt/tau) * (relu(I) - 0) = 0.5
        assert trace[0, 0] == pytest.approx(0.5, abs=1e-5)

    def test_signal_propagates_downstream(self):
        g = single_neuron_graph()
        sim = RateSimulator(g, RateParams(gain=2.0))
        steady = sim.steady_state(sim.make_stimulus(np.array([0]), 50.0))
        assert float(steady[1]) > 0, "downstream neuron never activated"

    def test_inhibition_reduces_response(self, pair):
        """Flipping a transmitter to GABA must lower the downstream rate."""
        exc = build_graph(pair.male, sign_rule="shiu2024")
        sim_e = RateSimulator(exc, RateParams(gain=2.2))
        base = float(
            sim_e.readout(sim_e.steady_state(sim_e.stimulus_for_type("SENS", 50.0)), "PREMOT")
        )
        # INHIB is GABAergic in the synthetic data; treating glutamate and
        # GABA as excitatory should raise PREMOT.
        exc2 = build_graph(pair.male, sign_rule="glut_excitatory")
        sim_e2 = RateSimulator(exc2, RateParams(gain=2.2))
        alt = float(
            sim_e2.readout(
                sim_e2.steady_state(sim_e2.stimulus_for_type("SENS", 50.0)), "PREMOT"
            )
        )
        assert alt >= base

    def test_rate_ceiling_respected(self, graphs):
        sim = RateSimulator(graphs["male"], RateParams(gain=10.0, r_max=50.0))
        steady = np.asarray(sim.steady_state(sim.stimulus_for_type("SENS", 500.0)))
        assert steady.max() <= 50.0 + 1e-4

    def test_no_nans_under_strong_drive(self, graphs):
        sim = RateSimulator(graphs["male"], RateParams(gain=8.0))
        steady = np.asarray(sim.steady_state(sim.stimulus_for_type("SENS", 1000.0)))
        assert np.isfinite(steady).all()


class TestBatching:
    def test_batch_matches_individual_runs(self, graphs, params):
        """vmap must give exactly what a loop gives, or benchmarks lie."""
        sim = RateSimulator(graphs["male"], params)
        stimuli = np.stack(
            [
                np.asarray(sim.stimulus_for_type("SENS", a))
                for a in (10.0, 25.0, 50.0)
            ]
        )
        batched = np.asarray(sim.run_batch(stimuli))
        for i, a in enumerate((10.0, 25.0, 50.0)):
            one = np.asarray(sim.steady_state(sim.stimulus_for_type("SENS", a)))
            assert np.allclose(batched[i], one, atol=1e-4)

    def test_batch_shape(self, graphs, params):
        sim = RateSimulator(graphs["male"], params)
        stimuli = np.zeros((7, sim.n), dtype=np.float32)
        assert np.asarray(sim.run_batch(stimuli)).shape == (7, sim.n)


class TestReadouts:
    def test_type_means_match_manual(self, graphs, params):
        sim = RateSimulator(graphs["male"], params)
        steady = sim.steady_state(sim.stimulus_for_type("SENS", 50.0))
        means = np.asarray(sim.type_means(steady)).ravel()
        for i, t in enumerate(sim.type_names):
            rows = graphs["male"].rows_for_type(t)
            assert means[i] == pytest.approx(
                float(np.asarray(steady)[rows].mean()), rel=1e-4
            )

    def test_readout_of_missing_type_is_zero(self, graphs, params):
        sim = RateSimulator(graphs["male"], params)
        steady = sim.steady_state(sim.stimulus_for_type("SENS", 50.0))
        assert float(sim.readout(steady, "NOT_A_TYPE")) == 0.0

    def test_monotonic_in_stimulus_amplitude(self, graphs, params):
        """Dose-response: more drive should not produce less output."""
        sim = RateSimulator(graphs["male"], params)
        vals = [
            float(
                sim.readout(
                    sim.steady_state(sim.stimulus_for_type("SENS", a)), "MOTOR"
                )
            )
            for a in (5.0, 10.0, 25.0, 50.0)
        ]
        assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:], strict=False))


class TestCalibration:
    def test_picks_a_gain_in_range(self, graphs):
        res = calibrate_gain(graphs["male"], "SENS", "MOTOR")
        assert res.passed
        assert res.gain in res.criteria["gains_tested"]

    def test_agrees_across_animals(self, graphs):
        """Gain should not depend on which animal it is calibrated on."""
        gains = {
            name: calibrate_gain(g, "SENS", "MOTOR").gain
            for name, g in graphs.items()
        }
        assert len(set(gains.values())) == 1, gains

    def test_rejects_saturating_gains(self, graphs):
        res = calibrate_gain(graphs["male"], "SENS", "MOTOR")
        for row in res.table:
            if row["saturated_fraction"] > res.criteria["max_saturated_fraction"]:
                assert not row["passes"]
