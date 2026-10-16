"""The tests that decide whether the method works.

The synthetic data has a dimorphism deliberately wired in: COURT_B's
connections are strong in the male and weak in the female, so stimulating
SENS should drive SONG in the male only, while MOTOR should behave the same
in both sexes.

Everything here asks one question in different ways: **does the pipeline
recover what we planted, and does it stay quiet where we planted nothing?**
A method that cannot find a difference placed there on purpose will not find
a real one, and a method that reports differences everywhere is useless even
when it happens to be right.
"""

import numpy as np
import pytest

from flyswap.attribution import compare_to_grafts, integrated_gradients
from flyswap.data.synthetic import (
    PLANTED_DIMORPHIC_TYPES,
    PLANTED_NULL_READOUT,
    PLANTED_READOUT,
    PLANTED_STIMULUS,
)
from flyswap.experiments import StimulusCondition, run_dimorphism_experiment
from flyswap.interventions import delete_type, graft_type, null_graft
from flyswap.sim import RateSimulator


def readout(graph, params, readout_type=PLANTED_READOUT, amplitude=50.0):
    sim = RateSimulator(graph, params)
    steady = sim.steady_state(sim.stimulus_for_type(PLANTED_STIMULUS, amplitude))
    return float(sim.readout(steady, readout_type))


@pytest.fixture(scope="module")
def graft_effects(graphs, space, params):
    """Readout change from grafting each type, female -> male."""
    base = readout(graphs["male"], params)
    effects = {}
    for t in graphs["male"].cell_types():
        grafted, _ = graft_type(
            graphs["male"], graphs["female_a"], t, space, "male", "female_a"
        )
        effects[t] = base - readout(grafted, params)
    return effects


class TestPlantedDimorphismIsDetected:
    def test_male_song_exceeds_female(self, graphs, params):
        """The planted circuit: SENS drives SONG in the male only."""
        male = readout(graphs["male"], params)
        fa = readout(graphs["female_a"], params)
        fb = readout(graphs["female_b"], params)
        assert male > 5 * max(fa, fb), f"male={male}, females={fa},{fb}"

    def test_isomorphic_readout_is_similar(self, graphs, params):
        """The control: MOTOR should not differ much between sexes.

        A loose tolerance is right here. The male connectome is generated
        1.8x denser, and normalisation cancels that only to the extent the
        difference is uniform, which it is not exactly.
        """
        male = readout(graphs["male"], params, PLANTED_NULL_READOUT)
        fa = readout(graphs["female_a"], params, PLANTED_NULL_READOUT)
        ratio = male / max(fa, 1e-9)
        assert 0.5 < ratio < 2.0, f"isomorphic readout differs {ratio:.2f}x"

    def test_experiment_flags_only_dimorphic_readouts(self, graphs, params):
        condition = StimulusCondition(
            "sensory_drive",
            PLANTED_STIMULUS,
            (PLANTED_READOUT, PLANTED_NULL_READOUT, "PREMOT", "RELAY"),
        )
        report = run_dimorphism_experiment(
            graphs["male"],
            {"female_a": graphs["female_a"], "female_b": graphs["female_b"]},
            [condition],
            params,
        )
        flagged = {s.readout_type for s in report.significant()}
        assert PLANTED_READOUT in flagged
        assert PLANTED_NULL_READOUT not in flagged
        assert "RELAY" not in flagged

    def test_effect_ratio_separates_signal_from_noise(self, graphs, params):
        """The planted readout's effect ratio must dwarf the control's."""
        condition = StimulusCondition(
            "sensory_drive", PLANTED_STIMULUS, (PLANTED_READOUT, PLANTED_NULL_READOUT)
        )
        report = run_dimorphism_experiment(
            graphs["male"],
            {"female_a": graphs["female_a"], "female_b": graphs["female_b"]},
            [condition],
            params,
        )
        by_type = {s.readout_type: s for s in report.scores}
        assert (
            by_type[PLANTED_READOUT].effect_ratio
            > 10 * by_type[PLANTED_NULL_READOUT].effect_ratio
        )

    def test_requires_two_females(self, graphs, params):
        """Without a within-sex baseline the experiment must refuse to run."""
        condition = StimulusCondition("c", PLANTED_STIMULUS, (PLANTED_READOUT,))
        with pytest.raises(ValueError, match="at least two female"):
            run_dimorphism_experiment(
                graphs["male"], {"female_a": graphs["female_a"]}, [condition], params
            )


class TestGraftsRecoverTheCarrier:
    def test_planted_type_has_largest_effect(self, graft_effects):
        """COURT_B carries the difference, so its graft should move the
        readout furthest. Its circuit neighbours move it too, which is
        correct: a graft changes a type's inputs and outputs together."""
        ranked = sorted(graft_effects, key=lambda t: -abs(graft_effects[t]))
        assert ranked[0] in PLANTED_DIMORPHIC_TYPES, f"ranking was {ranked[:3]}"

    def test_graft_moves_most_of_the_way(self, graphs, space, params, graft_effects):
        base_m = readout(graphs["male"], params)
        base_f = readout(graphs["female_a"], params)
        grafted, _ = graft_type(
            graphs["male"],
            graphs["female_a"],
            PLANTED_DIMORPHIC_TYPES[0],
            space,
            "male",
            "female_a",
        )
        value = readout(grafted, params)
        moved = (base_m - value) / (base_m - base_f)
        assert moved > 0.8, f"graft moved only {moved:.1%} toward female"

    def test_isomorphic_grafts_do_almost_nothing(self, graft_effects):
        """Grafting a type with no planted difference should barely register."""
        planted_effect = max(abs(graft_effects[t]) for t in PLANTED_DIMORPHIC_TYPES)
        for t in ("MOTOR", "PREMOT", "RELAY", "INHIB", "BKGD"):
            assert abs(graft_effects[t]) < 0.1 * planted_effect, t

    def test_null_graft_is_near_silent(self, graphs, space, params):
        """A graft between two females measures individual variation only."""
        base = readout(graphs["female_a"], params)
        grafted, spec = null_graft(
            graphs["female_a"],
            graphs["female_b"],
            PLANTED_DIMORPHIC_TYPES[0],
            space,
            "female_a",
            "female_b",
        )
        assert spec.kind == "null_graft"
        change = abs(base - readout(grafted, params))
        assert change < 0.2 * max(base, 1.0), f"null graft changed readout by {change}"

    def test_null_graft_rejects_cross_sex_donor(self, graphs, space):
        with pytest.raises(ValueError, match="same-sex"):
            null_graft(
                graphs["male"], graphs["female_a"], "COURT_B", space, "male", "female_a"
            )

    def test_graft_preserves_host_neuron_count(self, graphs, space):
        grafted, _ = graft_type(
            graphs["male"], graphs["female_a"], "COURT_B", space, "male", "female_a"
        )
        assert grafted.n == graphs["male"].n
        assert len(grafted.neurons) == len(graphs["male"].neurons)

    def test_graft_is_recorded_in_config(self, graphs, space):
        grafted, _ = graft_type(
            graphs["male"], graphs["female_a"], "COURT_B", space, "male", "female_a"
        )
        assert grafted.config["graft"]["cell_type"] == "COURT_B"
        assert grafted.config["graft"]["donor"] == "female_a"

    def test_unknown_type_raises(self, graphs, space):
        with pytest.raises(KeyError):
            graft_type(
                graphs["male"], graphs["female_a"], "NOPE", space, "male", "female_a"
            )


class TestDeletion:
    def test_deleting_the_carrier_abolishes_the_response(self, graphs, params):
        base = readout(graphs["male"], params)
        deleted, spec = delete_type(graphs["male"], PLANTED_DIMORPHIC_TYPES[0], "male")
        assert spec.kind == "deletion"
        assert readout(deleted, params) < 0.3 * base

    def test_deleting_an_unrelated_type_does_little(self, graphs, params):
        base = readout(graphs["male"], params)
        deleted, _ = delete_type(graphs["male"], "MOTOR", "male")
        assert readout(deleted, params) == pytest.approx(base, rel=0.1)


class TestAttributionAgreesWithGrafts:
    def test_top_gradient_types_match_top_graft_types(
        self, graphs, params, graft_effects
    ):
        sim = RateSimulator(graphs["male"], params)
        attribution = integrated_gradients(sim, PLANTED_STIMULUS, PLANTED_READOUT)
        top_grad = set(attribution.top(3))
        top_graft = {
            t for t in sorted(graft_effects, key=lambda x: -abs(graft_effects[x]))[:3]
        }
        assert top_grad == top_graft, f"gradient {top_grad} vs graft {top_graft}"

    def test_rank_correlation_is_high(self, graphs, params, graft_effects):
        sim = RateSimulator(graphs["male"], params)
        attribution = integrated_gradients(sim, PLANTED_STIMULUS, PLANTED_READOUT)
        report = compare_to_grafts(attribution, graft_effects, top_fraction=0.25)
        assert report.spearman_rho > 0.7, report.describe()

    def test_attribution_is_near_zero_for_irrelevant_types(self, graphs, params):
        sim = RateSimulator(graphs["male"], params)
        attribution = integrated_gradients(sim, PLANTED_STIMULUS, PLANTED_READOUT)
        carrier = abs(attribution.score_for(PLANTED_DIMORPHIC_TYPES[0]))
        for t in ("MOTOR", "PREMOT"):
            assert abs(attribution.score_for(t)) < 0.05 * carrier, t


class TestNullModels:
    def test_shuffled_graph_loses_the_pathway(self, graphs, params):
        """The key control: if the planted response survives randomised
        wiring, it was never about the wiring."""
        from flyswap.graph import degree_preserving_shuffle

        base = readout(graphs["male"], params)
        losses = []
        for seed in range(5):
            shuffled = degree_preserving_shuffle(graphs["male"], seed=seed)
            losses.append(readout(shuffled, params))
        assert np.mean(losses) < 0.5 * base, (
            f"shuffled graphs gave {np.mean(losses):.2f} vs real {base:.2f}"
        )
