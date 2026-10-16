"""Graph construction, with hand-computed answers where possible."""

import numpy as np
import pandas as pd
import pytest

from flyswap.data.schema import Connectome, DatasetSpec
from flyswap.graph import build_graph, degree_preserving_shuffle, type_level_matrix
from flyswap.graph.signs import SHIU2024, perturb


def tiny_connectome(counts, nts, sex="male"):
    """A 3-neuron connectome with explicit edges, for arithmetic by hand."""
    neurons = pd.DataFrame(
        {
            "body_id": [1, 2, 3],
            "cell_type": ["A", "B", "C"],
            "side": ["M", "M", "M"],
            "superclass": ["sensory", "intrinsic", "motor"],
            "nt": nts,
            "nt_conf": [1.0, 1.0, 1.0],
            "fru_dsx": ["none"] * 3,
        }
    )
    edges = pd.DataFrame(counts, columns=["pre", "post", "count"])
    spec = DatasetSpec(name="tiny", sex=sex, regions=("brain",), citation="", download="")
    c = Connectome(spec=spec, neurons=neurons, edges=edges)
    c.validate()
    return c


class TestInputFraction:
    def test_rows_sum_to_one(self, graphs):
        """Every neuron with input has input shares summing to 1."""
        for name, g in graphs.items():
            row_sums = np.abs(g.W).sum(axis=1).A.ravel()
            has_input = row_sums > 0
            assert np.allclose(row_sums[has_input], 1.0, atol=1e-6), name

    def test_hand_computed(self):
        """Neuron 3 gets 30 synapses from A and 10 from B -> 0.75 / 0.25."""
        c = tiny_connectome(
            [(1, 3, 30), (2, 3, 10)],
            ["acetylcholine", "acetylcholine", "acetylcholine"],
        )
        g = build_graph(c)
        assert g.W[2, 0] == pytest.approx(0.75)
        assert g.W[2, 1] == pytest.approx(0.25)

    def test_cancels_density_difference(self):
        """Doubling every synapse count leaves input fractions unchanged.

        This is the property that makes male and female connectomes
        comparable despite one being reconstructed more densely.
        """
        base = tiny_connectome(
            [(1, 3, 30), (2, 3, 10)], ["acetylcholine"] * 3
        )
        dense = tiny_connectome(
            [(1, 3, 60), (2, 3, 20)], ["acetylcholine"] * 3
        )
        gb, gd = build_graph(base), build_graph(dense)
        assert np.allclose(gb.W.toarray(), gd.W.toarray())

    def test_raw_normalisation_does_not_cancel(self):
        """The contrast case: raw counts do carry the density difference."""
        base = tiny_connectome([(1, 3, 30)], ["acetylcholine"] * 3)
        dense = tiny_connectome([(1, 3, 60)], ["acetylcholine"] * 3)
        gb = build_graph(base, normalisation="raw")
        gd = build_graph(dense, normalisation="raw")
        assert gd.W[2, 0] == pytest.approx(2 * gb.W[2, 0])


class TestSigns:
    def test_gaba_is_negative(self):
        c = tiny_connectome([(1, 3, 10), (2, 3, 10)], ["acetylcholine", "gaba", "gaba"])
        g = build_graph(c)
        assert g.W[2, 0] > 0, "cholinergic input should excite"
        assert g.W[2, 1] < 0, "GABAergic input should inhibit"

    def test_monoamines_have_no_fast_effect(self):
        c = tiny_connectome([(1, 3, 10)], ["dopamine", "gaba", "acetylcholine"])
        g = build_graph(c)
        assert g.W[2, 0] == 0.0

    def test_rule_mapping_complete(self):
        """Every transmitter in the vocabulary has a sign."""
        from flyswap.data.schema import TRANSMITTERS

        mapping = SHIU2024.as_dict
        assert set(TRANSMITTERS) <= set(mapping)

    def test_perturb_flips_requested_fraction(self):
        signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        out = perturb(signs, 0.5, seed=0)
        assert (out != signs).sum() == 5
        assert set(np.abs(out)) == {1.0}

    def test_perturb_leaves_zeros_alone(self):
        signs = np.array([0.0, 0.0, 1.0, -1.0])
        out = perturb(signs, 1.0, seed=0)
        assert out[0] == 0.0 and out[1] == 0.0


class TestTypeLevel:
    def test_shape_matches_type_count(self, graphs):
        g = graphs["male"]
        T, types = type_level_matrix(g)
        assert T.shape == (len(types), len(types))
        assert types == g.cell_types()

    def test_averages_not_sums(self):
        """Two identical postsynaptic neurons give the same type-level value
        as one, since the entry is a mean over members."""
        neurons = pd.DataFrame(
            {
                "body_id": [1, 2, 3],
                "cell_type": ["A", "B", "B"],
                "side": ["M"] * 3,
                "superclass": ["sensory", "motor", "motor"],
                "nt": ["acetylcholine"] * 3,
                "nt_conf": [1.0] * 3,
                "fru_dsx": ["none"] * 3,
            }
        )
        edges = pd.DataFrame([(1, 2, 10), (1, 3, 10)], columns=["pre", "post", "count"])
        spec = DatasetSpec("t", "male", ("brain",), "", "")
        c = Connectome(spec=spec, neurons=neurons, edges=edges)
        g = build_graph(c)
        T, types = type_level_matrix(g)
        # Each B neuron's sole input is A, so each input share is 1.0 and the
        # mean over the two B neurons is also 1.0.
        assert T[types.index("B"), types.index("A")] == pytest.approx(1.0)


class TestShuffle:
    def test_preserves_edge_count(self, graphs):
        g = graphs["male"]
        s = degree_preserving_shuffle(g, seed=1)
        assert s.nnz == g.nnz

    def test_preserves_degrees(self, graphs):
        g = graphs["male"]
        s = degree_preserving_shuffle(g, seed=1)
        in_before = np.diff(g.W.tocsr().indptr)
        in_after = np.diff(s.W.tocsr().indptr)
        assert np.array_equal(in_before, in_after)
        out_before = np.bincount(g.W.tocoo().col, minlength=g.n)
        out_after = np.bincount(s.W.tocoo().col, minlength=g.n)
        assert np.array_equal(out_before, out_after)

    def test_actually_rewires(self, graphs):
        g = graphs["male"]
        s = degree_preserving_shuffle(g, seed=1)
        before = set(zip(*g.W.tocoo().nonzero(), strict=True))
        after = set(zip(*s.W.tocoo().nonzero(), strict=True))
        assert len(before - after) > 0, "shuffle changed nothing"

    def test_deterministic_given_seed(self, graphs):
        a = degree_preserving_shuffle(graphs["male"], seed=3)
        b = degree_preserving_shuffle(graphs["male"], seed=3)
        assert np.allclose(a.W.toarray(), b.W.toarray())


class TestProvenance:
    def test_config_hash_changes_with_sign_rule(self, pair):
        a = build_graph(pair.male, sign_rule="shiu2024")
        b = build_graph(pair.male, sign_rule="glut_excitatory")
        assert a.config_hash != b.config_hash

    def test_config_records_normalisation(self, pair):
        g = build_graph(pair.male, normalisation="log")
        assert g.config["normalisation"] == "log"
