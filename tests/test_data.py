"""Schema validation, loaders and provenance."""

import pandas as pd
import pytest

from flyswap.data.loaders import MissingDataError, _rename_first_match
from flyswap.data.schema import (
    Connectome,
    DatasetSpec,
    config_digest,
    normalise_nt,
)
from flyswap.data.synthetic import make_connectome, make_pair


def base_neurons():
    return pd.DataFrame(
        {
            "body_id": [1, 2],
            "cell_type": ["A", "B"],
            "side": ["L", "R"],
            "superclass": ["sensory", "motor"],
            "nt": ["acetylcholine", "gaba"],
            "nt_conf": [0.9, 0.8],
            "fru_dsx": ["none", "fru"],
        }
    )


SPEC = DatasetSpec("t", "male", ("brain",), "", "")


class TestValidation:
    def test_accepts_wellformed(self):
        c = Connectome(
            spec=SPEC,
            neurons=base_neurons(),
            edges=pd.DataFrame([(1, 2, 5)], columns=["pre", "post", "count"]),
        )
        c.validate()

    def test_rejects_dangling_edge(self):
        c = Connectome(
            spec=SPEC,
            neurons=base_neurons(),
            edges=pd.DataFrame([(1, 99, 5)], columns=["pre", "post", "count"]),
        )
        with pytest.raises(ValueError, match="not in neurons"):
            c.validate()

    def test_rejects_duplicate_body_id(self):
        neurons = base_neurons()
        neurons.loc[1, "body_id"] = 1
        c = Connectome(
            spec=SPEC,
            neurons=neurons,
            edges=pd.DataFrame([(1, 1, 5)], columns=["pre", "post", "count"]),
        )
        with pytest.raises(ValueError, match="duplicate body_id"):
            c.validate()

    def test_rejects_nonpositive_count(self):
        c = Connectome(
            spec=SPEC,
            neurons=base_neurons(),
            edges=pd.DataFrame([(1, 2, 0)], columns=["pre", "post", "count"]),
        )
        with pytest.raises(ValueError, match="positive"):
            c.validate()

    def test_rejects_unknown_transmitter(self):
        neurons = base_neurons()
        neurons.loc[0, "nt"] = "nonsense"
        c = Connectome(
            spec=SPEC,
            neurons=neurons,
            edges=pd.DataFrame([(1, 2, 5)], columns=["pre", "post", "count"]),
        )
        with pytest.raises(ValueError, match="transmitter"):
            c.validate()


class TestTransmitterNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ACh", "acetylcholine"),
            ("acetylcholine", "acetylcholine"),
            ("GABA", "gaba"),
            ("Glut", "glutamate"),
            ("5HT", "serotonin"),
            ("", "unknown"),
            (None, "unknown"),
            ("something new", "unknown"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalise_nt(raw) == expected


class TestSelection:
    def test_select_by_type_and_side(self, pair):
        rows = pair.male.select(cell_type="SENS", side="L")
        assert len(rows) > 0
        types = pair.male.neurons.loc[rows, "cell_type"].unique()
        sides = pair.male.neurons.loc[rows, "side"].unique()
        assert list(types) == ["SENS"] and list(sides) == ["L"]

    def test_select_by_regex(self, pair):
        rows = pair.male.select(type_regex=r"^COURT")
        types = set(pair.male.neurons.loc[rows, "cell_type"])
        assert types == {"COURT_A", "COURT_B"}

    def test_index_of_roundtrip(self, pair):
        ids = pair.male.neurons["body_id"].to_numpy()[:5]
        idx = pair.male.index_of(ids)
        assert list(pair.male.neurons.loc[idx, "body_id"]) == list(ids)


class TestPersistence:
    def test_save_and_load(self, tmp_path, pair):
        path = pair.male.save(tmp_path / "male")
        restored = Connectome.load(path)
        assert restored.n_neurons == pair.male.n_neurons
        assert restored.n_edges == pair.male.n_edges
        assert restored.spec.sex == "male"
        assert restored.provenance["synthetic"] is True


class TestSyntheticProperties:
    def test_density_factor_applied(self):
        light = make_connectome("a", "male", seed=1, density_factor=1.0)
        heavy = make_connectome("b", "male", seed=1, density_factor=2.0)
        ratio = heavy.n_synapses / light.n_synapses
        assert 1.9 < ratio < 2.1

    def test_structure_shared_across_animals(self, pair):
        """Same connected pairs in every animal, so types are comparable."""
        edges = {
            name: set(zip(c.edges["pre"], c.edges["post"], strict=True))
            for name, c in pair.as_dict().items()
        }
        assert edges["female_a"] == edges["female_b"]

    def test_male_has_extra_courtship_strength(self, pair):
        """The planted difference exists in the raw counts."""

        def mean_count(conn, pre_type, post_type):
            n = conn.neurons
            pre_ids = set(n.loc[n["cell_type"] == pre_type, "body_id"])
            post_ids = set(n.loc[n["cell_type"] == post_type, "body_id"])
            e = conn.edges
            sel = e[e["pre"].isin(pre_ids) & e["post"].isin(post_ids)]
            return sel["count"].mean() if len(sel) else 0.0

        male = mean_count(pair.male, "COURT_B", "SONG")
        female = mean_count(pair.female_a, "COURT_B", "SONG")
        assert male > 5 * female

    def test_reproducible_given_seed(self):
        a = make_connectome("x", "male", seed=42)
        b = make_connectome("x", "male", seed=42)
        pd.testing.assert_frame_equal(a.edges, b.edges)


class TestLoaderHelpers:
    def test_rename_picks_first_available(self):
        df = pd.DataFrame({"bodyId": [1], "weight": [5]})
        out = _rename_first_match(
            df,
            {"body_id": ["bodyId", "body_id"], "count": ["weight", "count"]},
            "test",
            required=["body_id", "count"],
        )
        assert set(out.columns) == {"body_id", "count"}

    def test_rename_reports_missing_clearly(self):
        df = pd.DataFrame({"irrelevant": [1]})
        with pytest.raises(ValueError, match="could not find columns"):
            _rename_first_match(
                df, {"body_id": ["bodyId"]}, "test", required=["body_id"]
            )

    def test_missing_data_error_names_source(self, tmp_path):
        err = MissingDataError("malecns", tmp_path, ["connectome-weights.feather"])
        assert "male-cns.janelia.org" in str(err)
        assert "Cite:" in str(err)


class TestProvenance:
    def test_config_digest_is_stable(self):
        a = config_digest({"x": 1, "y": [2, 3]})
        b = config_digest({"y": [2, 3], "x": 1})
        assert a == b

    def test_config_digest_changes_with_content(self):
        assert config_digest({"x": 1}) != config_digest({"x": 2})


def test_pair_has_two_distinct_females():
    pair = make_pair()
    assert pair.female_a.spec.sex == pair.female_b.spec.sex == "female"
    assert pair.male.spec.sex == "male"
    assert not pair.female_a.edges.equals(pair.female_b.edges)
