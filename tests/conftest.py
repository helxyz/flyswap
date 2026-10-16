"""Shared fixtures. Synthetic data only — no test needs a download."""

import pytest

from flyswap.data.synthetic import make_pair
from flyswap.graph import build_graph
from flyswap.interventions import TypeSpace
from flyswap.sim import RateParams

#: Gain chosen by flyswap.sim.calibrate on the isomorphic control pathway
#: (SENS -> MOTOR), identical for all three synthetic animals.
CALIBRATED_GAIN = 2.2


@pytest.fixture(scope="session")
def pair():
    return make_pair()


@pytest.fixture(scope="session")
def graphs(pair):
    return {name: build_graph(c) for name, c in pair.as_dict().items()}


@pytest.fixture(scope="session")
def space(graphs):
    return TypeSpace(graphs)


@pytest.fixture(scope="session")
def params():
    return RateParams(gain=CALIBRATED_GAIN)
