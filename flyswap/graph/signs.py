"""Turning predicted transmitters into connection signs.

The connectome measures how many synapses join two neurons, not whether they
excite or inhibit. Sign comes from each neuron's predicted neurotransmitter,
and the prediction is a classifier output with a non-trivial error rate. Every
sign in this project is therefore an assumption, and assumptions get versioned
so a result can be re-run under a different one.

Each rule is a frozen mapping with a name and a version. The name is recorded
in provenance; `perturb` flips a fraction of signs for sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignRule:
    """A named mapping from transmitter to +1 / -1 / 0."""

    name: str
    version: int
    mapping: tuple[tuple[str, int], ...]
    note: str = ""

    @property
    def as_dict(self) -> dict[str, int]:
        return dict(self.mapping)

    def apply(self, nt: pd.Series) -> np.ndarray:
        """Sign per neuron, defaulting to 0 for anything unmapped."""
        table = self.as_dict
        return nt.map(lambda x: table.get(str(x), 0)).to_numpy(dtype=np.float32)

    def describe(self) -> dict:
        return {
            "sign_rule": self.name,
            "version": self.version,
            "mapping": self.as_dict,
            "note": self.note,
        }


#: The default, following Shiu et al. (2024) as used by published fly brain
#: models: acetylcholine excites, GABA and glutamate inhibit, histamine
#: inhibits, and monoamines have no fast effect in a spike-only model.
SHIU2024 = SignRule(
    name="shiu2024",
    version=1,
    mapping=(
        ("acetylcholine", 1),
        ("gaba", -1),
        ("glutamate", -1),
        ("histamine", -1),
        ("dopamine", 0),
        ("serotonin", 0),
        ("octopamine", 0),
        ("unknown", 0),
    ),
    note="Monoamines modulate rather than drive; excluded from fast dynamics.",
)

#: Sensitivity variant: glutamate treated as excitatory. Glutamate's sign in
#: the fly depends on the receptors of the *target*, which the connectome does
#: not record, so this is a genuine uncertainty rather than a contrived knob.
GLUT_EXCITATORY = SignRule(
    name="glut_excitatory",
    version=1,
    mapping=tuple(
        {**SHIU2024.as_dict, "glutamate": 1}.items()
    ),
    note="Tests whether conclusions survive the glutamate sign assumption.",
)

#: Sensitivity variant: monoamines excite. Crude, but bounds the effect of
#: ignoring neuromodulation entirely.
MONOAMINE_EXCITATORY = SignRule(
    name="monoamine_excitatory",
    version=1,
    mapping=tuple(
        {
            **SHIU2024.as_dict,
            "dopamine": 1,
            "serotonin": 1,
            "octopamine": 1,
        }.items()
    ),
    note="Upper bound on the influence of modulatory neurons.",
)

RULES: dict[str, SignRule] = {
    r.name: r for r in (SHIU2024, GLUT_EXCITATORY, MONOAMINE_EXCITATORY)
}


def get_rule(name: str) -> SignRule:
    if name not in RULES:
        raise KeyError(f"unknown sign rule {name!r}; have {sorted(RULES)}")
    return RULES[name]


def perturb(
    signs: np.ndarray,
    fraction: float,
    seed: int,
    only_nonzero: bool = True,
) -> np.ndarray:
    """Flip the sign of a random `fraction` of neurons.

    Used for the sensitivity experiment: if a conclusion dissolves when 10% of
    signs flip, it was resting on the transmitter classifier rather than on
    the wiring. Neurons with sign 0 are left alone by default, since flipping
    "no effect" has no meaning.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    out = signs.copy()
    candidates = np.flatnonzero(signs != 0) if only_nonzero else np.arange(len(signs))
    if len(candidates) == 0:
        return out
    n_flip = int(round(fraction * len(candidates)))
    if n_flip == 0:
        return out
    chosen = rng.choice(candidates, size=n_flip, replace=False)
    out[chosen] *= -1
    return out
