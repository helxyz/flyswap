"""Differentiable rate model (Tier A).

Each neuron's firing rate relaxes toward a rectified sum of its signed,
normalised inputs:

    tau * dr/dt = -r + relu(g[type] * (W @ r) + I(t) - theta)

The model is deliberately simple. It has no free parameters fitted to data:
W comes from the connectome, the gains g are fixed at 1 unless an experiment
changes them, and tau/theta are shared constants. That matters because the
scientific claim is "the wiring explains this", and every fitted parameter
weakens it.

Three properties make this the workhorse of the project:

* **Fast.** One sparse matmul per step, jitted end to end.
* **Batched.** ``vmap`` over stimuli runs hundreds of conditions at once,
  which is what makes thousands of graft simulations affordable.
* **Differentiable.** The per-cell-type gains are real inputs to the
  computation, so one backward pass gives d(readout)/d(gain) for every type —
  the attribution layer's whole basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse
from scipy import sparse as spsparse

from flyswap.graph.build import SignedGraph


@dataclass(frozen=True)
class RateParams:
    """Constants of the rate model.

    tau_ms
        Membrane time constant. 20 ms matches the value used by published fly
        brain models.
    dt_ms
        Integration step. Must be well under tau for the Euler step to be
        stable; 1 ms gives 20 steps per time constant.
    theta
        Activation threshold on the summed input.
    r_max
        Ceiling on firing rate. Real neurons saturate, and without a ceiling a
        strongly recurrent connectome can diverge, which shows up as NaN
        rather than as the biologically meaningful "this circuit is driven
        hard".
    gain
        Global scale on recurrent input. With input-fraction weights the total
        input to a neuron is at most 1, so a gain above 1 is needed for
        activity to propagate at all. This is the one knob that must be
        calibrated, and it is calibrated once, globally, not per experiment.
    """

    tau_ms: float = 20.0
    dt_ms: float = 1.0
    theta: float = 0.0
    r_max: float = 200.0
    gain: float = 4.0

    @property
    def alpha(self) -> float:
        """Euler coefficient dt/tau."""
        return self.dt_ms / self.tau_ms

    def describe(self) -> dict:
        return {
            "model": "rate",
            "tau_ms": self.tau_ms,
            "dt_ms": self.dt_ms,
            "theta": self.theta,
            "r_max": self.r_max,
            "gain": self.gain,
        }


def to_bcoo(W: spsparse.csr_matrix) -> jsparse.BCOO:
    """SciPy CSR -> JAX BCOO, sorted and deduplicated for fast matmul."""
    coo = W.tocoo()
    indices = jnp.stack(
        [jnp.asarray(coo.row, dtype=jnp.int32), jnp.asarray(coo.col, dtype=jnp.int32)],
        axis=1,
    )
    data = jnp.asarray(coo.data, dtype=jnp.float32)
    mat = jsparse.BCOO((data, indices), shape=W.shape)
    return mat.sort_indices()


class RateSimulator:
    """Runs the rate model on one signed graph.

    The graph is converted to BCOO once at construction; repeated runs reuse
    it. Methods are thin wrappers over jitted functions so that the tracing
    cost is paid on the first call only.
    """

    def __init__(self, graph: SignedGraph, params: RateParams | None = None):
        self.graph = graph
        self.params = params or RateParams()
        self.W = to_bcoo(graph.W)
        self.n = graph.n
        codes, type_names = graph.type_codes()
        self.type_codes = jnp.asarray(codes, dtype=jnp.int32)
        self.type_names = type_names
        self.n_types = len(type_names)

    # -- core dynamics ----------------------------------------------------

    @partial(jax.jit, static_argnums=(0, 4))
    def _run(
        self,
        W_data: jnp.ndarray,
        stimulus: jnp.ndarray,
        type_gains: jnp.ndarray,
        n_steps: int,
    ) -> jnp.ndarray:
        """Integrate for n_steps, returning the full rate trace (n_steps, n).

        W_data is passed separately from the sparsity pattern so that grafts,
        which change weights but not structure, do not force a retrace.
        """
        p = self.params
        W = jsparse.BCOO((W_data, self.W.indices), shape=self.W.shape)
        per_neuron_gain = type_gains[self.type_codes]

        def step(r, _):
            drive = p.gain * per_neuron_gain * (W @ r) + stimulus
            target = jnp.clip(jax.nn.relu(drive - p.theta), 0.0, p.r_max)
            r_next = r + p.alpha * (target - r)
            return r_next, r_next

        r0 = jnp.zeros(self.n, dtype=jnp.float32)
        _, trace = jax.lax.scan(step, r0, None, length=n_steps)
        return trace

    def run(
        self,
        stimulus: np.ndarray | jnp.ndarray,
        duration_ms: float = 200.0,
        type_gains: np.ndarray | jnp.ndarray | None = None,
        W_data: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Rate trace for one stimulus. Shape (n_steps, n_neurons)."""
        n_steps = int(round(duration_ms / self.params.dt_ms))
        gains = (
            jnp.ones(self.n_types, dtype=jnp.float32)
            if type_gains is None
            else jnp.asarray(type_gains, dtype=jnp.float32)
        )
        data = self.W.data if W_data is None else W_data
        return self._run(data, jnp.asarray(stimulus, dtype=jnp.float32), gains, n_steps)

    def steady_state(
        self,
        stimulus: np.ndarray | jnp.ndarray,
        duration_ms: float = 200.0,
        window_ms: float = 50.0,
        type_gains: np.ndarray | jnp.ndarray | None = None,
        W_data: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Mean rate over the final `window_ms`. Shape (n_neurons,).

        Averaging a window rather than taking the last step smooths over any
        residual oscillation, and is the readout every experiment uses.
        """
        trace = self.run(stimulus, duration_ms, type_gains, W_data)
        n_window = max(1, int(round(window_ms / self.params.dt_ms)))
        return trace[-n_window:].mean(axis=0)

    # -- batching ---------------------------------------------------------

    def run_batch(
        self,
        stimuli: np.ndarray | jnp.ndarray,
        duration_ms: float = 200.0,
        window_ms: float = 50.0,
        type_gains: np.ndarray | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Steady states for a batch of stimuli. Shape (batch, n_neurons).

        This is where the GPU earns its place: the sparse matrix is shared
        across the batch, so B conditions cost far less than B separate runs.
        """
        stimuli = jnp.asarray(stimuli, dtype=jnp.float32)
        gains = (
            jnp.ones(self.n_types, dtype=jnp.float32)
            if type_gains is None
            else jnp.asarray(type_gains, dtype=jnp.float32)
        )
        n_steps = int(round(duration_ms / self.params.dt_ms))
        n_window = max(1, int(round(window_ms / self.params.dt_ms)))

        def one(stim):
            trace = self._run(self.W.data, stim, gains, n_steps)
            return trace[-n_window:].mean(axis=0)

        return jax.vmap(one)(stimuli)

    # -- stimulus helpers -------------------------------------------------

    def make_stimulus(self, rows: np.ndarray, amplitude: float = 50.0) -> jnp.ndarray:
        """A stimulus vector driving the given neuron rows."""
        stim = np.zeros(self.n, dtype=np.float32)
        if len(rows):
            stim[np.asarray(rows, dtype=np.int64)] = amplitude
        return jnp.asarray(stim)

    def stimulus_for_type(self, cell_type: str, amplitude: float = 50.0) -> jnp.ndarray:
        return self.make_stimulus(self.graph.rows_for_type(cell_type), amplitude)

    # -- readouts ---------------------------------------------------------

    def type_means(self, rates: jnp.ndarray) -> jnp.ndarray:
        """Mean rate per cell type. Shape (..., n_types).

        Per-type means, not per-neuron rates, are the comparable quantity
        across datasets: two connectomes have different neuron counts and no
        neuron-to-neuron correspondence, but they share cell types.
        """
        rates = jnp.atleast_2d(rates)
        sums = jax.ops.segment_sum(
            rates.T, self.type_codes, num_segments=self.n_types
        ).T
        counts = jax.ops.segment_sum(
            jnp.ones(self.n), self.type_codes, num_segments=self.n_types
        )
        return sums / jnp.maximum(counts, 1.0)

    def readout(self, rates: jnp.ndarray, cell_type: str) -> jnp.ndarray:
        """Mean rate of one cell type, the scalar an experiment reports."""
        rows = self.graph.rows_for_type(cell_type)
        if len(rows) == 0:
            return jnp.asarray(0.0)
        return rates[..., jnp.asarray(rows)].mean(axis=-1)

    def __repr__(self) -> str:
        return (
            f"RateSimulator(n={self.n:,}, nnz={self.W.nnz:,}, "
            f"types={self.n_types}, gain={self.params.gain})"
        )
