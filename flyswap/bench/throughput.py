"""Measuring how fast the simulator actually is.

Three numbers matter for this project:

* **Neuron-steps per second** — raw throughput, the figure that scales to the
  real connectome's 166,700 neurons.
* **Batch speedup** — running B stimuli together versus one at a time. The
  sparse matrix is shared across the batch, so this is where the GPU pays
  off, and it is what makes thousands of grafts affordable.
* **Gradient versus grafts** — one backward pass ranks every cell type; doing
  it by brute force needs one simulation per type. The ratio is the argument
  for the attribution layer existing at all.

Timings exclude compilation. JAX traces and compiles on first call, so a
benchmark that includes the first call measures the compiler, not the kernel.
Every measurement here runs a warm-up first and blocks on the result, since
JAX dispatches asynchronously and an unblocked timing measures queueing.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import jax
import numpy as np

from flyswap.graph.build import SignedGraph
from flyswap.sim.rate import RateParams, RateSimulator


@dataclass
class BenchResult:
    label: str
    n_neurons: int
    nnz: int
    batch: int
    steps: int
    seconds: float
    neuron_steps_per_s: float
    extra: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "label": self.label,
            "n_neurons": self.n_neurons,
            "nnz": self.nnz,
            "batch": self.batch,
            "steps": self.steps,
            "seconds": round(self.seconds, 4),
            "neuron_steps_per_s": round(self.neuron_steps_per_s, 1),
            **self.extra,
        }


def environment() -> dict:
    """What the numbers were measured on. Meaningless without it."""
    try:
        devices = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    except Exception:  # pragma: no cover - device query is environment-specific
        devices = ["unknown"]
    return {
        "jax_version": jax.__version__,
        "jax_devices": devices,
        "default_backend": jax.default_backend(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "x64_enabled": jax.config.read("jax_enable_x64"),
    }


def _timed(fn, repeats: int = 3) -> float:
    """Best of `repeats`, after a warm-up, blocking on the result.

    Best-of rather than mean: the minimum is the measurement least polluted
    by other activity on the machine.
    """
    warm = fn()
    jax.block_until_ready(warm)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    return min(times)


def bench_single(
    graph: SignedGraph,
    params: RateParams | None = None,
    duration_ms: float = 200.0,
    stimulus_rows: np.ndarray | None = None,
    repeats: int = 3,
) -> BenchResult:
    """Throughput for one stimulus."""
    params = params or RateParams()
    sim = RateSimulator(graph, params)
    rows = (
        stimulus_rows
        if stimulus_rows is not None
        else np.arange(min(50, sim.n), dtype=np.int64)
    )
    stim = sim.make_stimulus(rows, 50.0)
    steps = int(round(duration_ms / params.dt_ms))

    seconds = _timed(lambda: sim.steady_state(stim, duration_ms), repeats)
    return BenchResult(
        label="single",
        n_neurons=sim.n,
        nnz=graph.nnz,
        batch=1,
        steps=steps,
        seconds=seconds,
        neuron_steps_per_s=sim.n * steps / seconds,
    )


def bench_batch(
    graph: SignedGraph,
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64),
    params: RateParams | None = None,
    duration_ms: float = 200.0,
    repeats: int = 3,
) -> list[BenchResult]:
    """Throughput as the number of simultaneous stimuli grows."""
    params = params or RateParams()
    sim = RateSimulator(graph, params)
    steps = int(round(duration_ms / params.dt_ms))
    rng = np.random.default_rng(0)
    results = []

    for b in batch_sizes:
        stimuli = np.zeros((b, sim.n), dtype=np.float32)
        for i in range(b):
            rows = rng.choice(sim.n, size=min(20, sim.n), replace=False)
            stimuli[i, rows] = 50.0

        seconds = _timed(lambda s=stimuli: sim.run_batch(s, duration_ms), repeats)
        results.append(
            BenchResult(
                label=f"batch={b}",
                n_neurons=sim.n,
                nnz=graph.nnz,
                batch=b,
                steps=steps,
                seconds=seconds,
                neuron_steps_per_s=b * sim.n * steps / seconds,
                extra={"seconds_per_stimulus": round(seconds / b, 5)},
            )
        )
    return results


def bench_gradient_vs_grafts(
    graph: SignedGraph,
    stimulus_type: str,
    readout_type: str,
    params: RateParams | None = None,
    duration_ms: float = 200.0,
    repeats: int = 3,
    ig_steps: int = 32,
) -> dict:
    """Cost of one gradient ranking against one simulation per cell type.

    The honest framing is a break-even, not a speedup. Integrated gradients
    runs `ig_steps` backward passes, each costing a few forward simulations,
    so its cost is roughly fixed. Brute force costs one simulation per cell
    type, so its cost grows linearly with the number of types.

    Below the break-even the gradient is *slower* — and on a 9-type synthetic
    graph it is far slower, which is the correct result and is reported as
    such. The gradient's advantage appears only at the scale the real
    connectome has, where around 11,700 types must be ranked.
    """
    from flyswap.attribution import integrated_gradients

    params = params or RateParams()
    sim = RateSimulator(graph, params)
    n_types = sim.n_types

    grad_seconds = _timed(
        lambda: integrated_gradients(
            sim, stimulus_type, readout_type, steps=ig_steps, duration_ms=duration_ms
        ).scores,
        repeats,
    )

    stim = sim.stimulus_for_type(stimulus_type, 50.0)
    one_sim = _timed(lambda: sim.steady_state(stim, duration_ms), repeats)
    brute_seconds = one_sim * n_types

    break_even = grad_seconds / one_sim if one_sim > 0 else float("inf")
    speedup = brute_seconds / grad_seconds if grad_seconds > 0 else None

    return {
        "n_types": n_types,
        "ig_steps": ig_steps,
        "gradient_seconds": round(grad_seconds, 4),
        "single_sim_seconds": round(one_sim, 5),
        "brute_force_seconds_estimated": round(brute_seconds, 4),
        "speedup": round(speedup, 3) if speedup else None,
        "break_even_n_types": round(break_even, 1),
        "gradient_is_faster": bool(speedup and speedup > 1.0),
        "note": (
            "Brute force is estimated as one simulation per cell type; the "
            "gradient is measured. Both exclude compilation. The gradient "
            "only pays off above break_even_n_types, so a speedup below 1 on "
            "a small graph is expected rather than a defect."
        ),
    }


def bench_scaling(
    sizes: tuple[int, ...] = (500, 2000, 8000),
    density: int = 40,
    params: RateParams | None = None,
    duration_ms: float = 100.0,
    seed: int = 0,
) -> list[BenchResult]:
    """Throughput against network size, on random graphs.

    Random graphs, not real ones, because the point is how cost scales with
    n and nnz. Absolute rates here are not biologically meaningful.
    """
    import pandas as pd
    from scipy import sparse

    from flyswap.data.schema import Connectome, DatasetSpec
    from flyswap.graph import build_graph

    params = params or RateParams()
    results = []
    rng = np.random.default_rng(seed)

    for n in sizes:
        pre = rng.integers(0, n, size=n * density)
        post = rng.integers(0, n, size=n * density)
        keep = pre != post
        pre, post = pre[keep], post[keep]
        counts = rng.integers(1, 100, size=len(pre))

        neurons = pd.DataFrame(
            {
                "body_id": np.arange(n),
                "cell_type": [f"T{i % 200}" for i in range(n)],
                "side": "M",
                "superclass": "intrinsic",
                "nt": rng.choice(
                    ["acetylcholine", "gaba", "glutamate"], size=n, p=[0.7, 0.2, 0.1]
                ),
                "nt_conf": 0.9,
                "fru_dsx": "none",
            }
        )
        edges = pd.DataFrame({"pre": pre, "post": post, "count": counts})
        edges = edges.groupby(["pre", "post"], as_index=False)["count"].sum()

        spec = DatasetSpec("random", "male", ("brain",), "synthetic", "n/a")
        conn = Connectome(spec=spec, neurons=neurons, edges=edges)
        graph = build_graph(conn)

        res = bench_single(graph, params, duration_ms)
        res.label = f"n={n}"
        results.append(res)

    _ = sparse  # imported for clarity about the matrix backend
    return results
