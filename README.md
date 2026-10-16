# flyswap

**Does a brain's wiring alone explain sex differences in behaviour?**

flyswap simulates the *Drosophila* connectome and answers that question with
counterfactuals: it grafts one sex's wiring for a single cell type into the
other sex's brain and measures which simulated circuits change.

```bash
pip install -e .
flyswap all --data synthetic --out results/   # full pipeline, no downloads
```

---

## Why this, and not another connectome simulator

Running a spiking model on the fly connectome is well-trodden: Shiu et al.
(2024) did it on the female brain, and open projects have since run the same
model on the male. Those projects are careful about their limits, and one
limit they state is the opening this project works in:

> There is one animal per dataset. There is no measure of individual
> variability, so sex differences cannot be separated from reconstruction
> differences.

flyswap takes that seriously and adds three things:

**1. A variability baseline.** Two female connectomes exist (FlyWire FAFB and
BANC). The female-female contrast measures everything that is *not* sex —
individual variation, reconstruction differences, annotation differences. A
male-female difference only counts when it exceeds that baseline.

**2. Grafts, not just silencing.** Silencing a cell type asks whether it is
*necessary*. Grafting the other sex's version of it asks whether that wiring
is *sufficient* — the question that identifies which types carry a difference
rather than merely participate in it.

**3. Gradient attribution, checked against the grafts.** The rate model is
differentiable in a per-cell-type gain, so one backward pass ranks every type.
The deliverable is not the ranking but its *agreement* with the grafts it is
meant to replace, reported as rank correlation and precision@k.

---

## What it does, concretely

| Stage | Command | What comes out |
| --- | --- | --- |
| Calibrate the one free parameter | `flyswap calibrate` | A global gain chosen by stated rules, on control pathways only |
| E1 — variability-controlled comparison | `flyswap dimorphism` | A dimorphism score per readout, with bootstrap intervals |
| E2 — cross-sex grafts | `flyswap graft` | Which cell types move a readout toward the other sex, with null grafts |
| E3 — gradient attribution | `flyswap attribute` | A ranking of all cell types, scored against the grafts |
| Benchmarks | `flyswap bench` | Throughput, batch scaling, gradient-vs-brute-force break-even |

Every command writes a JSON record carrying the full configuration — datasets,
sign rule, gain, seeds — so each number traces back to what produced it.

---

## Results on the synthetic fixtures

The repo ships a synthetic male/female pair with a dimorphism deliberately
wired in: `COURT_B`'s connections are strong in the male and weak in the
female, so stimulating `SENS` should drive `SONG` in the male only, while
`MOTOR` should behave the same in both sexes. The male is also generated 1.8×
denser, mimicking the reconstruction difference between the real datasets.

This is the project's own test: a method that cannot recover a difference
placed there on purpose will not recover a real one.

**E1 finds the planted readouts and nothing else.**

| Readout | Score | Effect ratio | Called dimorphic |
| --- | --- | --- | --- |
| SONG | 1.691 | 12.9× | **yes** |
| COURT_B | 1.055 | 14.8× | **yes** |
| MOTOR | 0.022 | 0.13× | no |
| PREMOT | −0.019 | −0.26× | no |
| RELAY | −0.001 | −0.50× | no |

**E2 identifies the carrier.** Grafting `COURT_B` from female into male moves
the SONG readout 99.9% of the way to the female value. Its circuit neighbours
move it partway, which is correct — a graft changes a type's inputs and
outputs together — and every isomorphic type moves it by less than 0.1% of
that.

**E3 agrees with E2.** Integrated gradients rank the same top three types the
grafts do, with Spearman ρ = 0.90.

**Controls hold.** Degree-preserving shuffles destroy the pathway; null grafts
between the two females barely move the readout.

---

## Benchmarks

Measured on CPU (JAX 0.11.2, `jax.default_backend() == "cpu"`), excluding
compilation, best of three runs. The repo records the environment alongside
every number.

| Batch size | Neuron-steps / s |
| --- | --- |
| 1 | 10.4 M |
| 4 | 37.1 M |
| 16 | 96.0 M |
| 64 | 130.8 M |

Batching gives a 12.5× throughput gain because the sparse matrix is shared
across the batch — the property that makes thousands of graft simulations
affordable. Throughput holds roughly flat from 500 to 8,000 neurons (≈7 M
neuron-steps/s), as expected when cost tracks edge count rather than neuron
count.

**One benchmark comes out negative, and is reported as such.** Integrated
gradients cost a fixed ~32 backward passes, while brute force costs one
simulation per cell type. On the 9-type synthetic graph the gradient is
**47× slower**. It only pays off above ~427 cell types. The real connectome
has about 11,700, which is where the approach is aimed — but on small graphs
the honest answer is to run the grafts directly.

---

## Working with the real connectomes

No connectome data ship with this repo. Download each from its official
source, keep the local copies, and cite them:

| Dataset | Sex | Coverage | Source |
| --- | --- | --- | --- |
| MaleCNS v1.0 | male | brain + nerve cord | [male-cns.janelia.org](https://male-cns.janelia.org/download/) |
| FlyWire FAFB v783 | female | brain only | [Codex](https://codex.flywire.ai/api/download?dataset=fafb) |
| BANC v888 | female | brain + nerve cord | [BANC Codex](https://codex.flywire.ai/banc) |

```bash
flyswap dimorphism --data real --data-dir /path/to/datasets --out results/
```

Loaders tolerate the column renames that happen between releases, and fail
with the dataset's download URL attached rather than a `KeyError` three merges
deep.

---

## Design decisions worth knowing

**Weights are input fractions, not synapse counts.** A male neuron carries
roughly 1.8× the synapses of its female counterpart — a reconstruction
difference, not a biological one. Comparing raw counts would measure the
microscope. Dividing each neuron's incoming weights by its total incoming
synapses gives a dimensionless share that cancels a uniform density
difference exactly. A test asserts this invariance.

**Grafts happen at type level.** There is no neuron-to-neuron correspondence
between two connectomes, and no fact of the matter about which male neuron
matches which female one. What is matched is the *type*, so a graft rewrites
type-pair connection strengths and leaves the host's own neurons and their
counts untouched.

**Significance needs two bars, not one.** The rate model is deterministic, so
bootstrap intervals over stimulus jitter come out extremely narrow and a
biologically meaningless difference will still exclude zero. A readout counts
as dimorphic only if its interval excludes zero **and** the sex effect is at
least as large as the female-female baseline.

**Calibration happens before the comparison, on control pathways.** The model
has one free parameter. It is chosen by stated criteria (no saturation,
signal propagates, activity decays) on a pathway that is not under study —
never tuned per experiment until the hypothesis comes out right.

---

## What this cannot tell you

- **Weights are estimates.** Synapse count is a proxy for strength; sign comes
  from a transmitter classifier with a real error rate. Run
  `--sign-rule glut_excitatory` or perturb signs to see which conclusions
  survive.
- **The model is wiring-only.** No gap junctions, no neuromodulation, no
  plasticity, no cell-specific electrical properties.
- **Three animals is not a population.** Variability estimates from two
  females are rough, and that is the ceiling on what any sex claim here can
  mean.
- **Outputs are neuron readouts in an isolated brain, not behaviour.**

Findings from this tool are hypotheses for experimentalists to test, not
measurements of what a fly does.

---

## Development

```bash
pip install -e ".[dev]"
pytest                 # 78 tests, synthetic data only, no downloads
ruff check flyswap
```

Tests run on synthetic fixtures with known ground truth, so CI needs no
gigabyte downloads. The suite that matters is `tests/test_ground_truth.py`:
it asserts that the pipeline recovers the planted dimorphism and stays quiet
where nothing was planted.

## Citation

If you use this code, cite the data and the model it builds on:

- Berg, S. et al. Sexual dimorphism in the complete *Drosophila* male central
  nervous system connectome. *Cell* 189, 5504–5526 (2026).
  doi:10.1016/j.cell.2026.08.015
- Shiu, P. K. et al. A *Drosophila* computational brain model reveals
  sensorimotor processing. *Nature* 634, 210–219 (2024).
- FlyWire: Dorkenwald et al. (2024); Schlegel et al. (2024); Matsliah et al.
  (2024).

Code: MIT. Connectome data are not part of this repository and carry their own
licences (CC-BY 4.0).
