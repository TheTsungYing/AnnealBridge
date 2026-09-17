"""Simulated bifurcation solver backend (local, numpy; optional torch GPU).

Simulated bifurcation (SB) is Toshiba's classical heuristic for Ising
problems: every spin is a continuous oscillator ``x_i`` with momentum
``y_i``, driven by the couplings and by a "pump" that grows from 0 to 1 over
the run; as the pump passes the bifurcation point each oscillator falls into
``+1`` or ``-1`` and the signs are the answer. Two variants are implemented:

* ``ballistic`` (bSB) -- the force uses the continuous positions, from
  H. Goto, K. Endo, M. Suzuki, Y. Kanao, Y. Hamakawa, R. Hidaka, M. Yamasaki,
  K. Tatsumura, "High-performance combinatorial optimization based on
  classical mechanics", Science Advances 7, eabe7953 (2021);
* ``discrete`` (dSB) -- the force uses the *signs* of the positions, from the
  same paper, and the default here: on dense +-1 SK instances at 200 and
  1000 variables it reached the same best energy as simulated annealing and
  tabu search in a third to an eighth of their time, where bSB stayed a few
  units short.

Unlike simulated annealing and tabu search, one SB step updates *every*
spin of *every* read at once with one matrix product ``J @ x`` (``J`` is
``N x N``, ``x`` is ``N x reads``), which is why the backend is a matrix
routine and not a loop over reads, why it scales with dense problems, and why
it has no thread fan-out of its own: the BLAS behind ``numpy.matmul`` already
uses every core (``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` bound it).
Measured on the knapsack examples the answer is bit-identical at 1 and at 6
BLAS threads.

The weak spot, measured on the shipped examples: small penalty-dominated
QUBOs (a hard-constraint penalty of the order 10^3..10^5 next to objective
coefficients of the order 1) are found in only 1 to 5 % of the reads at
either mode, against 7 % for simulated annealing; neither more steps (up to
20 000) nor a steepest-descent polish improved that, so no polish is added.
Which variant wins is problem-dependent: dSB on the dense instances above,
bSB on the integer knapsack example (where dSB stalls at a local minimum
three units above the optimum). Hence ``mode`` is a solver option.
"""

import importlib
import importlib.util
import logging
from typing import Any, Literal

import dimod
import numpy as np

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    SolverExecutionMetadata,
    SolverPreferences,
)
from annealbridge.solvers.base import (
    AvailabilityStatus,
    BackendAliases,
    CompiledVariableLimit,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    log_solved,
)

logger = logging.getLogger(__name__)

Device = Literal["cpu", "cuda"]
Mode = Literal["discrete", "ballistic"]

# Seeds are fed to ``numpy.random.SeedSequence``, which takes any
# non-negative integer; the range is capped at the same ``0 <= seed < 2**32``
# the tabu backend declares so that the two local backends added after the
# annealer agree, and so the capabilities view has one rule to report. The
# torch path does not draw random numbers at all (see ``_initial_states``),
# so the rule is the same on both devices.
_SEED_LIMIT = 2**32

# Reads (SB "agents") are simulated in batches of this many columns of the
# state matrix, one batch after another, to bound the state memory no
# matter how many reads are asked: per batch, the positions, momenta, the
# force product and the wall mask, ``N x AGENTS_PER_BATCH`` each in single
# precision (a few hundred MB at 10 000 variables), plus the float64 draw
# the initial positions come from. It is *not* a thread fan-out (see the
# module docstring). The batch layout and every batch's initial state are a
# function of ``(num_reads, seed)`` alone, so the constant is part of the
# reproducibility contract: changing it may change results, exactly like the
# shard size of the other local backends.
AGENTS_PER_BATCH = 1024

# The dynamics' fixed constants, the paper's choices; none is a user
# preference, because tuning them is not a dial an agent can reason about.
#   a0 -- the pump's target (the paper's ``a_0``): the pump rises linearly
#        from 0 towards it over the run, ``a0 * k / steps`` at step ``k``;
#   dt -- the time step of the symplectic Euler integration (``Delta_t``);
#   c0 = _C0_FACTOR / (rms(J) * sqrt(N)) -- the coupling gain, the paper's
#        ``c_0 = 0.5 / (sigma sqrt(N))`` with ``sigma`` taken as the RMS of
#        the off-diagonal couplings (see ``_dense_ising`` for why RMS and
#        not the paper's standard deviation). Measured against the
#        alternative of scaling by per-row norms including the field: no
#        better anywhere, worse on the dense instances, so the rule stands;
#   initial positions -- uniform in ``(-0.1, 0.1)``, momenta zero.
_A0 = 1.0
_DT = 1.0
_C0_FACTOR = 0.5
_INIT_AMPLITUDE = 0.1

# Single precision throughout the dynamics: it halves the dense coupling
# matrix (400 MB at 10 000 variables), doubles the matrix-product rate, and
# is what the published implementations use. Energies are *not* taken from
# the dynamics: every returned sample is re-scored by the BQM itself in
# float64 (``dimod``'s ``energies``), so the precision of the search never
# touches the precision of the answer.
_DTYPE = np.float32

_CAPABILITIES = SolverCapabilities(
    name="simulated_bifurcation",
    remote=False,
    heuristic=True,
    exhaustive=False,
    supports_seed=True,
    supports_num_reads=True,
    # ``num_sweeps`` is the number of integration steps, the paper's ``N_step``.
    supports_num_sweeps=True,
    # The step count bounds the work; there is no wall clock anywhere in
    # the backend, so no time limit is declared.
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    seed_min=0,
    seed_max=_SEED_LIMIT - 1,
    # Both measured (module docstring, docs/backends.md): the fastest of the
    # local heuristics to the same energy on a dense 1000-variable SK, and
    # the lowest hit rate per read on the shipped hard-constrained examples,
    # where the penalty dominates the objective. The weakness was measured
    # on small examples but is declared for the shape at any size: a large
    # constrained model has not been measured, so recommend stays
    # conservative there. It matches these against the problem's shape and
    # never names this backend.
    strong_on_large_dense=True,
    weak_on_penalty_dominated=True,
    description=(
        "Local simulated bifurcation (Toshiba's bSB/dSB, in numpy; optional "
        "CUDA via PyTorch), a dense-matrix heuristic that updates every "
        "variable of every read at once; fastest to the same energy on large "
        "dense unconstrained QUBOs, weak where hard constraints compile to "
        "penalties; honours num_reads (parallel "
        "trajectories), num_sweeps (integration steps) and seed."
    ),
    # Reads and steps bound the CPU time (or GPU time) one request can hold
    # a concurrency slot for, under the same local keys as the annealer.
    parameter_limits=[
        ParameterLimit(
            preference="num_reads", limit="local_reads", error_code="LOCAL_READS_LIMIT"
        ),
        ParameterLimit(
            preference="num_sweeps", limit="sweeps", error_code="SWEEPS_LIMIT"
        ),
    ],
)

_FAILED = "Simulated bifurcation solver failed"


class SimulatedBifurcationBackend(BackendAliases):
    """Samples the compiled BQM with simulated bifurcation.

    All reads are kept (never just the best) so every candidate reaches the
    solution validator. The seed goes to a private ``SeedSequence`` only;
    global random state is never touched (spec §37).

    ``device`` names where the dynamics run: ``"cpu"`` (numpy, part of the
    core install) or ``"cuda"`` (PyTorch from the ``gpu`` extra, imported
    lazily and only on that setting). A device that is not usable makes
    :meth:`is_available` say so; the backend never falls back to the CPU on
    its own, because a silent fallback would hide a misconfigured server
    behind a result that merely arrived more slowly.

    ``max_variables`` caps the compiled problem: the dynamics need the
    couplings as a dense ``N x N`` single-precision matrix (``4 N^2`` bytes,
    400 MB at the default 10 000) on top of the compiled ``dimod`` model
    itself, which on a dense problem holds every interaction and is the
    larger of the two. The cap is declared as ``compiled_variable_limit``,
    so the service refuses a problem above it before compiling, with
    ``SB_VARIABLE_LIMIT`` under ``resource_limit_exceeded``, and
    ``recommend`` lists the same refusal; nothing is allocated or clamped.

    The result is a function of ``(problem, num_reads, num_sweeps, seed,
    mode, device)``: the initial states are derived from the seed by numpy
    on every device, the batch layout is fixed by ``AGENTS_PER_BATCH``, and
    nothing in the run depends on a clock. On one machine with one BLAS
    build the same request returns bit-identical samples (measured at 1 and 6
    BLAS threads); a different CPU instruction set or a different BLAS may
    round the matrix products differently and arrive at different samples,
    which is a weaker promise than the count-bounded tabu search makes.
    """

    def __init__(
        self,
        *,
        device: Device = "cpu",
        max_variables: int = 10_000,
    ) -> None:
        """``device``: ``"cpu"`` or ``"cuda"``. ``max_variables``: the dense
        matrix cap. Neither is clamped: a value outside its range is a caller
        bug."""
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
        if max_variables < 1:
            raise ValueError(f"max_variables must be >= 1, got {max_variables}")
        self._device: Device = device
        self._max_variables = max_variables
        # The cap is this instance's, so the declaration carries it: the
        # service and ``recommend`` refuse an oversized problem from here,
        # before compiling, and the capabilities view reports it.
        self._capabilities = _CAPABILITIES.model_copy(
            update={
                "compiled_variable_limit": CompiledVariableLimit(
                    maximum=max_variables, error_code="SB_VARIABLE_LIMIT"
                )
            }
        )

    @property
    def capabilities(self) -> SolverCapabilities:
        return self._capabilities

    @property
    def device(self) -> Device:
        """Where the dynamics run."""
        return self._device

    @property
    def max_variables(self) -> int:
        """Largest compiled problem accepted (dense coupling matrix cap)."""
        return self._max_variables

    def is_available(self) -> AvailabilityStatus:
        """Available on ``"cpu"`` unconditionally; on ``"cuda"`` only when
        PyTorch is installed *and* sees a CUDA device. No network I/O, no
        caching: a driver or an install can change between calls."""
        if self._device == "cpu":
            return AvailabilityStatus(category="available")
        if importlib.util.find_spec("torch") is None:
            return AvailabilityStatus(
                category="not_installed",
                detail="PyTorch not installed (annealbridge[gpu] extra)",
            )
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return AvailabilityStatus(
                category="unavailable",
                detail="no CUDA device visible to PyTorch",
            )
        return AvailabilityStatus(category="available")

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Run the dynamics, returning every read as a sample."""
        num_variables = compiled_problem.num_variables
        if num_variables > self._max_variables:
            # The service refuses this before compiling, from the declared
            # ``compiled_variable_limit``; this is the last line for a direct
            # caller, refused before any allocation and worded the same.
            raise SolverExecutionError(
                f"Compiled problem has {num_variables} variables (including "
                f"internal), exceeding the {self.name} backend limit of "
                f"{self._max_variables}",
                code="SB_VARIABLE_LIMIT",
                status="resource_limit_exceeded",
            )
        seed = preferences.seed
        if seed is not None and not 0 <= seed < _SEED_LIMIT:
            # Declared as ``seed_min`` / ``seed_max`` so validation refuses it
            # first; this is the last line for a direct caller.
            raise SolverExecutionError(
                f"{_FAILED}: seed must be between 0 and 2**32 - 1, got {seed}"
            )
        options = preferences.simulated_bifurcation
        mode: Mode = options.mode if options is not None else "discrete"

        try:
            variables, field, couplings, gain = _dense_ising(compiled_problem.model)
            samples, batches = self._sample(
                couplings,
                field,
                gain,
                num_reads=preferences.num_reads,
                steps=preferences.num_sweeps,
                seed=seed,
                discrete=(mode == "discrete"),
            )
            # Re-scored by the original binary model in float64, offset
            # included: the dynamics' own energy is never reported.
            energies = np.asarray(
                compiled_problem.model.energies((samples, variables)),
                dtype=np.float64,
            )
        except Exception as exc:
            raise SolverExecutionError(f"{_FAILED}: {exc}") from exc

        result = RawSolverResult(
            variables=variables,
            samples=samples,
            energies=energies,
            backend=self.name,
            # A local run leaves no vendor facts behind: no solver id, no
            # timing, no quota. Reported are the backend, that it ran here,
            # and the read count asked of it -- the batch layout is an
            # implementation detail and is not part of the output.
            metadata=SolverExecutionMetadata(
                backend=self.name,
                remote=False,
                num_reads_requested=preferences.num_reads,
            ),
        )
        log_solved(
            logger,
            self.name,
            compiled_problem.original_problem.name,
            num_variables,
            len(result.samples),
            num_reads=preferences.num_reads,
            num_sweeps=preferences.num_sweeps,
            seed=seed,
            mode=mode,
            device=self._device,
            batches=batches,
        )
        return result

    def _sample(
        self,
        couplings: np.ndarray,
        field: np.ndarray,
        gain: float,
        *,
        num_reads: int,
        steps: int,
        seed: int | None,
        discrete: bool,
    ) -> tuple[np.ndarray, int]:
        """Run every batch of reads in order; return the binary samples
        (``num_reads x N``, int8) and the number of batches run."""
        n = couplings.shape[0]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        sizes = _batch_sizes(num_reads)
        xp, to_device, to_host = self._runtime()
        j_dev = to_device(couplings)
        h_dev = to_device(field.reshape(n, 1))
        # Allocated once: the output is ``num_reads x N`` bytes, and filling
        # it batch by batch avoids a second copy at the end.
        samples = np.zeros((max(num_reads, 0), n), dtype=np.int8)
        start = 0
        for size in sizes:
            x0, y0 = _initial_states(rng, n, size)
            x = _run(
                xp, j_dev, h_dev, to_device(x0), to_device(y0), steps, gain, discrete
            )
            if not bool(xp.isfinite(x).all()):
                # Cannot happen for a finite model after the rescaling
                # above; if it ever does, ``sign(nan)`` would otherwise turn
                # into a silent all-zero sample (2026-09-17 review).
                raise ArithmeticError(
                    "the dynamics produced a non-finite state; the model's "
                    "biases are outside the range single precision can hold"
                )
            # A spin is +1 exactly when its position is non-negative, and a
            # +1 spin is binary 1: ``(s + 1) / 2`` in one comparison.
            samples[start : start + size] = to_host(x >= 0).T
            start += size
        return samples, len(sizes)

    def _runtime(self) -> tuple[Any, Any, Any]:
        """The array module the dynamics run on, plus the two edge
        conversions. On ``"cpu"`` all three are numpy identities."""
        if self._device == "cpu":
            return np, (lambda array: array), (lambda array: np.asarray(array))
        torch = importlib.import_module("torch")
        device = torch.device("cuda")

        def to_device(array: np.ndarray) -> Any:
            return torch.from_numpy(np.ascontiguousarray(array)).to(device)

        def to_host(tensor: Any) -> np.ndarray:
            return tensor.cpu().numpy()

        return torch, to_device, to_host


def _dense_ising(
    model: dimod.BinaryQuadraticModel,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """The compiled binary model as the Ising form the dynamics take.

    Returns ``(variables, h, J, c0)``: ``h`` the field (``N``) and ``J`` the
    dense symmetric coupling matrix (``N x N``), both ``float32``, such that
    the spin energy is ``-1/2 s^T J s + h . s`` up to the model's offset --
    i.e. ``J[i, j] = -q_ij`` for the spin bias ``q_ij`` -- so that the
    force ``J s - h`` is minus the energy gradient; and ``c0`` the coupling
    gain that goes with them.

    The binary-to-spin change is done here on the model's own bias vectors
    (``x = (1 + s) / 2``: a linear bias ``a`` becomes ``a / 2`` of field, a
    quadratic bias ``b`` becomes ``b / 4`` of coupling plus ``b / 4`` of
    field on each end) rather than through ``change_vartype``, which would
    copy the whole model in float64 first. What is held in float64 is one
    vector per bias list -- about 24 bytes per interaction, transient -- and
    the compiled ``dimod`` model itself remains the larger cost on a dense
    problem.

    **Both arrays are uniformly rescaled by their largest magnitude before
    the single-precision cast**, and ``c0`` absorbs the factor: the paper's
    ``c_0 = 0.5 / (sigma sqrt(N))`` is scale-invariant (the force
    ``c0 (J s - h)`` is unchanged), so the dynamics are the same, but no
    finite float64 bias can overflow float32 to ``inf`` on the way in
    (2026-09-17 review). ``sigma`` is the RMS of the off-diagonal couplings.
    It equals the paper's standard deviation for the zero-mean couplings the
    paper benchmarks, and it is the force magnitude the dynamics actually
    see when the mean is not zero -- a penalty QUBO's couplings all share a
    sign -- so the RMS is the deliberate choice. A model without couplings
    is scaled by the RMS of its field, its only scale; a model with neither
    has a constant energy, every sample is optimal, and ``c0 = 0`` leaves
    the trajectories at their random start: no error, nothing to optimise.
    """
    variables = list(model.variables)
    n = len(variables)
    vectors = model.to_numpy_vectors(variable_order=variables)
    linear = np.asarray(vectors.linear_biases, dtype=np.float64)
    rows, cols, quadratic = vectors.quadratic
    quadratic = np.asarray(quadratic, dtype=np.float64)
    field = linear / 2.0
    if quadratic.size:
        quarter = quadratic / 4.0
        field += np.bincount(rows, weights=quarter, minlength=n)
        field += np.bincount(cols, weights=quarter, minlength=n)
        pair_couplings = -quarter
    else:
        pair_couplings = quadratic

    peak = max(
        float(np.abs(field).max()) if n else 0.0,
        float(np.abs(pair_couplings).max()) if pair_couplings.size else 0.0,
    )
    if peak > 0.0:
        field /= peak
        pair_couplings = pair_couplings / peak
    off_diagonal = n * (n - 1)
    # Each pair appears twice in the symmetric matrix.
    scale = (
        float(np.sqrt(2.0 * np.sum(np.square(pair_couplings)) / off_diagonal))
        if off_diagonal
        else 0.0
    )
    if scale == 0.0 and n:
        scale = float(np.sqrt(np.mean(np.square(field))))
    # A plain float: a numpy float64 scalar would promote the float32 state
    # arrays to float64 on every step (NEP 50), and would not be a torch
    # scalar at all.
    gain = float(_C0_FACTOR / (scale * np.sqrt(n))) if scale > 0.0 else 0.0

    couplings = np.zeros((n, n), dtype=_DTYPE)
    couplings[rows, cols] = pair_couplings.astype(_DTYPE)
    couplings[cols, rows] = couplings[rows, cols]
    return variables, field.astype(_DTYPE), couplings, gain


def _batch_sizes(num_reads: int) -> list[int]:
    """Batches of ``AGENTS_PER_BATCH`` reads, the last one partial."""
    if num_reads <= 0:
        return []
    full, rest = divmod(num_reads, AGENTS_PER_BATCH)
    return [AGENTS_PER_BATCH] * full + ([rest] if rest else [])


def _initial_states(
    rng: np.random.Generator, n: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initial positions and momenta for ``size`` reads, ``N x size``.

    Drawn read-major (``size x N``) and transposed, so that drawing the
    batches one after another consumes the generator exactly as one draw of
    all reads would: the initial state of read ``k`` depends on the seed and
    on ``k``, never on the batch it lands in.
    """
    positions = rng.uniform(-_INIT_AMPLITUDE, _INIT_AMPLITUDE, size=(size, n))
    x = np.array(positions.T, dtype=_DTYPE, order="C")
    return x, np.zeros_like(x)


def _run(
    xp: Any,
    couplings: Any,
    field: Any,
    x: Any,
    y: Any,
    steps: int,
    c0: float,
    discrete: bool,
) -> Any:
    """The SB dynamics, in place on ``x`` / ``y`` (``N x reads``).

    ``xp`` is the array module -- ``numpy`` or ``torch`` -- and everything
    used here (``sign``, ``abs``, ``matmul``, arithmetic, boolean indexing)
    has the same name and meaning in both, which is what lets one routine
    serve both devices. Per step, with pump ``a = a0 * k / steps``::

        z  = sign(x) if discrete else x
        y += (-(a0 - a) * x + c0 * (J @ z - h)) * dt
        x += a0 * y * dt
        x clipped to [-1, 1]; where it hit the wall, y is reset to 0

    (the symplectic Euler scheme of Goto et al. 2021, their eqs. 3-4 with
    the wall condition). Returns ``x``.
    """
    for step in range(steps):
        pump = _A0 * step / steps
        z = xp.sign(x) if discrete else x
        y += ((-(_A0 - pump)) * x + c0 * (xp.matmul(couplings, z) - field)) * _DT
        x += (_A0 * _DT) * y
        outside = xp.abs(x) > 1
        x[outside] = xp.sign(x[outside])
        y[outside] = 0
    return x
