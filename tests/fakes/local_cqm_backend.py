"""``FakeLocalCQMBackend`` — a local, exhaustive constraint-model backend (3a §26.1).

The first backend that declares ``supported_model_types=["cqm"]``, so the
first thing to walk the service's CQM path end to end (3a step 7): the
service must pick ``CQMCompiler`` from this declaration alone, compile
with ``hard_penalty=None``, make exactly one attempt and still judge every
returned sample with the independent validator.

It wraps ``dimod.ExactCQMSolver().sample_cqm``, which enumerates *every*
assignment, hence ``exhaustive=True``: a run with no feasible sample proves
infeasibility (§16.2 step 15). Every sample is returned in the sampler's
own order — nothing is filtered on ``is_feasible`` and nothing is sorted —
because the project never trusts a sampler's feasibility verdict
(overview principle 2, §16.5). The verdict is only *counted* into
``metadata.sampler_reported_feasible`` (§22), and ``override_sampleset``
lets a test feed a tampered sampleset (all rows flagged feasible, some of
them violating a hard constraint) to prove that the count is reported as
given while the offending rows never reach ``solutions``.

Test-only: never shipped, never production.
"""

import dimod
import numpy as np

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    AvailabilityStatus,
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_arrays,
)
from annealbridge.solvers.metadata import sanitize_sampleset_info

FAKE_LOCAL_CQM_NAME = "fake_local_cqm"

_CAPABILITIES = SolverCapabilities(
    name=FAKE_LOCAL_CQM_NAME,
    remote=False,
    heuristic=False,
    exhaustive=True,
    supports_seed=False,
    supports_num_reads=False,
    supports_time_limit=False,
    supported_model_types=["cqm"],
    returns_multiple_samples=True,
    description=(
        "Test-only local constraint-model backend enumerating every "
        "assignment with dimod.ExactCQMSolver (Phase 3a spec §26.1); "
        "never production."
    ),
)


def all_feasible_sampleset(
    rows: list[dict[str, int]], energies: list[float]
) -> dimod.SampleSet:
    """A sampleset whose every row is flagged ``is_feasible=True``.

    Built without a CQM on purpose: dimod would compute the real verdict
    from the model, whereas a test wants to *lie* about it to prove the
    service ignores the flag. Variable order follows the first row.
    """
    return dimod.SampleSet.from_samples(
        rows,
        "BINARY",
        energy=energies,
        is_feasible=[True] * len(rows),
        sort_labels=False,
    )


class FakeLocalCQMBackend:
    """Implements ``SolverBackend`` over ``dimod.ExactCQMSolver``.

    Records the last compiled problem and preferences it received so a
    test can inspect the model the service actually built (hard
    constraints native, no penalty, no slack) without reaching into the
    service.
    """

    def __init__(self, override_sampleset: dimod.SampleSet | None = None) -> None:
        self._override = override_sampleset
        self.solve_calls = 0
        self.last_compiled: CompiledProblem | None = None
        self.last_preferences: SolverPreferences | None = None

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Local backend, always available. No network I/O."""
        return AvailabilityStatus(category="available")

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self.capabilities.exhaustive

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float | None:
        """No time limit support: always None."""
        return None

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        self.solve_calls += 1
        self.last_compiled = compiled_problem
        self.last_preferences = preferences
        model = compiled_problem.model
        if not isinstance(model, dimod.ConstrainedQuadraticModel):
            raise SolverExecutionError(
                f"{self.name} accepts a ConstrainedQuadraticModel, "
                f"got {type(model).__name__}"
            )
        if self._override is not None:
            sampleset = self._override
        else:
            sampleset = dimod.ExactCQMSolver().sample_cqm(model)
        return self._to_raw(sampleset)

    def _to_raw(self, sampleset: dimod.SampleSet) -> RawSolverResult:
        # Every read, sampler order, no filtering (§16.5): feasibility is
        # decided downstream against the original problem, never here.
        variables, samples, energies = sampleset_to_arrays(sampleset)
        reported = int(np.count_nonzero(sampleset.record.is_feasible))
        metadata = sanitize_sampleset_info(
            dict(sampleset.info), self.name, remote=False
        ).model_copy(update={"sampler_reported_feasible": reported})
        return RawSolverResult(
            variables=variables,
            samples=samples,
            energies=energies,
            backend=self.name,
            metadata=metadata,
        )
