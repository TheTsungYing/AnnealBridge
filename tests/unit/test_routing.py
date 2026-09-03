"""``orchestration/routing.py`` — backend recommendation (3a spec §23, §26.1).

The ranking is advisory, deterministic and free: it never solves, never
calls ``resolve_time_limit()``, never touches the network and never
rewrites ``problem.solver.backend``. Remote backends are made "available"
here by replacing ``is_available`` on the registry instances (no D-Wave
package or config is involved), and the fake declared backend of
``tests/fakes`` stands in for an always-available remote.
"""

import ast
import json
from pathlib import Path

import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import AvailabilityStatus, OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.orchestration.routing import REASON_DESCRIPTIONS, recommend
from annealbridge.solvers import SolverRegistry
from tests.conftest import EXAMPLES_DIR
from tests.fakes.declared_backend import FAKE_DECLARED_NAME, FakeDeclaredBackend

ALL_REASON_CODES = {
    "R_UNUSABLE",
    "R_EXACT_FITS",
    "R_EXACT_NEAR_LIMIT",
    "R_EXACT_OVER_LIMIT",
    "R_LOCAL_HEURISTIC",
    "R_NATIVE_CONSTRAINTS",
    "R_REMOTE",
    "R_SINGLE_SAMPLE",
    "R_DENSE_FOR_QPU",
}


def compilers() -> dict:
    return {c.model_type: c for c in (BQMCompiler(), CQMCompiler())}


def knapsack(**solver) -> OptimizationProblem:
    payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    payload["solver"] = {**payload.get("solver", {}), **solver}
    return OptimizationProblem.model_validate(payload)


def make_problem(
    num_variables: int,
    *,
    constraint_width: int | None = None,
    rhs: int = 1,
    **solver,
) -> OptimizationProblem:
    """minimize x1 s.t. (optionally) x1 + ... + x_width <= rhs."""
    names = [f"x{index}" for index in range(1, num_variables + 1)]
    constraints = []
    if constraint_width:
        constraints.append(
            {
                "id": "cap",
                "type": "hard",
                "terms": [
                    {"variable": name, "coefficient": 1}
                    for name in names[:constraint_width]
                ],
                "operator": "<=",
                "rhs": rhs,
            }
        )
    return OptimizationProblem.model_validate(
        {
            "name": "routing",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "minimize",
                "linear_terms": [{"variable": names[0], "coefficient": 1}],
            },
            "constraints": constraints,
            "solver": solver,
        }
    )


def by_name(result, name: str):
    return next(entry for entry in result.recommendations if entry.backend == name)


def remote_names(registry: SolverRegistry) -> list[str]:
    return [n for n in registry.names() if registry.get(n).capabilities.remote]


def make_remotes_available(monkeypatch, registry: SolverRegistry) -> None:
    """Every remote backend reports available; no network, no config file."""
    for name in remote_names(registry):
        monkeypatch.setattr(
            registry.get(name),
            "is_available",
            lambda: AvailabilityStatus(category="available"),
        )


def forbid_solving(monkeypatch, registry: SolverRegistry) -> None:
    """Any call to ``solve`` or ``resolve_time_limit`` fails the test."""

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("routing must never solve or resolve a time limit")

    for name in registry.names():
        backend = registry.get(name)
        monkeypatch.setattr(backend, "solve", _forbidden)
        monkeypatch.setattr(backend, "resolve_time_limit", _forbidden)


@pytest.fixture
def registry() -> SolverRegistry:
    return SolverRegistry.default()


class TestDefaultPolicy:
    def test_knapsack_order_and_usability(self, registry):
        result = recommend(knapsack(), registry, ExecutionPolicy(), compilers())

        assert result.valid is True
        assert result.errors == []
        assert [e.backend for e in result.recommendations] == [
            "exact",
            "simulated_annealing",
            "leap_hybrid_cqm",
            "dwave_qpu",
            "leap_hybrid_bqm",
        ]
        assert [e.rank for e in result.recommendations] == [1, 2, 3, 4, 5]
        assert by_name(result, "exact").usable is True
        assert by_name(result, "simulated_annealing").usable is True
        for name in remote_names(registry):
            entry = by_name(result, name)
            assert entry.usable is False
            assert "REMOTE_DISABLED" in [b.code for b in entry.blocking]
            assert entry.reasons[0] == "R_UNUSABLE"

    def test_reasons_follow_the_spec_example(self, registry):
        result = recommend(knapsack(), registry, ExecutionPolicy(), compilers())
        assert by_name(result, "exact").reasons == ["R_EXACT_FITS"]
        assert by_name(result, "simulated_annealing").reasons == ["R_LOCAL_HEURISTIC"]
        assert by_name(result, "leap_hybrid_cqm").reasons == [
            "R_UNUSABLE",
            "R_NATIVE_CONSTRAINTS",
        ]
        assert by_name(result, "dwave_qpu").reasons == ["R_UNUSABLE", "R_REMOTE"]
        assert by_name(result, "leap_hybrid_bqm").reasons == [
            "R_UNUSABLE",
            "R_REMOTE",
            "R_SINGLE_SAMPLE",
        ]

    def test_model_type_and_estimate_follow_the_compiler_path(self, registry):
        result = recommend(knapsack(), registry, ExecutionPolicy(), compilers())
        exact = by_name(result, "exact")
        cqm = by_name(result, "leap_hybrid_cqm")
        assert exact.model_type == "bqm"
        assert cqm.model_type == "cqm"
        # BQM path counts slack bits (capacity 10 -> 4 bits); CQM has none.
        assert exact.estimated_compiled_variables == 8
        assert cqm.estimated_compiled_variables == 4

    def test_advisory_sentence_is_fixed(self, registry):
        result = recommend(knapsack(), registry, ExecutionPolicy(), compilers())
        assert result.advisory.startswith("Advisory only: solve_optimization always uses")

    def test_never_rewrites_the_users_backend_choice(self, registry):
        problem = knapsack(backend="leap_hybrid_cqm")
        recommend(problem, registry, ExecutionPolicy(), compilers())
        assert problem.solver.backend == "leap_hybrid_cqm"


class TestRemoteAllowed:
    @pytest.fixture
    def policy(self) -> ExecutionPolicy:
        return ExecutionPolicy(allow_remote=True)

    def test_cqm_before_bqm_hybrid_when_hard_constraints_exist(
        self, monkeypatch, registry, policy
    ):
        make_remotes_available(monkeypatch, registry)
        result = recommend(knapsack(), registry, policy, compilers())

        order = [e.backend for e in result.recommendations]
        assert all(e.usable for e in result.recommendations)
        assert order.index("leap_hybrid_cqm") < order.index("leap_hybrid_bqm")
        assert order[:2] == ["exact", "simulated_annealing"]
        assert by_name(result, "leap_hybrid_cqm").reasons == ["R_NATIVE_CONSTRAINTS"]

    def test_bqm_hybrid_before_cqm_without_constraints(
        self, monkeypatch, registry, policy
    ):
        make_remotes_available(monkeypatch, registry)
        result = recommend(make_problem(3), registry, policy, compilers())

        order = [e.backend for e in result.recommendations]
        assert order.index("leap_hybrid_bqm") < order.index("leap_hybrid_cqm")
        assert by_name(result, "leap_hybrid_cqm").reasons == ["R_REMOTE"]
        # Same tier (4): registry order decides.
        assert order[2:] == ["dwave_qpu", "leap_hybrid_bqm", "leap_hybrid_cqm"]

    def test_fake_declared_remote_is_listed_and_usable(self, monkeypatch, policy):
        defaults = SolverRegistry.default()
        backends = {name: defaults.get(name) for name in defaults.names()}
        backends[FAKE_DECLARED_NAME] = FakeDeclaredBackend()
        registry = SolverRegistry(backends)
        make_remotes_available(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True, limits={"iterations": 1000})

        result = recommend(knapsack(), registry, policy, compilers())

        fake = by_name(result, FAKE_DECLARED_NAME)
        assert fake.usable is True
        assert fake.blocking == []
        assert fake.reasons == ["R_REMOTE"]


class TestDenseForQpu:
    def test_qpu_ranks_last_among_usable_backends(self, monkeypatch, registry):
        # 35 variables in one constraint (> 30) triggers DENSE_FOR_QPU while
        # the exhaustive backend still fits under a raised ceiling.
        make_remotes_available(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True, exact_max_variables=200)
        problem = make_problem(35, constraint_width=35, rhs=20)

        result = recommend(problem, registry, policy, compilers())

        assert all(e.usable for e in result.recommendations)
        assert result.recommendations[-1].backend == "dwave_qpu"
        qpu = by_name(result, "dwave_qpu")
        assert qpu.reasons == ["R_DENSE_FOR_QPU", "R_REMOTE"]
        assert "DENSE_FOR_QPU" in [w.code for w in qpu.warnings]

    def test_qpu_ranks_last_under_the_default_policy_too(self, registry):
        problem = make_problem(35, constraint_width=35, rhs=20)
        result = recommend(problem, registry, ExecutionPolicy(), compilers())
        assert result.recommendations[-1].backend == "dwave_qpu"
        assert by_name(result, "dwave_qpu").reasons == [
            "R_UNUSABLE",
            "R_DENSE_FOR_QPU",
            "R_REMOTE",
        ]

    def test_many_estimated_variables_also_count_as_dense(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True, exact_max_variables=500)
        result = recommend(make_problem(160), registry, policy, compilers())
        assert result.recommendations[-1].backend == "dwave_qpu"
        assert "R_DENSE_FOR_QPU" in by_name(result, "dwave_qpu").reasons


class TestPreferenceLimits:
    def test_num_reads_over_limit_blocks_qpu_but_not_sa(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True)
        result = recommend(knapsack(num_reads=5000), registry, policy, compilers())

        qpu = by_name(result, "dwave_qpu")
        assert qpu.usable is False
        assert [b.code for b in qpu.blocking] == ["QPU_READS_LIMIT"]
        assert qpu.reasons[0] == "R_UNUSABLE"
        sa = by_name(result, "simulated_annealing")
        assert sa.usable is True
        assert sa.blocking == []
        # Unusable sorts after every usable backend.
        assert result.recommendations[-1].backend == "dwave_qpu"

    def test_disabled_by_policy_is_blocking(self, registry):
        policy = ExecutionPolicy(enabled_backends={"simulated_annealing"})
        result = recommend(knapsack(), registry, policy, compilers())
        assert result.recommendations[0].backend == "simulated_annealing"
        exact = by_name(result, "exact")
        assert exact.usable is False
        assert [b.code for b in exact.blocking] == ["BACKEND_DISABLED_BY_POLICY"]


class TestExhaustiveLimit:
    def test_over_limit_is_blocking(self, registry):
        # 30 plain variables > the default ceiling of 24 compiled variables.
        result = recommend(make_problem(30), registry, ExecutionPolicy(), compilers())

        exact = by_name(result, "exact")
        assert exact.usable is False
        assert [b.code for b in exact.blocking] == ["EXACT_VARIABLE_LIMIT"]
        assert exact.reasons == ["R_UNUSABLE", "R_EXACT_OVER_LIMIT"]
        assert "EXACT_OVER_LIMIT" in [w.code for w in exact.warnings]
        assert result.recommendations[0].backend == "simulated_annealing"

    def test_near_limit_stays_usable_with_its_own_reason(self, registry):
        # 20 > 80% of 24, but within the ceiling.
        result = recommend(make_problem(20), registry, ExecutionPolicy(), compilers())
        exact = by_name(result, "exact")
        assert exact.usable is True
        assert exact.reasons == ["R_EXACT_NEAR_LIMIT"]
        assert result.recommendations[0].backend == "exact"


class TestNoCompiler:
    def test_missing_compiler_blocks_with_model_type_none(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        only_bqm = {BQMCompiler().model_type: BQMCompiler()}
        result = recommend(
            knapsack(), registry, ExecutionPolicy(allow_remote=True), only_bqm
        )

        cqm = by_name(result, "leap_hybrid_cqm")
        assert cqm.model_type is None
        assert cqm.usable is False
        assert [b.code for b in cqm.blocking] == ["NO_COMPILER_FOR_MODEL_TYPE"]
        assert by_name(result, "exact").model_type == "bqm"


class TestInvalidProblem:
    def test_invalid_problem_has_no_recommendations(self, registry):
        problem = knapsack()
        problem.constraints[0].terms[0].variable = "ghost"

        result = recommend(problem, registry, ExecutionPolicy(), compilers())

        assert result.valid is False
        assert "UNKNOWN_VARIABLE" in [e.code for e in result.errors]
        assert result.recommendations == []
        assert result.advisory


class TestDeterminism:
    def test_same_input_gives_equal_results(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True)
        first = recommend(knapsack(), registry, policy, compilers())
        second = recommend(knapsack(), registry, policy, compilers())
        assert first == second
        assert first.model_dump() == second.model_dump()


class TestReasonCatalog:
    def test_every_reason_code_has_a_description(self):
        assert set(REASON_DESCRIPTIONS) == ALL_REASON_CODES
        assert all(text.strip() for text in REASON_DESCRIPTIONS.values())

    def test_every_emitted_reason_is_described(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        scenarios = [
            (knapsack(), ExecutionPolicy()),
            (knapsack(), ExecutionPolicy(allow_remote=True)),
            (make_problem(3), ExecutionPolicy(allow_remote=True)),
            (make_problem(30), ExecutionPolicy()),
            (make_problem(20), ExecutionPolicy()),
            (
                make_problem(35, constraint_width=35, rhs=20),
                ExecutionPolicy(allow_remote=True, exact_max_variables=200),
            ),
        ]
        seen: set[str] = set()
        for problem, policy in scenarios:
            for entry in recommend(problem, registry, policy, compilers()).recommendations:
                seen.update(entry.reasons)
        assert seen <= set(REASON_DESCRIPTIONS)
        # The scenarios above exercise the whole vocabulary.
        assert seen == ALL_REASON_CODES


class TestNothingIsSolved:
    def test_no_backend_solve_or_time_limit_is_called(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        forbid_solving(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True)

        result = recommend(knapsack(), registry, policy, compilers())
        assert result.valid is True
        assert len(result.recommendations) == len(registry.names())

    def test_disallowed_remotes_are_not_even_asked_for_availability(
        self, monkeypatch, registry
    ):
        def _forbidden():
            raise AssertionError("is_available must not run when policy blocks remote")

        for name in remote_names(registry):
            monkeypatch.setattr(registry.get(name), "is_available", _forbidden)
        result = recommend(knapsack(), registry, ExecutionPolicy(), compilers())
        assert result.valid is True


class TestServiceDelegation:
    def test_service_recommend_equals_the_pure_function(self, monkeypatch, registry):
        make_remotes_available(monkeypatch, registry)
        forbid_solving(monkeypatch, registry)
        policy = ExecutionPolicy(allow_remote=True)
        service = OptimizationService(registry=registry, policy=policy)

        assert service.recommend(knapsack()) == recommend(
            knapsack(), registry, policy, compilers()
        )

    def test_service_uses_the_same_model_type_rule_as_validate(self, registry):
        service = OptimizationService(registry=registry)
        result = service.recommend(knapsack())
        for entry in result.recommendations:
            validated = service.validate(knapsack(backend=entry.backend))
            assert entry.model_type == validated.model_type
            assert entry.estimated_compiled_variables == (
                validated.estimated_compiled_variables
            )
            assert entry.warnings == validated.warnings


def _reason_constants_in_source() -> tuple[set[str], set[str], set[str]]:
    """``(emitted, keyed, all_r)`` R_ string constants found in routing.py.

    ``emitted`` are the codes appended to a ``reasons`` list, ``keyed`` the
    keys of the ``REASON_DESCRIPTIONS`` dict literal, ``all_r`` every R_
    constant anywhere in the module (3a §23.4 / step 10).
    """
    from annealbridge.orchestration import routing

    tree = ast.parse(Path(routing.__file__).read_text(encoding="utf-8"))
    emitted: set[str] = set()
    keyed: set[str] = set()
    all_r: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("R_"):
                all_r.add(node.value)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "append":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.startswith("R_"):
                            emitted.add(arg.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value.startswith("R_"):
                        keyed.add(key.value)
    return emitted, keyed, all_r


class TestReasonCatalogMatchesTheSource:
    """The reason codes *emitted* by routing.py, collected from its source,
    are exactly the described vocabulary (3a §23.4, step 10)."""

    def test_every_source_reason_code_is_described(self):
        emitted, keyed, _all_r = _reason_constants_in_source()
        assert emitted, "no reason emitters were found; the collector is broken"
        assert emitted == set(REASON_DESCRIPTIONS)
        assert keyed == set(REASON_DESCRIPTIONS)
        assert emitted == ALL_REASON_CODES

    def test_no_other_r_constant_hides_in_the_module(self):
        _emitted, _keyed, all_r = _reason_constants_in_source()
        assert all_r == set(REASON_DESCRIPTIONS)
