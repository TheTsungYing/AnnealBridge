"""BQM 分段編譯（``BQMCompiler.prepare`` / ``PreparedBQM.compile``）與重試快取的測試。

必守條件：同一次 solve 內重用與 hard penalty 無關的編譯產物時，每一次嘗試
（不只第 1 次）產出的 BQM 都必須與從頭編譯位元完全相同——變數順序、
linear / quadratic 係數（含 dtype）、offset，以及 trace、內部變數、整數編碼等
附帶資訊。這裡的「從頭編譯」是每次都重新建立 ``BQMCompiler`` 並呼叫
``compile``；與舊版演算法逐位元相同則由 golden 測試
（``test_golden_phase3a.py``，version 1.0）與本檔的 ``_reference_compile``
（分段前演算法的逐字抄錄，涵蓋整數、slack 與 soft）釘住。

服務層另外確認：快取只存在單次 solve 之內（每次 solve 各自 ``prepare``
一次），以及不支援 ``prepare`` 的 compiler 照舊每次嘗試從頭 ``compile``、
結果與快取路徑相同。

所有隨機部分都用固定種子；不 skip、不 xfail。
"""

import random

import numpy as np
import pytest

from annealbridge.compiler import (
    BQMCompiler,
    CQMCompiler,
    build_objective_bqm,
    encode_integer_variables,
    encode_slack,
    expand_square,
    substitute_linear,
)
from annealbridge.compiler.base import SupportsPrepare
from annealbridge.compiler.bqm import PreparedBQM, _BiasAccumulator
from annealbridge.compiler.slack import nonzero_coefficients
from annealbridge.exceptions import CompilationError, NonFiniteModelError
from annealbridge.models import ConstraintTrace, OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.validation.estimates import compute_objective_scale, variable_bounds
from tests.builders import hard, lin, quad, soft

# 預設策略的倍增梯，再加上反序、重複與極端值，確認 PreparedBQM 沒有被
# 前一次 compile 汙染。
PENALTY_LADDER = [0.31, 0.62, 1.24, 2.48]
PENALTY_ORDERS = [
    PENALTY_LADDER,
    list(reversed(PENALTY_LADDER)),
    [2.48, 2.48, 0.31, 1e3, 0.31],
]


def _problem(
    variables: list[dict],
    constraints: list[dict],
    *,
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    direction: str = "minimize",
    version: str = "1.0",
) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        {
            "version": version,
            "name": "prepared",
            "variables": variables,
            "objective": {
                "direction": direction,
                "linear_terms": linear or [],
                "quadratic_terms": quadratic or [],
            },
            "constraints": constraints,
            "solver": {"backend": "simulated_annealing"},
        }
    )


def _binary(*names: str) -> list[dict]:
    return [{"name": name, "type": "binary"} for name in names]


def _integer(name: str, lower: int, upper: int) -> dict:
    return {"name": name, "type": "integer", "lower_bound": lower, "upper_bound": upper}


def _binary_equality() -> OptimizationProblem:
    """純 binary：一條 one-hot 等式與一條三選二等式，目標含二次項。"""
    return _problem(
        _binary("a", "b", "c", "d"),
        [
            hard("one_hot", "==", 1, [lin("a", 1), lin("b", 1), lin("c", 1)]),
            hard("pair", "==", 2, [lin("b", 1), lin("c", 1), lin("d", 1)]),
        ],
        linear=[lin("a", 3), lin("b", -2), lin("c", 1.5), lin("d", 4)],
        quadratic=[quad("a", "d", 2), quad("b", "c", -1)],
        direction="maximize",
    )


def _bounded_integer() -> OptimizationProblem:
    """version 1.1 有界整數（含負下界）與 binary 混用。"""
    return _problem(
        [_integer("n", -3, 6), _integer("m", 0, 11), *_binary("x", "y")],
        [
            hard("sum", "==", 4, [lin("n", 1), lin("m", 1), lin("x", 2)]),
            hard("cap", "<=", 9, [lin("m", 1), lin("y", 3), lin("n", -1)]),
        ],
        linear=[lin("n", 2), lin("m", -1), lin("x", 5)],
        quadratic=[quad("n", "m", 1), quad("x", "n", -2)],
        version="1.1",
    )


def _inequality_slack() -> OptimizationProblem:
    """不等式 slack：``<=``、``>=``，以及一條永遠成立的 redundant 限制。"""
    return _problem(
        _binary("a", "b", "c", "d", "e"),
        [
            hard("cap", "<=", 7, [lin("a", 3), lin("b", 4), lin("c", 2), lin("d", 5)]),
            hard("floor", ">=", 2, [lin("b", 1), lin("c", 1), lin("e", 1)]),
            hard("always", "<=", 100, [lin("a", 1), lin("e", 1)]),
        ],
        linear=[lin("a", -3), lin("b", -4), lin("c", -1), lin("d", -6), lin("e", -2)],
    )


def _soft_constraints() -> OptimizationProblem:
    """soft 等式與不等式（含整數），與 hard 限制交錯排列、共用變數。"""
    return _problem(
        [_integer("k", 0, 5), *_binary("a", "b", "c")],
        [
            soft("prefer_two", "==", 2, [lin("a", 1), lin("b", 1), lin("c", 1)], 1.7),
            hard("cap", "<=", 6, [lin("k", 1), lin("a", 2), lin("b", 2)]),
            soft("at_most", "<=", 3, [lin("k", 1), lin("c", 1)], 0.45),
            hard("need", ">=", 1, [lin("a", 1), lin("c", 1)]),
            soft("never_binding", ">=", -50, [lin("k", 1)], 3.0),
        ],
        linear=[lin("k", 2), lin("a", -1), lin("c", 3)],
        quadratic=[quad("k", "a", 1)],
        version="1.1",
    )


def _non_dyadic() -> OptimizationProblem:
    """非二進位小數係數與 rhs：浮點捨入順序一改就會露出位元差異。"""
    return _problem(
        [_integer("n", 0, 7), *_binary("a", "b", "c")],
        [
            hard("mix", "<=", 3.3, [lin("a", 0.1), lin("b", 0.7), lin("n", 0.3)]),
            soft("tilt", "==", 1.1, [lin("a", 0.3), lin("c", 0.6), lin("n", 0.2)], 0.37),
            hard("bal", "==", 1, [lin("b", 0.1), lin("c", 0.9)]),
        ],
        linear=[lin("a", 0.1), lin("b", 0.2), lin("c", 0.3), lin("n", 0.7)],
        quadratic=[quad("a", "b", 0.1), quad("n", "c", 0.3)],
        direction="maximize",
        version="1.1",
    )


def _random_problem(seed: int) -> OptimizationProblem:
    """混合所有情境的隨機題（固定種子）。"""
    rng = random.Random(seed)
    variables = _binary(*(f"x{i}" for i in range(10)))
    variables += [_integer(f"k{i}", rng.randint(-2, 1), rng.randint(2, 9)) for i in range(4)]
    names = [variable["name"] for variable in variables]

    def coefficient() -> float:
        return round(rng.uniform(-2.0, 5.0), 2) or 1.0

    constraints = []
    for index in range(8):
        terms = [lin(name, coefficient()) for name in rng.sample(names, 5)]
        operator = rng.choice(["<=", ">=", "=="])
        if operator == "<=":
            rhs = 60.0  # 寬鬆上界：保證不會是 trivially infeasible
        elif operator == ">=":
            rhs = -60.0 if index % 3 == 0 else -1.5  # 前者 redundant
        else:
            rhs = 0.0  # 全取下界附近可達；等式不做 slack
        if index % 2:
            constraints.append(soft(f"s{index}", operator, rhs, terms, round(rng.uniform(0.2, 4.0), 3)))
        else:
            constraints.append(hard(f"h{index}", operator, rhs, terms))
    return _problem(
        variables,
        constraints,
        linear=[lin(name, coefficient()) for name in names],
        quadratic=[quad(*rng.sample(names, 2), coefficient()) for _ in range(12)],
        direction=rng.choice(["minimize", "maximize"]),
        version="1.1",
    )


CASES = {
    "binary_equality": _binary_equality,
    "bounded_integer": _bounded_integer,
    "inequality_slack": _inequality_slack,
    "soft_constraints": _soft_constraints,
    "non_dyadic": _non_dyadic,
    **{f"random_{seed}": (lambda seed=seed: _random_problem(seed)) for seed in range(6)},
}


def _reference_compile(problem: OptimizationProblem, hard_penalty: float):
    """分段編譯之前（115679d）的 ``BQMCompiler.compile``，逐字抄錄。

    一次跑完、不經 ``prepare``，讓「與從頭編譯相同」不是拿新程式和它自己比：
    ``BQMCompiler.compile`` 現在就是 ``prepare(problem).compile(...)``。
    golden 只釘住 version 1.0 題目，這裡另外守住整數、slack 與 soft 路徑。
    回傳 ``(bqm, internal_variables, constraint_trace, objective_scale)``。
    """
    bounds = variable_bounds(problem)
    forms, encodings = encode_integer_variables(problem)
    model = _BiasAccumulator()
    internal_variables: set[str] = set()
    for variable in problem.variables:
        encoding = encodings.get(variable.name)
        if encoding is None:
            model.add_variable(variable.name)
            continue
        for bit in encoding.bits:
            model.add_variable(bit)
        internal_variables.update(encoding.bits)

    objective_bqm = build_objective_bqm(problem.objective, forms)
    for variable, bias in objective_bqm.linear.items():
        model.add_linear(variable, float(bias))
    for u, v, bias in objective_bqm.iter_quadratic():
        model.add_quadratic(u, v, float(bias))
    model.offset += float(objective_bqm.offset)

    def add_squared_penalty(coefficients, constant, lam):
        linear, quadratic, offset = expand_square(coefficients, constant, lam)
        for variable, value in linear.items():
            model.add_linear(variable, value)
        for (var_i, var_j), value in quadratic.items():
            model.add_quadratic(var_i, var_j, value)
        model.offset += offset

    constraint_trace = []
    for constraint in problem.constraints:
        lam = hard_penalty if constraint.type == "hard" else constraint.weight
        generated_variables: list[str] = []
        slack_range = None
        redundant = False
        if constraint.operator == "==":
            coefficients = nonzero_coefficients(constraint.terms)
            bit_coefficients, shift = substitute_linear(coefficients, forms)
            add_squared_penalty(bit_coefficients, shift - constraint.rhs, lam)
        else:
            encoding = encode_slack(constraint, bounds)
            redundant = encoding.redundant
            slack_range = encoding.slack_range
            if not redundant:
                generated_variables = list(encoding.slack_coefficients)
                for name in generated_variables:
                    model.add_variable(name)
                internal_variables.update(generated_variables)
                coefficients, shift = substitute_linear(encoding.coefficients, forms)
                for name, value in encoding.slack_coefficients.items():
                    coefficients[name] = coefficients.get(name, 0.0) + float(value)
                add_squared_penalty(coefficients, encoding.constant + shift, lam)
        constraint_trace.append(
            ConstraintTrace(
                constraint_id=constraint.id,
                constraint_type=constraint.type,
                operator=constraint.operator,
                source_description=constraint.description,
                generated_variables=generated_variables,
                penalty=lam,
                slack_range=slack_range,
                redundant=redundant,
                compiler="BQMCompiler",
            )
        )
    return (
        model.to_bqm(),
        internal_variables,
        constraint_trace,
        compute_objective_scale(problem.objective, bounds),
    )


def _assert_same_bqm(actual, expected) -> None:
    """兩個 dimod BQM 逐位元相同：變數順序、係數（含 dtype）、offset。"""
    a_lin, (a_row, a_col, a_quad), a_off, a_labels = actual.to_numpy_vectors(
        return_labels=True
    )
    e_lin, (e_row, e_col, e_quad), e_off, e_labels = expected.to_numpy_vectors(
        return_labels=True
    )
    assert list(a_labels) == list(e_labels)
    for got, want in ((a_lin, e_lin), (a_row, e_row), (a_col, e_col), (a_quad, e_quad)):
        assert got.dtype == want.dtype
        assert got.tobytes() == want.tobytes()
    assert float(a_off).hex() == float(e_off).hex()
    assert actual.vartype == expected.vartype


def _assert_bit_identical(actual, expected) -> None:
    """兩個 CompiledProblem 逐位元相同（模型與所有附帶資訊）。"""
    _assert_same_bqm(actual.model, expected.model)
    assert actual.model_type == expected.model_type
    assert actual.hard_penalty == expected.hard_penalty
    assert actual.objective_scale.hex() == expected.objective_scale.hex()
    assert actual.internal_variables == expected.internal_variables
    assert actual.constraint_trace == expected.constraint_trace
    assert actual.integer_encodings == expected.integer_encodings
    assert actual.num_variables == expected.num_variables
    assert actual.num_interactions == expected.num_interactions
    assert actual.original_problem is expected.original_problem


class TestPreparedMatchesFromScratch:
    @pytest.mark.parametrize("case", sorted(CASES))
    @pytest.mark.parametrize("order", range(len(PENALTY_ORDERS)))
    def test_every_compile_is_bit_identical(self, case, order):
        problem = CASES[case]()
        prepared = BQMCompiler().prepare(problem)
        for penalty in PENALTY_ORDERS[order]:
            _assert_bit_identical(
                prepared.compile(penalty), BQMCompiler().compile(problem, penalty)
            )

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_every_compile_matches_the_pre_prepare_algorithm(self, case):
        """每次 compile 都與分段前的演算法（``_reference_compile``）位元相同。"""
        problem = CASES[case]()
        prepared = BQMCompiler().prepare(problem)
        for penalty in PENALTY_ORDERS[2]:
            compiled = prepared.compile(penalty)
            bqm, internal, trace, scale = _reference_compile(problem, penalty)
            _assert_same_bqm(compiled.model, bqm)
            assert compiled.internal_variables == internal
            assert compiled.constraint_trace == trace
            assert compiled.objective_scale.hex() == scale.hex()

    def test_cases_cover_every_path(self):
        """測資確實涵蓋整數編碼、slack、redundant、soft 等式與不等式。"""
        traces = [
            trace
            for build in CASES.values()
            for trace in BQMCompiler().prepare(build()).compile(1.0).constraint_trace
        ]
        assert any(t.redundant for t in traces)
        assert any(t.generated_variables for t in traces)
        assert any(t.constraint_type == "soft" and t.operator == "==" for t in traces)
        assert any(
            t.constraint_type == "soft" and t.operator != "==" and t.generated_variables
            for t in traces
        )
        assert any(t.constraint_type == "hard" and t.operator == ">=" for t in traces)
        assert BQMCompiler().compile(_bounded_integer(), 1.0).integer_encodings

    def test_compiled_models_do_not_share_mutable_state(self):
        """修改一次 compile 的結果，不影響之後的 compile。"""
        problem = _soft_constraints()
        prepared = BQMCompiler().prepare(problem)
        first = prepared.compile(1.0)
        first.model.add_linear(first.model.variables[0], 123.0)
        first.model.offset += 7.0
        first.internal_variables.add("__polluted")
        first.constraint_trace.clear()
        first.integer_encodings.clear()
        _assert_bit_identical(prepared.compile(1.0), BQMCompiler().compile(problem, 1.0))

    def test_integer_encodings_are_not_shared(self):
        """改動某次結果的 IntegerEncoding 內容，不影響之後的 compile。"""
        problem = _bounded_integer()
        prepared = BQMCompiler().prepare(problem)
        first = prepared.compile(1.0)
        encoding = first.integer_encodings["m"]
        encoding.coefficients[0] = 999
        encoding.bits.append("__polluted")
        _assert_bit_identical(prepared.compile(1.0), BQMCompiler().compile(problem, 1.0))

    def test_penalties_stay_separate(self):
        """hard 用傳入的 penalty，soft 用自身 weight，兩者不互相替代。"""
        compiled = BQMCompiler().prepare(_soft_constraints()).compile(9.5)
        penalties = {t.constraint_id: t.penalty for t in compiled.constraint_trace}
        assert penalties == {
            "prefer_two": 1.7,
            "cap": 9.5,
            "at_most": 0.45,
            "need": 9.5,
            "never_binding": 3.0,
        }


class TestPreparedErrors:
    def test_none_penalty_is_rejected(self):
        prepared = BQMCompiler().prepare(_binary_equality())
        with pytest.raises(CompilationError, match="requires a hard_penalty"):
            prepared.compile(None)

    def test_overflow_then_finite_compile(self):
        """溢位的 compile 拋 NonFiniteModelError，之後的 compile 仍與從頭編譯相同。"""
        problem = _inequality_slack()
        prepared = BQMCompiler().prepare(problem)
        with pytest.raises(NonFiniteModelError):
            prepared.compile(1.7e308)
        _assert_bit_identical(prepared.compile(2.48), BQMCompiler().compile(problem, 2.48))

    def test_problem_error_surfaces_in_prepare(self):
        """不可能滿足的 hard 不等式：prepare 與 compile 拋出相同錯誤。"""
        problem = _problem(
            _binary("a", "b"),
            [hard("impossible", ">=", 5, [lin("a", 1), lin("b", 1)])],
        )
        with pytest.raises(CompilationError) as from_prepare:
            BQMCompiler().prepare(problem)
        with pytest.raises(CompilationError) as from_compile:
            BQMCompiler().compile(problem, 1.0)
        assert str(from_prepare.value) == str(from_compile.value)


class TestCapability:
    def test_bqm_compiler_supports_prepare(self):
        assert isinstance(BQMCompiler(), SupportsPrepare)
        assert isinstance(BQMCompiler().prepare(_binary_equality()), PreparedBQM)

    def test_cqm_compiler_does_not(self):
        """CQM 沒有 hard penalty、不會重試，所以不提供 prepare。"""
        assert not isinstance(CQMCompiler(), SupportsPrepare)


# ---------------------------------------------------------------------------
# 服務層：快取只活在單次 solve 之內
# ---------------------------------------------------------------------------

# knapsack 範例配極小 multiplier：第 1 次嘗試必定找不到可行解而重試
# （見 tests/scenarios/test_retry.py 的推導）。
TINY_MULTIPLIER = 0.01


class _RecordingPrepared:
    def __init__(self, inner: PreparedBQM, log: list) -> None:
        self._inner = inner
        self._log = log

    def compile(self, hard_penalty):
        compiled = self._inner.compile(hard_penalty)
        self._log.append(compiled)
        return compiled


class RecordingBQMCompiler(BQMCompiler):
    """記錄 prepare 次數與每次嘗試編出的模型。"""

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.compiled: list = []

    def prepare(self, problem):
        self.prepare_calls += 1
        return _RecordingPrepared(super().prepare(problem), self.compiled)


class PlainBQMCompiler:
    """只有 ``compile`` 的 compiler：不支援 prepare，每次嘗試從頭編譯。"""

    model_type = "bqm"
    uses_hard_penalty = True

    def __init__(self) -> None:
        self._inner = BQMCompiler()
        self.compile_calls = 0

    def compile(self, problem, hard_penalty):
        self.compile_calls += 1
        return self._inner.compile(problem, hard_penalty)

    def decode(self, compiled, raw):
        return self._inner.decode(compiled, raw)


@pytest.fixture
def retry_problem(load_example):
    return OptimizationProblem.model_validate(
        load_example(
            "knapsack.json",
            backend="simulated_annealing",
            seed=0,
            num_reads=100,
            penalty_multiplier=TINY_MULTIPLIER,
        )
    )


class TestServiceCache:
    def test_prepare_once_per_solve_and_every_attempt_bit_identical(self, retry_problem):
        compiler = RecordingBQMCompiler()
        service = OptimizationService(compilers=[compiler, CQMCompiler()])

        result = service.solve(retry_problem)

        assert result.status == "success"
        assert len(result.attempts) > 1
        assert compiler.prepare_calls == 1
        assert len(compiler.compiled) == len(result.attempts)
        for attempt, compiled in zip(result.attempts, compiler.compiled, strict=True):
            assert compiled.hard_penalty == attempt.penalty
            _assert_bit_identical(
                compiled, BQMCompiler().compile(retry_problem, attempt.penalty)
            )

    def test_cache_does_not_outlive_a_solve(self, retry_problem):
        compiler = RecordingBQMCompiler()
        service = OptimizationService(compilers=[compiler, CQMCompiler()])

        first = service.solve(retry_problem)
        second = service.solve(retry_problem)

        assert compiler.prepare_calls == 2
        assert len(compiler.compiled) == len(first.attempts) + len(second.attempts)

    def test_compiler_without_prepare_gives_the_same_result(self, retry_problem):
        plain = PlainBQMCompiler()
        uncached = OptimizationService(compilers=[plain, CQMCompiler()]).solve(
            retry_problem
        )
        cached = OptimizationService().solve(retry_problem)

        assert plain.compile_calls == len(uncached.attempts) > 1
        assert [a.penalty for a in cached.attempts] == [a.penalty for a in uncached.attempts]
        assert [a.feasible_samples for a in cached.attempts] == [
            a.feasible_samples for a in uncached.attempts
        ]
        assert cached.solutions == uncached.solutions


def test_numpy_dtype_guard():
    """``_assert_bit_identical`` 真的會抓到 1 ulp 的差異。"""
    problem = _non_dyadic()
    compiled = BQMCompiler().compile(problem, 1.0)
    other = BQMCompiler().compile(problem, 1.0)
    variable = other.model.variables[0]
    bias = other.model.get_linear(variable)
    other.model.set_linear(variable, float(np.nextafter(bias, np.inf)))
    with pytest.raises(AssertionError):
        _assert_bit_identical(other, compiled)
