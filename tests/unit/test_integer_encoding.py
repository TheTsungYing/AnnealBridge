"""Binary expansion of integer variables (3b spec §14).

``compiler/integer_encoding.py`` is the single place that knows an integer
variable is really a string of bits. Everything here checks that knowledge
against the two sources it must never drift from:

* the *arithmetic* — ``integer_encoding_bits`` /
  ``compute_slack_coefficients`` in ``validation.estimates``, which the
  validator's estimates use, so a compiled model can never be a different
  size than the estimate promised;
* the *meaning* — for every bit pattern and every business assignment the
  substituted expression evaluates to exactly the same number as the
  original expression over business variables. That is checked by brute
  force: bit patterns are enumerated with ``itertools.product`` and random
  assignments are converted back into bits by search, so the tests never
  re-implement the expansion they are testing.

Identity is the other half of the contract: a binary variable is its own
affine form, so an all-binary problem produces no encodings and
``substitute_linear`` returns its input unchanged — which is what keeps
the Phase 1/2/3a compiled output bit-for-bit stable.
"""

import itertools
import random

import dimod
import pytest

from annealbridge.compiler.integer_encoding import (
    AffineForm,
    encode_integer_variables,
    expand_product,
    expand_square,
    expand_square_qm,
    substitute_linear,
)
from annealbridge.compiler.objective import build_objective_bqm
from annealbridge.models import (
    IntegerEncoding,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    Variable,
)
from annealbridge.validation.estimates import (
    compute_slack_coefficients,
    integer_encoding_bits,
)

SEED = 20260908
CASES = 200

# (lower, upper) pairs: negative ranges, a fully negative one, the
# degenerate integer-that-is-really-a-bit, a power-of-two range and an
# offset one whose top coefficient is a remainder rather than a power.
BOUND_CASES = [(-3, 2), (-5, -1), (0, 1), (0, 7), (2, 9), (-1, 0), (-4, 6)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_problem(variables: list[Variable]) -> OptimizationProblem:
    """A minimal, schema-valid 1.1 problem carrying ``variables``."""
    return OptimizationProblem(
        version="1.1",
        name="integer-encoding",
        variables=variables,
        objective=Objective(
            direction="minimize",
            linear_terms=[LinearTerm(variable=variables[0].name, coefficient=1.0)],
        ),
        constraints=[],
    )


def integer_variable(name: str, lower: int, upper: int) -> Variable:
    return Variable(name=name, type="integer", lower_bound=lower, upper_bound=upper)


def form_value(form: AffineForm, assignment: dict[str, float]) -> float:
    """Evaluate ``constant + sum(coefficients[v] * assignment[v])``."""
    return form.constant + sum(
        coefficient * assignment[name] for name, coefficient in form.coefficients.items()
    )


def bits_for_value(encoding: IntegerEncoding, value: int) -> dict[str, int]:
    """Find one bit pattern of ``encoding`` that represents ``value``.

    Brute force on purpose: the test must not reproduce the expansion's own
    arithmetic to check it. Raises if the value is unreachable, which is
    itself a failure worth surfacing.
    """
    for pattern in itertools.product((0, 1), repeat=len(encoding.coefficients)):
        total = encoding.lower + sum(
            coefficient * bit for coefficient, bit in zip(encoding.coefficients, pattern)
        )
        if total == value:
            return dict(zip(encoding.bits, pattern))
    raise AssertionError(f"value {value} is not representable by {encoding!r}")


def compiled_assignment(
    encodings: dict[str, IntegerEncoding], values: dict[str, int]
) -> dict[str, int]:
    """Turn a business assignment into an assignment over compiled variables."""
    assignment: dict[str, int] = {}
    for name, value in values.items():
        encoding = encodings.get(name)
        if encoding is None:
            assignment[name] = value
        else:
            assignment.update(bits_for_value(encoding, value))
    return assignment


def evaluate_substitution(
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
    constant: float,
    assignment: dict[str, int],
) -> float:
    """``sum(linear * bit) + sum(quadratic * bit * bit) + constant``."""
    total = constant
    total += sum(value * assignment[name] for name, value in linear.items())
    total += sum(
        value * assignment[left] * assignment[right]
        for (left, right), value in quadratic.items()
    )
    return total


# --------------------------------------------------------------------------
# Random generation
# --------------------------------------------------------------------------


def random_variables(rng: random.Random) -> list[Variable]:
    """One to four variables, a mix of binary and (possibly negative) integer."""
    variables: list[Variable] = []
    for index in range(rng.randint(1, 4)):
        name = f"v{index}"
        if rng.random() < 0.35:
            variables.append(Variable(name=name))
        else:
            lower = rng.randint(-4, 2)
            variables.append(integer_variable(name, lower, lower + rng.randint(1, 5)))
    return variables


def random_coefficient(rng: random.Random) -> float:
    return rng.choice([-2.0, -1.5, -1.0, 0.5, 1.0, 2.5, 3.0])


def random_assignment(rng: random.Random, variables: list[Variable]) -> dict[str, int]:
    return {
        variable.name: rng.randint(*variable.bounds()) for variable in variables
    }


# --------------------------------------------------------------------------
# 1. Bit count, naming and encoding metadata
# --------------------------------------------------------------------------


class TestEncodeIntegerVariables:
    @pytest.mark.parametrize("lower,upper", BOUND_CASES)
    def test_bit_count_naming_and_coefficients_follow_the_estimates(self, lower, upper):
        problem = make_problem([integer_variable("x", lower, upper)])

        forms, encodings = encode_integer_variables(problem)

        encoding = encodings["x"]
        expected_bits = integer_encoding_bits(lower, upper)
        assert len(encoding.bits) == expected_bits
        assert encoding.bits == [f"__int_x_{k}" for k in range(expected_bits)]
        assert encoding.coefficients == compute_slack_coefficients(upper - lower)
        assert encoding.lower == lower
        assert encoding.variable == "x"

        form = forms["x"]
        assert form.constant == float(lower)
        # The form's coefficients are the encoding's, in bit order.
        assert list(form.coefficients) == encoding.bits
        assert list(form.coefficients.values()) == [
            float(c) for c in encoding.coefficients
        ]

    @pytest.mark.parametrize("lower,upper", BOUND_CASES)
    def test_every_bit_pattern_is_in_range_and_every_value_is_reachable(
        self, lower, upper
    ):
        problem = make_problem([integer_variable("x", lower, upper)])
        forms, encodings = encode_integer_variables(problem)
        form = forms["x"]
        bits = encodings["x"].bits

        values = set()
        for pattern in itertools.product((0, 1), repeat=len(bits)):
            value = form_value(form, dict(zip(bits, pattern)))
            assert value == int(value)
            assert lower <= value <= upper, (pattern, value)
            values.add(int(value))

        assert values == set(range(lower, upper + 1))

    def test_a_binary_variable_is_its_own_identity_form(self):
        problem = make_problem([Variable(name="flag")])

        forms, encodings = encode_integer_variables(problem)

        assert forms == {"flag": AffineForm(constant=0.0, coefficients={"flag": 1.0})}
        assert encodings == {}

    def test_an_all_binary_problem_has_no_encodings(self):
        variables = [Variable(name=name) for name in ("c", "a", "b")]

        forms, encodings = encode_integer_variables(make_problem(variables))

        assert encodings == {}
        for variable in variables:
            assert forms[variable.name] == AffineForm(
                constant=0.0, coefficients={variable.name: 1.0}
            )

    def test_form_keys_follow_the_declaration_order(self):
        variables = [
            integer_variable("z", -3, 2),
            Variable(name="a"),
            integer_variable("m", 0, 5),
            Variable(name="b"),
        ]

        forms, encodings = encode_integer_variables(make_problem(variables))

        assert list(forms) == ["z", "a", "m", "b"]
        # Encodings keep the same relative order, integers only.
        assert list(encodings) == ["z", "m"]

    def test_bounds_of_width_one_still_take_one_bit(self):
        # upper - lower == 1: one bit whose coefficient is 1.
        forms, encodings = encode_integer_variables(
            make_problem([integer_variable("x", -1, 0)])
        )
        assert encodings["x"].coefficients == [1]
        assert forms["x"] == AffineForm(constant=-1.0, coefficients={"__int_x_0": 1.0})


# --------------------------------------------------------------------------
# 2. substitute_linear
# --------------------------------------------------------------------------


class TestSubstituteLinear:
    def test_identity_forms_return_the_input_unchanged(self):
        variables = [Variable(name=name) for name in ("c", "a", "b")]
        forms, _ = encode_integer_variables(make_problem(variables))
        coefficients = {"c": 3.0, "a": -1.5, "b": 2.0}

        bit_coefficients, constant = substitute_linear(coefficients, forms)

        assert bit_coefficients == coefficients
        assert constant == 0.0

    def test_lower_bounds_move_into_the_constant(self):
        variables = [integer_variable("x", -3, 2), Variable(name="f")]
        forms, encodings = encode_integer_variables(make_problem(variables))

        bit_coefficients, constant = substitute_linear({"x": 2.0, "f": -1.0}, forms)

        assert constant == 2.0 * -3
        assert bit_coefficients == {
            "__int_x_0": 2.0,
            "__int_x_1": 4.0,
            "__int_x_2": 4.0,
            "f": -1.0,
        }
        # ... and those are exactly 2 * the encoding coefficients.
        assert encodings["x"].coefficients == [1, 2, 2]

    def test_random_assignments_evaluate_to_the_business_expression(self):
        rng = random.Random(SEED)
        for _ in range(CASES):
            variables = random_variables(rng)
            forms, encodings = encode_integer_variables(make_problem(variables))
            coefficients = {
                variable.name: random_coefficient(rng)
                for variable in variables
                if rng.random() < 0.85
            }
            if not coefficients:
                continue
            values = random_assignment(rng, variables)
            assignment = compiled_assignment(encodings, values)

            bit_coefficients, constant = substitute_linear(coefficients, forms)

            expected = sum(
                coefficient * values[name] for name, coefficient in coefficients.items()
            )
            actual = constant + sum(
                value * assignment[bit] for bit, value in bit_coefficients.items()
            )
            assert actual == pytest.approx(expected), (variables, coefficients, values)


# --------------------------------------------------------------------------
# 3. Quadratic terms through build_objective_bqm
#
# 2026-09-09 review (F-13e): the former ``substitute_quadratic`` duplicated
# the per-term ``expand_product`` accumulation of ``build_objective_bqm``
# with a different float association and had no production caller, so the
# production builder is what these tests now exercise directly.
# --------------------------------------------------------------------------


def quadratic_objective(terms: list[QuadraticTerm]) -> Objective:
    return Objective(direction="minimize", linear_terms=[], quadratic_terms=terms)


def bqm_energy(bqm: dimod.BinaryQuadraticModel, assignment: dict[str, int]) -> float:
    return float(bqm.energy({name: assignment[name] for name in bqm.variables}))


def quadratic_case_variables() -> list[Variable]:
    """Two integers (one with a negative lower bound) and two binaries."""
    return [
        integer_variable("x", -2, 3),
        integer_variable("y", 1, 4),
        Variable(name="b"),
        Variable(name="d"),
    ]


class TestQuadraticTermsInObjectiveBqm:
    @pytest.mark.parametrize(
        "first,second",
        [("x", "x"), ("x", "y"), ("y", "y"), ("x", "b"), ("b", "x"), ("b", "d")],
    )
    def test_one_term_evaluates_to_the_business_product(self, first, second):
        rng = random.Random(SEED)
        variables = quadratic_case_variables()
        forms, encodings = encode_integer_variables(make_problem(variables))
        term = QuadraticTerm(variable1=first, variable2=second, coefficient=-1.5)

        bqm = build_objective_bqm(quadratic_objective([term]), forms)

        # Every compiled variable is a bit, so a square always folds into
        # the linear part (a BINARY BQM cannot even hold ``(v, v)``).
        assert all(left != right for left, right in bqm.quadratic), bqm.quadratic
        for _ in range(CASES):
            values = random_assignment(rng, variables)
            assignment = compiled_assignment(encodings, values)
            expected = -1.5 * values[first] * values[second]
            assert bqm_energy(bqm, assignment) == pytest.approx(expected), values

    def test_a_pair_and_its_reverse_share_one_entry(self):
        variables = quadratic_case_variables()
        forms, encodings = encode_integer_variables(make_problem(variables))
        terms = [
            QuadraticTerm(variable1="x", variable2="y", coefficient=1.0),
            QuadraticTerm(variable1="y", variable2="x", coefficient=2.0),
        ]

        bqm = build_objective_bqm(quadratic_objective(terms), forms)

        # The merged model is the same as one term with c = 3.
        merged = build_objective_bqm(
            quadratic_objective(
                [QuadraticTerm(variable1="x", variable2="y", coefficient=3.0)]
            ),
            forms,
        )
        assert dict(bqm.linear) == pytest.approx(dict(merged.linear))
        assert {frozenset(k): v for k, v in bqm.quadratic.items()} == pytest.approx(
            {frozenset(k): v for k, v in merged.quadratic.items()}
        )
        assert bqm.offset == pytest.approx(merged.offset)

        rng = random.Random(SEED)
        for _ in range(CASES):
            values = random_assignment(rng, variables)
            assignment = compiled_assignment(encodings, values)
            expected = 3.0 * values["x"] * values["y"]
            assert bqm_energy(bqm, assignment) == pytest.approx(expected)

    def test_a_nonzero_constant_appears_for_products_of_shifted_variables(self):
        # x in -2..3 and y in 1..4: the constant is (-2) * 1 * coefficient.
        variables = quadratic_case_variables()
        forms, _ = encode_integer_variables(make_problem(variables))

        bqm = build_objective_bqm(
            quadratic_objective(
                [QuadraticTerm(variable1="x", variable2="y", coefficient=2.0)]
            ),
            forms,
        )

        assert bqm.offset == 2.0 * -2 * 1

    def test_random_term_lists_evaluate_to_the_business_expression(self):
        rng = random.Random(SEED + 1)
        for _ in range(CASES):
            variables = random_variables(rng)
            names = [variable.name for variable in variables]
            forms, encodings = encode_integer_variables(make_problem(variables))
            integer_names = {
                variable.name for variable in variables if variable.type == "integer"
            }
            terms = []
            for _ in range(rng.randint(1, 4)):
                first = rng.choice(names)
                # b * b is not a legal IR term for a binary variable; x * x is.
                second = (
                    first
                    if first in integer_names and rng.random() < 0.35
                    else rng.choice(names)
                )
                if first == second and first not in integer_names:
                    continue
                terms.append(
                    QuadraticTerm(
                        variable1=first,
                        variable2=second,
                        coefficient=random_coefficient(rng),
                    )
                )
            if not terms:
                continue

            bqm = build_objective_bqm(quadratic_objective(terms), forms)

            # On the BQM path every compiled variable is a bit, so a square
            # always folds into the linear part.
            assert all(left != right for left, right in bqm.quadratic), bqm.quadratic
            values = random_assignment(rng, variables)
            assignment = compiled_assignment(encodings, values)
            expected = sum(
                term.coefficient * values[term.variable1] * values[term.variable2]
                for term in terms
            )
            assert bqm_energy(bqm, assignment) == pytest.approx(expected), (terms, values)


# --------------------------------------------------------------------------
# 3b. expand_square: the one squared-penalty expansion (review F-13a)
# --------------------------------------------------------------------------


def bqm_penalty_formula(
    coefficients: dict[str, float], constant: float, weight: float
) -> tuple[dict[str, float], dict[tuple[str, str], float], float]:
    """The Phase 1 ``_add_squared_penalty`` arithmetic, spelled out verbatim.

    Written the way the BQM compiler always wrote it (and the 3a golden
    test pins): ``lam * (a*a + 2.0*c0*a)``, ``2.0 * lam * a_i * a_j``,
    ``lam * c0 * c0``. ``expand_square`` must reproduce these floats exactly,
    not approximately.
    """
    items = list(coefficients.items())
    linear = {v: weight * (a * a + 2.0 * constant * a) for v, a in items}
    quadratic = {}
    for i, (var_i, value_i) in enumerate(items):
        for var_j, value_j in items[i + 1 :]:
            quadratic[(var_i, var_j)] = 2.0 * weight * value_i * value_j
    return linear, quadratic, weight * constant * constant


def random_square_inputs(rng: random.Random) -> tuple[dict[str, float], float, float]:
    """Non-dyadic floats on purpose: rounding differences must show up."""
    names = [f"v{i}" for i in range(rng.randint(1, 5))]
    coefficients = {name: rng.uniform(-7.0, 7.0) for name in names}
    return coefficients, rng.uniform(-9.0, 9.0), rng.uniform(0.01, 1000.0)


class TestExpandSquare:
    def test_is_bit_identical_to_the_bqm_penalty_formula(self):
        rng = random.Random(SEED + 2)
        for _ in range(CASES):
            coefficients, constant, weight = random_square_inputs(rng)

            linear, quadratic, offset = expand_square(coefficients, constant, weight)

            expected = bqm_penalty_formula(coefficients, constant, weight)
            # Exact equality (no approx): this is the golden-pinned order.
            assert (linear, quadratic, offset) == expected
            assert list(linear) == list(coefficients)

    def test_every_pair_appears_exactly_once_in_coefficient_order(self):
        coefficients = {"a": 1.5, "b": -2.0, "c": 0.25}

        _, quadratic, _ = expand_square(coefficients, 0.0, 1.0)

        assert list(quadratic) == [("a", "b"), ("a", "c"), ("b", "c")]

    def test_unfolded_square_stays_a_quadratic_entry(self):
        coefficients = {"x": 3.0, "b": 1.0}

        linear, quadratic, offset = expand_square(
            coefficients, -2.0, 0.5, fold_square=lambda name: name == "b"
        )

        # x keeps its square: linear part is only the cross term with the
        # constant, the square is the (x, x) entry.
        assert linear["x"] == 0.5 * (2.0 * -2.0 * 3.0)
        assert quadratic[("x", "x")] == 0.5 * (3.0 * 3.0)
        # b folds: the full diagonal goes to the linear part.
        assert linear["b"] == 0.5 * (1.0 * 1.0 + 2.0 * -2.0 * 1.0)
        assert ("b", "b") not in quadratic
        assert quadratic[("x", "b")] == 2.0 * 0.5 * 3.0 * 1.0
        assert offset == 0.5 * -2.0 * -2.0

    def test_evaluates_to_the_weighted_square(self):
        rng = random.Random(SEED + 3)
        for _ in range(CASES):
            coefficients, constant, weight = random_square_inputs(rng)
            unfolded = {name for name in coefficients if rng.random() < 0.5}
            linear, quadratic, offset = expand_square(
                coefficients, constant, weight, fold_square=lambda n: n not in unfolded
            )
            # Folded names are bits (0/1); unfolded ones may take any integer.
            assignment = {
                name: rng.randint(-5, 5) if name in unfolded else rng.randint(0, 1)
                for name in coefficients
            }
            affine = constant + sum(a * assignment[n] for n, a in coefficients.items())

            actual = evaluate_substitution(linear, quadratic, offset, assignment)

            assert actual == pytest.approx(weight * affine * affine)

    def test_qm_adapter_uses_the_same_floats(self):
        rng = random.Random(SEED + 4)
        for _ in range(CASES):
            coefficients, constant, weight = random_square_inputs(rng)
            qm = dimod.QuadraticModel()
            unfolded = set()
            for name in coefficients:
                if rng.random() < 0.5:
                    qm.add_variable("INTEGER", name, lower_bound=-5, upper_bound=5)
                    unfolded.add(name)
                else:
                    qm.add_variable("BINARY", name)

            expand_square_qm(qm, coefficients, constant, weight)

            linear, quadratic, offset = expand_square(
                coefficients, constant, weight, fold_square=lambda n: n not in unfolded
            )
            assert {name: qm.get_linear(name) for name in coefficients} == linear
            assert {pair: qm.get_quadratic(*pair) for pair in quadratic} == quadratic
            assert qm.offset == offset


# --------------------------------------------------------------------------
# 4. expand_product: fold_square and scale
# --------------------------------------------------------------------------


class TestExpandProduct:
    @staticmethod
    def zero_based_form() -> AffineForm:
        """``0 + 1*b0 + 2*b1 + 2*b2`` -- an integer in 0..5, constant-free.

        A zero constant is what makes the fold visible: without it the
        linear part of ``f * f`` contains *only* the folded squares.
        """
        forms, _ = encode_integer_variables(
            make_problem([integer_variable("x", 0, 5)])
        )
        return forms["x"]

    def test_squares_fold_into_the_linear_part_by_default(self):
        form = self.zero_based_form()

        linear, quadratic, constant = expand_product(form, form)

        assert constant == 0.0
        assert not [key for key in quadratic if key[0] == key[1]]
        # Exactly the squared coefficients, one per bit.
        assert linear == {"__int_x_0": 1.0, "__int_x_1": 4.0, "__int_x_2": 4.0}
        # Each unordered pair of distinct bits, counted twice (i*j and j*i).
        assert quadratic == {
            ("__int_x_0", "__int_x_1"): 4.0,
            ("__int_x_0", "__int_x_2"): 4.0,
            ("__int_x_1", "__int_x_2"): 8.0,
        }

    def test_fold_square_false_keeps_the_self_pair_quadratic(self):
        form = self.zero_based_form()

        linear, quadratic, constant = expand_product(
            form, form, fold_square=lambda _: False
        )

        assert constant == 0.0
        assert linear == {}
        assert [key for key in quadratic if key[0] == key[1]] == [
            ("__int_x_0", "__int_x_0"),
            ("__int_x_1", "__int_x_1"),
            ("__int_x_2", "__int_x_2"),
        ]
        assert quadratic[("__int_x_0", "__int_x_0")] == 1.0
        assert quadratic[("__int_x_1", "__int_x_1")] == 4.0

    def test_both_folding_modes_agree_on_every_bit_pattern(self):
        # b*b == b for a 0/1 value, so folding cannot change any evaluation.
        form = self.zero_based_form()
        bits = list(form.coefficients)
        folded = expand_product(form, form)
        kept = expand_product(form, form, fold_square=lambda _: False)

        for pattern in itertools.product((0, 1), repeat=len(bits)):
            assignment = dict(zip(bits, pattern))
            value = form_value(form, assignment)
            assert evaluate_substitution(*folded, assignment) == pytest.approx(
                value * value
            )
            assert evaluate_substitution(*kept, assignment) == pytest.approx(
                value * value
            )

    @pytest.mark.parametrize("scale", [-2.5, -1.0, 0.5, 3.0])
    def test_scale_multiplies_every_part(self, scale):
        left = AffineForm(constant=-3.0, coefficients={"p": 1.0, "q": 2.0})
        right = AffineForm(constant=2.0, coefficients={"q": 1.5, "r": -1.0})

        base_linear, base_quadratic, base_constant = expand_product(left, right)
        linear, quadratic, constant = expand_product(left, right, scale=scale)

        assert constant == pytest.approx(scale * base_constant)
        assert linear == pytest.approx(
            {name: scale * value for name, value in base_linear.items()}
        )
        assert quadratic == pytest.approx(
            {key: scale * value for key, value in base_quadratic.items()}
        )

    def test_constants_produce_the_cross_linear_terms_and_the_product_constant(self):
        left = AffineForm(constant=-3.0, coefficients={"p": 1.0})
        right = AffineForm(constant=2.0, coefficients={"r": 4.0})

        linear, quadratic, constant = expand_product(left, right)

        assert constant == -6.0
        # left.constant * right coefficients, right.constant * left coefficients
        assert linear == {"r": -12.0, "p": 2.0}
        assert quadratic == {("p", "r"): 4.0}
