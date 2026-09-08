"""Binary expansion of integer variables for the BQM path (3b spec §14).

The IR carries only ``type: "integer"`` and bounds; turning that into bits
is the compiler's job and nothing above it ever sees the bits. Every
business variable gets an :class:`AffineForm` over compiled variables:

* a binary variable is its own form, ``0 + 1 * name`` (identity);
* an integer variable ``x in lower..upper`` becomes
  ``lower + sum(coefficients[k] * __int_x_k)`` with the coefficients of
  :func:`~annealbridge.validation.estimates.compute_slack_coefficients`
  applied to ``upper - lower`` — the same expansion the slack bits use, so
  "every bit pattern stays inside the range and every integer is
  reachable" is guaranteed by the same existing test, and the bit count is
  exactly :func:`~annealbridge.validation.estimates.integer_encoding_bits`.

The substitution helpers rewrite linear and quadratic expressions over
business variables into expressions over compiled variables. The
expansion core, :func:`expand_product`, is shared with the CQM path's
``expand_square_qm`` (3b step 4): the only difference between the two is
whether a self-product ``v * v`` folds into a linear term (a bit, since
``b * b == b``) or stays quadratic (an INTEGER model variable).

Only binary expansion is implemented; there is no encoding option
(3b §29). Dependencies: ``annealbridge.models`` and
``annealbridge.validation.estimates`` only.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from annealbridge.models import IntegerEncoding, OptimizationProblem, QuadraticTerm
from annealbridge.validation.estimates import compute_slack_coefficients

__all__ = [
    "AffineForm",
    "encode_integer_variables",
    "expand_product",
    "substitute_linear",
    "substitute_quadratic",
]

Linear = dict[str, float]
# Keyed by an *ordered* pair: the orientation is the one first produced,
# and the reversed pair accumulates into the same entry. A tuple rather
# than a frozenset keeps the ``add_quadratic(u, v)`` call order
# deterministic under hash randomisation and lets a self-product that is
# kept quadratic (CQM INTEGER variables, step 4) read as ``(v, v)``.
Quadratic = dict[tuple[str, str], float]


@dataclass(frozen=True)
class AffineForm:
    """A business variable as ``constant + sum(coefficients[v] * v)``.

    ``coefficients`` is insertion-ordered (the bit order ``k = 0, 1, ...``
    for an integer variable), which is what keeps the compiled model
    deterministic.
    """

    constant: float
    coefficients: dict[str, float]


def encode_integer_variables(
    problem: OptimizationProblem,
) -> tuple[dict[str, AffineForm], dict[str, IntegerEncoding]]:
    """Return the affine form of every variable plus the integer encodings.

    Forms are keyed by variable name in ``problem.variables`` order; bits
    of one integer variable are ``__int_<name>_<k>`` for ascending ``k``.
    The second mapping holds only the integer variables (the
    ``CompiledProblem.integer_encodings`` content); it is empty for a
    binary-only problem, whose forms are all identities.
    """
    forms: dict[str, AffineForm] = {}
    encodings: dict[str, IntegerEncoding] = {}
    for variable in problem.variables:
        name = variable.name
        if variable.type == "binary":
            forms[name] = AffineForm(constant=0.0, coefficients={name: 1.0})
            continue
        lower, upper = variable.bounds()
        coefficients = compute_slack_coefficients(upper - lower)
        bits = [f"__int_{name}_{k}" for k in range(len(coefficients))]
        forms[name] = AffineForm(
            constant=float(lower),
            coefficients={bit: float(c) for bit, c in zip(bits, coefficients)},
        )
        encodings[name] = IntegerEncoding(
            variable=name, lower=lower, bits=bits, coefficients=coefficients
        )
    return forms, encodings


def _add_pair(quadratic: Quadratic, u: str, v: str, value: float) -> None:
    """Accumulate ``value`` on the pair ``{u, v}`` keeping its first orientation."""
    if (u, v) in quadratic:
        quadratic[(u, v)] += value
    elif (v, u) in quadratic:
        quadratic[(v, u)] += value
    else:
        quadratic[(u, v)] = value


def _fold_every(_: str) -> bool:
    return True


def expand_product(
    left: AffineForm,
    right: AffineForm,
    *,
    fold_square: Callable[[str], bool] = _fold_every,
    scale: float = 1.0,
) -> tuple[Linear, Quadratic, float]:
    """Expand ``scale * left * right`` into ``(linear, quadratic, constant)``.

    With ``left = a0 + sum(a_i v_i)`` and ``right = c0 + sum(c_j w_j)``:
    the constant is ``a0 * c0``, the linear part ``a0 * c_j`` on every
    ``w_j`` plus ``c0 * a_i`` on every ``v_i``, and each cross term
    ``a_i * c_j`` goes to the pair ``(v_i, w_j)``. When ``v_i == w_j`` the
    product is a square: ``fold_square(v)`` True (a binary variable,
    ``v * v == v``) sends it to the linear part, False keeps it as the
    quadratic entry ``(v, v)``. The default folds everything, which is
    right for the BQM path where every compiled variable is a bit.

    Contributions accumulate in first-appearance order; a pair and its
    reverse share one entry.
    """
    linear: Linear = {}
    quadratic: Quadratic = {}
    constant = scale * left.constant * right.constant

    if left.constant != 0.0:
        for w, c in right.coefficients.items():
            linear[w] = linear.get(w, 0.0) + scale * left.constant * c
    if right.constant != 0.0:
        for v, a in left.coefficients.items():
            linear[v] = linear.get(v, 0.0) + scale * right.constant * a
    for v, a in left.coefficients.items():
        for w, c in right.coefficients.items():
            value = scale * a * c
            if v == w and fold_square(v):
                linear[v] = linear.get(v, 0.0) + value
            else:
                _add_pair(quadratic, v, w, value)
    return linear, quadratic, constant


def substitute_linear(
    coefficients: Mapping[str, float], forms: Mapping[str, AffineForm]
) -> tuple[Linear, float]:
    """Rewrite ``sum(c_v * v)`` over business variables as ``(bit_coefficients, constant)``.

    Each ``c_v`` multiplies the variable's form; compiled-variable
    coefficients accumulate in first-appearance order and the constants
    (``c_v * lower``) sum into the returned constant. For identity forms
    the result equals the input (``c * 1.0``) with constant ``0.0``.
    """
    bit_coefficients: Linear = {}
    constant = 0.0
    for name, value in coefficients.items():
        form = forms[name]
        constant += value * form.constant
        for bit, c in form.coefficients.items():
            bit_coefficients[bit] = bit_coefficients.get(bit, 0.0) + value * c
    return bit_coefficients, constant


def substitute_quadratic(
    terms: Iterable[QuadraticTerm], forms: Mapping[str, AffineForm]
) -> tuple[Linear, Quadratic, float]:
    """Rewrite ``sum(c * u * v)`` over business variables into bit terms.

    Every term goes through :func:`expand_product` with squares folded
    (bits satisfy ``b * b == b``), so ``x * x`` of an integer variable
    yields the linear part ``sum(a_k^2 b_k)`` plus pairwise bit products,
    and ``x * y`` yields the full bit-by-bit product. Results accumulate
    across terms in first-appearance order.
    """
    linear: Linear = {}
    quadratic: Quadratic = {}
    constant = 0.0
    for term in terms:
        term_linear, term_quadratic, term_constant = expand_product(
            forms[term.variable1], forms[term.variable2], scale=term.coefficient
        )
        constant += term_constant
        for bit, value in term_linear.items():
            linear[bit] = linear.get(bit, 0.0) + value
        for (u, v), value in term_quadratic.items():
            _add_pair(quadratic, u, v, value)
    return linear, quadratic, constant
