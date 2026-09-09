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
business variables into expressions over compiled variables. Two
expansion cores live here (2026-09-09 review F-13a / F-13e):

* :func:`expand_product` expands the product of two affine forms; the
  objective builders of ``compiler/objective.py`` use it for every
  quadratic term on both paths.
* :func:`expand_square` expands ``weight * (sum(a_v * v) + constant)**2``
  in the exact floating-point order the BQM penalty terms have always
  used (pinned bit for bit by the 3a golden test); the BQM compiler's
  penalty terms and the CQM path's :func:`expand_square_qm` (3b §15.3)
  are both thin adapters over it, so the two paths cannot drift.

The only difference between the paths is whether a self-product ``v * v``
folds into a linear term (a bit, since ``b * b == b``) or stays quadratic
(an INTEGER model variable); both cores take that as ``fold_square``.

Only binary expansion is implemented; there is no encoding option
(3b §29). Dependencies: ``annealbridge.models`` and
``annealbridge.validation.estimates`` only.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import dimod

from annealbridge.models import IntegerEncoding, OptimizationProblem
from annealbridge.validation.estimates import compute_slack_coefficients

__all__ = [
    "AffineForm",
    "encode_integer_variables",
    "expand_product",
    "expand_square",
    "expand_square_qm",
    "substitute_linear",
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


def expand_square(
    coefficients: Mapping[str, float],
    constant: float,
    weight: float,
    *,
    fold_square: Callable[[str], bool] = _fold_every,
) -> tuple[Linear, Quadratic, float]:
    """Expand ``weight * (sum(coefficients[v] * v) + constant)**2``.

    The one squared-penalty expansion of both compilers (2026-09-09 review
    F-13a). Per variable ``v`` with coefficient ``a``: when ``fold_square(v)``
    (a bit, ``v * v == v``) the whole diagonal goes to the linear part as
    ``weight * (a * a + 2 * constant * a)``; otherwise the linear part is
    ``weight * (2 * constant * a)`` and the square stays the quadratic
    entry ``(v, v) = weight * (a * a)``. Every pair ``u < v`` in
    ``coefficients`` order gets ``2 * weight * a_u * a_v`` exactly once and
    the constant is ``weight * constant * constant``.

    The arithmetic order is the BQM penalty term's original one (the 3a
    golden test pins those floats bit for bit), so it is deliberately not
    ``expand_product(form, form)``: that doubles each cross term as two
    half-products and rounds differently on non-dyadic data. The linear
    entries come first, in ``coefficients`` order, then the pairs.
    """
    linear: Linear = {}
    quadratic: Quadratic = {}
    items = list(coefficients.items())
    for variable, value in items:
        if fold_square(variable):
            linear[variable] = weight * (value * value + 2.0 * constant * value)
        else:
            linear[variable] = weight * (2.0 * constant * value)
            quadratic[(variable, variable)] = weight * (value * value)
    for i, (var_i, value_i) in enumerate(items):
        for var_j, value_j in items[i + 1 :]:
            quadratic[(var_i, var_j)] = 2.0 * weight * value_i * value_j
    return linear, quadratic, weight * constant * constant


def expand_square_qm(
    qm: dimod.QuadraticModel,
    coefficients: Mapping[str, float],
    constant: float,
    weight: float,
) -> None:
    """Add ``weight * (sum(coefficients[v] * v) + constant)**2`` to ``qm`` (3b §15.3).

    The CQM path's squared soft penalty in objective form: a thin adapter
    over :func:`expand_square`, the very same expansion the BQM penalty
    terms use. Here a self-product stays the quadratic entry ``(v, v)`` for
    an ``INTEGER`` model variable and folds into the linear part for a
    ``BINARY`` one (``b * b == b``; a QM rejects a binary self-interaction).
    Contributions accumulate with dimod ``add_*`` semantics.

    Every variable in ``coefficients`` must already be declared in ``qm``
    (with its vartype and bounds); ``qm.vartype`` raises for an unknown one.
    """
    linear, quadratic, offset = expand_square(
        coefficients,
        constant,
        weight,
        fold_square=lambda name: qm.vartype(name) is dimod.BINARY,
    )
    for variable, value in linear.items():
        qm.add_linear(variable, value)
    for (u, v), value in quadratic.items():
        qm.add_quadratic(u, v, value)
    qm.offset += offset
