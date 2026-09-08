"""Objective → model translation shared by the compilers (3a spec §14, §15; 3b §15.1).

Both the BQM compiler (which then adds penalty terms) and the CQM compiler
(which sets this as the CQM objective) need the same energy form of the
objective, so it lives in one place and the two cannot drift apart:
:func:`build_objective_bqm` for the bit-level BQM path and
:func:`build_objective_qm` for the CQM path, where integer variables stay
INTEGER model variables with their bounds.
"""

from collections.abc import Iterable, Mapping

import dimod

from annealbridge.compiler.integer_encoding import (
    AffineForm,
    expand_product,
    substitute_linear,
)
from annealbridge.models import Objective, Variable

__all__ = ["add_model_variable", "build_objective_bqm", "build_objective_qm"]


def add_model_variable(
    model: dimod.QuadraticModel | dimod.ConstrainedQuadraticModel, variable: Variable
) -> None:
    """Declare ``variable`` in a quadratic or constrained quadratic model.

    A binary variable becomes a ``BINARY`` model variable, an integer one an
    ``INTEGER`` model variable carrying its declared bounds. The CQM itself,
    its objective and every constraint lhs must agree on vartype and bounds
    (dimod rejects a conflict), so all three declare through this one
    function. Declaring an already present variable is a no-op for dimod.
    """
    if variable.type == "binary":
        model.add_variable("BINARY", variable.name)
        return
    lower, upper = variable.bounds()
    model.add_variable("INTEGER", variable.name, lower_bound=lower, upper_bound=upper)


def build_objective_bqm(
    objective: Objective, forms: Mapping[str, AffineForm] | None = None
) -> dimod.BinaryQuadraticModel:
    """Return ``objective`` as a binary quadratic model to be *minimised*.

    Maximization objectives are converted to minimization energy by negating
    every objective coefficient including the constant, so that
    ``energy == sign * objective`` holds exactly. Terms are added with dimod
    ``add_*`` semantics (accumulate, never overwrite) in the order they
    appear, so the result is bit-for-bit identical to the Phase 1/2 BQM
    compiler. Only variables mentioned by the objective are present; the
    caller adds the rest.

    ``forms`` (3b §14.2, from ``encode_integer_variables``) rewrites every
    business variable into its affine form over compiled variables: a
    linear term goes through ``substitute_linear``, a quadratic term
    through ``expand_product`` with bit squares folded (``b * b == b``).
    ``None`` treats every variable as its own identity form, which is the
    binary-only behaviour and leaves single-argument callers unchanged; an
    identity form only multiplies by ``1.0`` and adds ``0.0``, so an
    all-binary objective compiles to the very same floats either way.
    """
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    sign = -1.0 if objective.direction == "maximize" else 1.0

    def form_of(name: str) -> AffineForm:
        if forms is None:
            return AffineForm(constant=0.0, coefficients={name: 1.0})
        return forms[name]

    for term in objective.linear_terms:
        coefficients, constant = substitute_linear(
            {term.variable: sign * term.coefficient}, {term.variable: form_of(term.variable)}
        )
        for variable, value in coefficients.items():
            bqm.add_linear(variable, value)
        bqm.offset += constant
    for term in objective.quadratic_terms:
        linear, quadratic, constant = expand_product(
            form_of(term.variable1),
            form_of(term.variable2),
            scale=sign * term.coefficient,
        )
        for variable, value in linear.items():
            bqm.add_linear(variable, value)
        for (u, v), value in quadratic.items():
            bqm.add_quadratic(u, v, value)
        bqm.offset += constant
    bqm.offset += sign * objective.constant
    return bqm


def build_objective_qm(
    objective: Objective, variables: Iterable[Variable]
) -> dimod.QuadraticModel:
    """Return ``objective`` as a quadratic model to be *minimised* (3b §15.1).

    The CQM-path twin of :func:`build_objective_bqm`: the same ``sign``
    rule (maximize negates every coefficient and the constant), the same
    accumulating ``add_*`` semantics and the same first-appearance order,
    but every variable is declared with its own vartype and bounds through
    :func:`add_model_variable`, so an integer variable is an ``INTEGER``
    model variable rather than a set of bits. ``variables`` is the
    problem's declaration list (bounds alone cannot tell an integer
    variable in ``[0, 1]`` from a binary one, and the QM must agree with
    the CQM's vartype); only the variables the objective mentions are
    present in the result (as in the BQM version) and the caller adds the
    rest to the CQM, which is what keeps an all-binary objective identical
    to 3a.

    A quadratic term goes through :func:`expand_product` with identity
    forms: ``x * x`` of an integer variable stays a quadratic ``(x, x)``
    entry (allowed for INTEGER), while a binary self-product folds into the
    linear part (a QM rejects ``b * b``). The validator rejects the latter
    anyway; folding keeps the rule identical to the BQM path.
    """
    declared = {variable.name: variable for variable in variables}
    qm = dimod.QuadraticModel()
    sign = -1.0 if objective.direction == "maximize" else 1.0

    def declare(name: str) -> None:
        if name not in qm.variables:
            add_model_variable(qm, declared[name])

    def is_binary(name: str) -> bool:
        return declared[name].type == "binary"

    for term in objective.linear_terms:
        declare(term.variable)
        qm.add_linear(term.variable, sign * term.coefficient)
    for term in objective.quadratic_terms:
        declare(term.variable1)
        declare(term.variable2)
        linear, quadratic, constant = expand_product(
            AffineForm(constant=0.0, coefficients={term.variable1: 1.0}),
            AffineForm(constant=0.0, coefficients={term.variable2: 1.0}),
            fold_square=is_binary,
            scale=sign * term.coefficient,
        )
        for variable, value in linear.items():
            qm.add_linear(variable, value)
        for (u, v), value in quadratic.items():
            qm.add_quadratic(u, v, value)
        qm.offset += constant
    qm.offset += sign * objective.constant
    return qm
