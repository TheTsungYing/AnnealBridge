"""Objective → BQM translation shared by the compilers (3a spec §14, §15).

Both the BQM compiler (which then adds penalty terms) and the CQM compiler
(which sets this as the CQM objective) need the same energy form of the
objective, so it lives in one place and the two cannot drift apart.
"""

from collections.abc import Mapping

import dimod

from annealbridge.compiler.integer_encoding import (
    AffineForm,
    expand_product,
    substitute_linear,
)
from annealbridge.models import Objective


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
