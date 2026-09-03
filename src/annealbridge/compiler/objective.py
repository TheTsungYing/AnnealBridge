"""Objective → BQM translation shared by the compilers (3a spec §14, §15).

Both the BQM compiler (which then adds penalty terms) and the CQM compiler
(which sets this as the CQM objective) need the same energy form of the
objective, so it lives in one place and the two cannot drift apart.
"""

import dimod

from annealbridge.models import Objective


def build_objective_bqm(objective: Objective) -> dimod.BinaryQuadraticModel:
    """Return ``objective`` as a binary quadratic model to be *minimised*.

    Maximization objectives are converted to minimization energy by negating
    every objective coefficient including the constant, so that
    ``energy == sign * objective`` holds exactly. Terms are added with dimod
    ``add_*`` semantics (accumulate, never overwrite) in the order they
    appear, so the result is bit-for-bit identical to the Phase 1/2 BQM
    compiler. Only variables mentioned by the objective are present; the
    caller adds the rest.
    """
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    sign = -1.0 if objective.direction == "maximize" else 1.0
    for term in objective.linear_terms:
        bqm.add_linear(term.variable, sign * term.coefficient)
    for term in objective.quadratic_terms:
        bqm.add_quadratic(term.variable1, term.variable2, sign * term.coefficient)
    bqm.offset += sign * objective.constant
    return bqm
