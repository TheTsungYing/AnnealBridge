"""Compilers from the optimization IR to solver-specific models."""

from annealbridge.compiler.base import ModelCompiler
from annealbridge.compiler.bqm import BQMCompiler
from annealbridge.compiler.cqm import CQMCompiler
from annealbridge.compiler.integer_encoding import (
    AffineForm,
    encode_integer_variables,
    expand_product,
    expand_square,
    expand_square_qm,
    substitute_linear,
)
from annealbridge.compiler.objective import (
    add_model_variable,
    build_objective_bqm,
    build_objective_qm,
)
from annealbridge.compiler.slack import (
    InequalityEncoding,
    compute_slack_coefficients,
    encode_slack,
)

__all__ = [
    "AffineForm",
    "BQMCompiler",
    "CQMCompiler",
    "InequalityEncoding",
    "ModelCompiler",
    "add_model_variable",
    "build_objective_bqm",
    "build_objective_qm",
    "compute_slack_coefficients",
    "encode_integer_variables",
    "encode_slack",
    "expand_product",
    "expand_square",
    "expand_square_qm",
    "substitute_linear",
]
