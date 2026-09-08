"""Compilers from the optimization IR to solver-specific models."""

from annealbridge.compiler.base import ModelCompiler
from annealbridge.compiler.bqm import BQMCompiler
from annealbridge.compiler.cqm import CQMCompiler
from annealbridge.compiler.integer_encoding import (
    AffineForm,
    encode_integer_variables,
    expand_product,
    substitute_linear,
    substitute_quadratic,
)
from annealbridge.compiler.objective import build_objective_bqm
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
    "build_objective_bqm",
    "compute_slack_coefficients",
    "encode_integer_variables",
    "encode_slack",
    "expand_product",
    "substitute_linear",
    "substitute_quadratic",
]
