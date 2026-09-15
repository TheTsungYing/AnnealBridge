"""Model-term builders shared by the integer and soft-constraint unit tests.

Each function is a verbatim move of the helper those tests used to define
locally. The dict payload builders of the same names live in
``tests/builders.py``.
"""

from annealbridge.models import LinearTerm, QuadraticTerm


def lin(variable: str, coefficient: float) -> LinearTerm:
    return LinearTerm(variable=variable, coefficient=coefficient)


def quad(variable1: str, variable2: str, coefficient: float) -> QuadraticTerm:
    return QuadraticTerm(
        variable1=variable1, variable2=variable2, coefficient=coefficient
    )
