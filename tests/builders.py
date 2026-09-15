"""Dict payload builders shared by the compiler and validator unit tests.

Each function is a verbatim move of the helper those tests used to define
locally. ``tests/golden/record_phase3a_golden.py`` keeps its own independent
copies on purpose; do not make it import from here.
"""


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def quad(variable1: str, variable2: str, coefficient: float) -> dict:
    return {"variable1": variable1, "variable2": variable2, "coefficient": coefficient}


def hard(constraint_id: str, operator: str, rhs: float, terms: list[dict]) -> dict:
    return {
        "id": constraint_id,
        "type": "hard",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
    }


def soft(
    constraint_id: str, operator: str, rhs: float, terms: list[dict], weight: float
) -> dict:
    return {
        "id": constraint_id,
        "type": "soft",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
        "weight": weight,
    }
