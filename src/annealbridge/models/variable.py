"""Decision variable model (IR 1.0 binary, IR 1.1 bounded integer)."""

from typing import Literal

from pydantic import BaseModel

from annealbridge.models.quantities import Count


class Variable(BaseModel):
    """A binary or bounded-integer decision variable in an optimization problem.

    ``lower_bound`` / ``upper_bound`` are required for ``integer`` variables
    and must be absent for ``binary`` ones (3b spec §7); the semantic rules
    (both present, ``upper > lower``, within ±(2^31-1), schema version 1.1)
    are the problem validator's job so they surface as ``invalid_problem``
    with a recommended action, not as a bare type error. Integer encoding is
    the compiler's concern: the IR carries no bits and no encoding choice.

    The bounds are ``Count`` (models/quantities.py), the shared IR integer
    type: a boolean or a string is refused rather than coerced, while an
    integral float (``2.0``) is still accepted (2026-09-09 review F-11).
    """

    name: str
    type: Literal["binary", "integer"] = "binary"
    lower_bound: Count | None = None
    upper_bound: Count | None = None
    description: str | None = None

    def bounds(self) -> tuple[int, int]:
        """Return ``(lower, upper)``: ``(0, 1)`` for binary, the declared
        bounds for integer.

        Only for the validator's warning layer, the estimates and the
        compilers, all of which run *after* the validator's error pass has
        established the bounds are legal (3b §7); raises ``ValueError`` for
        an integer variable missing a bound. The error pass itself never
        calls this.
        """
        if self.type == "binary":
            return (0, 1)
        if self.lower_bound is None or self.upper_bound is None:
            raise ValueError(
                f"integer variable {self.name} has no complete bounds; "
                "validate_problem must reject it before bounds() is used"
            )
        return (self.lower_bound, self.upper_bound)
