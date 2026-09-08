"""Decision variable model (IR 1.0 binary, IR 1.1 bounded integer)."""

from typing import Literal

from pydantic import BaseModel, field_validator


class Variable(BaseModel):
    """A binary or bounded-integer decision variable in an optimization problem.

    ``lower_bound`` / ``upper_bound`` are required for ``integer`` variables
    and must be absent for ``binary`` ones (3b spec §7); the semantic rules
    (both present, ``upper > lower``, within ±(2^31-1), schema version 1.1)
    are the problem validator's job so they surface as ``invalid_problem``
    with a recommended action, not as a bare type error. Integer encoding is
    the compiler's concern: the IR carries no bits and no encoding choice.
    """

    name: str
    type: Literal["binary", "integer"] = "binary"
    lower_bound: int | None = None
    upper_bound: int | None = None
    description: str | None = None

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def _reject_bool(cls, value: object) -> object:
        # pydantic's lax mode would coerce True -> 1; a bound is a quantity,
        # not a flag, so a boolean is a mistake worth rejecting at the schema.
        if isinstance(value, bool):
            raise ValueError("a bound must be an integer, not a boolean")
        return value

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
