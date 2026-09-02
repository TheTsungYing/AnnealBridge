"""Decision variable model."""

from typing import Literal

from pydantic import BaseModel


class Variable(BaseModel):
    """A binary decision variable in an optimization problem."""

    name: str
    type: Literal["binary"] = "binary"
    description: str | None = None
