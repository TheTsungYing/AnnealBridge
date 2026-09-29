"""The two small grammars of schema 1.3 templates (schema 1.3 spec §14.6).

A template string is one of:

* a ``for_each`` item, ``index in set``;
* a reference, ``name`` or ``name[index, ...]`` where an index is a bound
  index name optionally shifted by a whole number (``p+1``, ``d-2``);
* a ``where`` condition, ``operand CMP operand`` where an operand is a bound
  index name, a parameter reference without shifts, or a number.

That is the whole language: there is no arithmetic, no function call and no
evaluation of any kind. Parsing is a hand-written scanner over explicit
ASCII character classes (never ``\\d``, ``\\w``, ``\\s``, ``isidentifier`` or
``isdigit``, which accept Unicode digits, letters and spaces), run once per
string before any binding is iterated; the expander only walks the parsed
structures this module returns. A malformed string raises
:class:`GrammarError`, whose message names the character position and shows
the offending character JSON-escaped, never more of the input.
"""

import json
import math
import re
from dataclasses import dataclass

__all__ = [
    "Condition",
    "ForEachItem",
    "GrammarError",
    "IDENT_LIMIT",
    "Index",
    "NUMBER",
    "Operand",
    "RESERVED",
    "Reference",
    "SHIFT_DIGITS",
    "is_identifier",
    "parse_condition",
    "parse_for_each",
    "parse_reference",
]

# Longest identifier (index set, parameter, family, template id, index name).
IDENT_LIMIT = 64
# Longest shift: nine digits, so int() never meets Python's digit limit.
SHIFT_DIGITS = 9

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIGITS = re.compile(r"[0-9]+")
NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_CMP = ("==", "!=", "<=", ">=", "<", ">")  # longest first
# Characters that may not directly follow a number (it would be malformed).
_AFTER_NUMBER = frozenset(
    "._0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

# Reserved: the keyword of a for_each item.
RESERVED = frozenset({"in"})


class GrammarError(ValueError):
    """A template string does not follow the grammar; the message says why."""


def is_identifier(text: str) -> bool:
    """Whether ``text`` is an ASCII identifier of at most 64 characters."""
    return len(text) <= IDENT_LIMIT and _IDENT.fullmatch(text) is not None


@dataclass(frozen=True)
class ForEachItem:
    """``index in set``."""

    index: str
    set_name: str


@dataclass(frozen=True)
class Index:
    """An index name inside brackets, shifted by ``shift`` positions."""

    name: str
    shift: int = 0


@dataclass(frozen=True)
class Reference:
    """``head`` alone (``indices is None``) or ``head[index, ...]``."""

    head: str
    indices: tuple[Index, ...] | None


@dataclass(frozen=True)
class Operand:
    """One side of a condition.

    ``kind`` is ``"index"`` (a bare index name in ``name``), ``"parameter"``
    (``name`` with the unshifted index names ``indices``) or ``"number"``
    (``value``).
    """

    kind: str
    name: str = ""
    indices: tuple[str, ...] = ()
    value: float = 0.0


@dataclass(frozen=True)
class Condition:
    left: Operand
    operator: str
    right: Operand


class _Scanner:
    """Position-tracking reader over one template string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def fail(self, expected: str) -> GrammarError:
        if self.pos >= len(self.text):
            found = "the end of the text"
        else:
            found = json.dumps(self.text[self.pos])
        return GrammarError(
            f"expected {expected} at character {self.pos + 1}, found {found}"
        )

    def spaces(self) -> int:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1
        return self.pos - start

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def take(self, literal: str) -> bool:
        if self.text.startswith(literal, self.pos):
            self.pos += len(literal)
            return True
        return False

    def identifier(self, what: str) -> str:
        match = _IDENT.match(self.text, self.pos)
        if match is None:
            raise self.fail(what)
        name = match.group()
        if len(name) > IDENT_LIMIT:
            raise GrammarError(
                f"the name at character {self.pos + 1} is longer than "
                f"{IDENT_LIMIT} characters"
            )
        self.pos = match.end()
        return name

    def end(self) -> None:
        self.spaces()
        if not self.at_end():
            raise self.fail("the end of the text")


def parse_for_each(text: str) -> ForEachItem:
    """``index in set``: two identifiers around the word ``in``, spaced."""
    scanner = _Scanner(text)
    scanner.spaces()
    index_at = scanner.pos
    index = scanner.identifier('an index name (a for_each item is "index in set")')
    if scanner.spaces() == 0:
        raise scanner.fail('a space and "in" (a for_each item is "index in set")')
    keyword = _IDENT.match(text, scanner.pos)
    if keyword is None or keyword.group() != "in":
        raise scanner.fail('"in" (a for_each item is "index in set")')
    scanner.pos = keyword.end()
    if scanner.spaces() == 0:
        raise scanner.fail('a space and an index set name after "in"')
    set_at = scanner.pos
    set_name = scanner.identifier("an index set name")
    scanner.end()
    if index in RESERVED:
        raise GrammarError(
            f'the index name at character {index_at + 1} is "in", which is '
            "reserved; choose another name"
        )
    if set_name in RESERVED:
        raise GrammarError(
            f'the index set name at character {set_at + 1} is "in", which is '
            "reserved and cannot name an index set"
        )
    return ForEachItem(index=index, set_name=set_name)


def _index(scanner: _Scanner) -> Index:
    scanner.spaces()
    name = scanner.identifier(
        "an index name (literal elements such as a or 0 are not supported; "
        "bind an index with for_each)"
    )
    scanner.spaces()
    sign = scanner.peek()
    if sign not in ("+", "-"):
        return Index(name=name)
    scanner.pos += 1
    scanner.spaces()
    match = _DIGITS.match(scanner.text, scanner.pos)
    if match is None:
        raise scanner.fail(f"a whole number after {json.dumps(sign)}")
    digits = match.group()
    if digits[0] == "0":
        raise GrammarError(
            f"the shift at character {scanner.pos + 1} must be a positive whole "
            "number without leading zeros"
        )
    if len(digits) > SHIFT_DIGITS:
        raise GrammarError(
            f"the shift at character {scanner.pos + 1} has more than "
            f"{SHIFT_DIGITS} digits"
        )
    scanner.pos = match.end()
    shift = int(digits)
    return Index(name=name, shift=shift if sign == "+" else -shift)


def parse_reference(text: str) -> Reference:
    """``name`` or ``name[index, ...]``."""
    scanner = _Scanner(text)
    scanner.spaces()
    head = scanner.identifier("a family, parameter or variable name")
    scanner.spaces()
    if not scanner.take("["):
        scanner.end()
        return Reference(head=head, indices=None)
    indices = [_index(scanner)]
    while True:
        scanner.spaces()
        if scanner.take("]"):
            break
        if not scanner.take(","):
            raise scanner.fail('"," or "]"')
        indices.append(_index(scanner))
    scanner.end()
    return Reference(head=head, indices=tuple(indices))


def _operand(scanner: _Scanner) -> Operand:
    scanner.spaces()
    char = scanner.peek()
    if char == "-" or "0" <= char <= "9":
        match = NUMBER.match(scanner.text, scanner.pos)
        if match is None or scanner.text[match.end() : match.end() + 1] in _AFTER_NUMBER:
            raise scanner.fail("a number such as 3, -2 or 0.5")
        value = float(match.group())
        if not math.isfinite(value):
            raise GrammarError(
                f"the number at character {scanner.pos + 1} is not finite"
            )
        scanner.pos = match.end()
        return Operand(kind="number", value=value)
    name = scanner.identifier("an index name, a parameter reference or a number")
    scanner.spaces()
    if not scanner.take("["):
        return Operand(kind="index", name=name)
    names = []
    while True:
        index_at = scanner.pos
        index = _index(scanner)
        if index.shift:
            raise GrammarError(
                f"the index at character {index_at + 1} is shifted, but a where "
                "condition cannot shift an index; shifts are only allowed in the "
                "generated entry's own references"
            )
        names.append(index.name)
        scanner.spaces()
        if scanner.take("]"):
            break
        if not scanner.take(","):
            raise scanner.fail('"," or "]"')
    return Operand(kind="parameter", name=name, indices=tuple(names))


def parse_condition(text: str) -> Condition:
    """``operand CMP operand``."""
    scanner = _Scanner(text)
    left = _operand(scanner)
    scanner.spaces()
    for operator in _CMP:
        if scanner.take(operator):
            break
    else:
        if scanner.peek() == "=":
            raise GrammarError(
                f'use "==" to compare (character {scanner.pos + 1} is a single "=")'
            )
        raise scanner.fail("one of ==, !=, <, <=, >, >=")
    right = _operand(scanner)
    scanner.end()
    return Condition(left=left, operator=operator, right=right)
