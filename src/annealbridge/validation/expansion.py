"""Expanding schema 1.3 templates into an ordinary problem (spec §14.7-§14.11).

:func:`expand_problem` is the first step of every validation of a problem
that carries index sets, parameters, variable families or templates. It
returns an :class:`ExpandedProblem`: either the expanded problem -- the
explicit entries first, then every generated one, all template fields empty
and ``version`` unchanged -- or the errors that stopped it, never both and
never a partial result. It never raises: every problem with the document is
a ``SolveError`` with a path into the template that caused it.

The expansion runs in three phases:

1. **Static checks** of every name, index set, parameter row, family and
   template string. Each string is parsed once (``template_grammar``) into
   the structures the later phases walk; nothing is parsed per binding.
2. **The upper bound U** on the bindings the templates iterate, computed
   without iterating and refused above ``max_template_bindings`` before any
   entry is generated (spec §14.9).
3. **Generation**, in the order spec §14.7 fixes, counting the work W
   (every generated entry plus, per generated constraint, one unit per pair
   of its terms or members) against the same ceiling and stopping as soon
   as it is crossed.

Afterwards generated variables that no objective term or constraint uses are
left out (spec §14.14 item 2, the user's choice B). The origins kept here
map every path the ordinary validator reports on the expanded problem back
to the template that generated it (:meth:`ExpandedProblem.locate`).

Depends on ``annealbridge.models``, ``annealbridge.exceptions`` (through the
models), ``validation.issues``, ``validation.template_grammar`` and the
standard library only (``tests/architecture``).
"""

import bisect
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from annealbridge.models import (
    CardinalityConstraint,
    Constraint,
    LinearTerm,
    OptimizationProblem,
    QuadraticTerm,
    SolveError,
    Variable,
)
from annealbridge.validation.issues import (
    INTEGER_BOUND_LIMIT,
    _check_variable_bounds,
    _error,
    _warning,
    int_text,
)
from annealbridge.validation.template_grammar import (
    NUMBER,
    RESERVED,
    GrammarError,
    Index,
    Reference,
    is_identifier,
    parse_condition,
    parse_for_each,
    parse_reference,
)

__all__ = [
    "DEFAULT_MAX_TEMPLATE_BINDINGS",
    "ExpandedProblem",
    "Location",
    "expand_problem",
]

# The policy default (orchestration/policy.py), pinned equal by a test; the
# value a library caller of validate_problem(_full) gets (spec §14.9).
DEFAULT_MAX_TEMPLATE_BINDINGS = 250_000

# Static ceilings (spec §14.5-§14.6): together with the 256-character
# template strings they bound the work of one binding by a constant.
MAX_ENTRIES = 8  # for_each items, where conditions, index sets of a family or parameter
ELEMENT_LIMIT = 64
# Error-list ceilings (spec §14.11): per source, and for the whole expansion.
ERRORS_PER_SOURCE = 20
ERRORS_TOTAL = 100
# How many left-out variable names an UNUSED_TEMPLATE_VARIABLES warning shows.
UNUSED_NAMES_SHOWN = 5

_ELEMENT = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
_NAME_RULE = (
    "an identifier: letters, digits and underscores, not starting with a "
    "digit, at most 64 characters"
)
_LITERAL_HINT = (
    "literal elements are not supported: bind an index with for_each and "
    "compare a 0/1 parameter such as is_first[p] == 1 instead"
)


def _show(value: object) -> str:
    """A caller-supplied value for a message, bounded and escaped."""
    if isinstance(value, int):
        text = int_text(value)
        return text if len(text) <= ELEMENT_LIMIT else f"<an integer of {len(text)} digits>"
    text = str(value)
    if len(text) > ELEMENT_LIMIT:
        return json.dumps(text[:ELEMENT_LIMIT]) + "..."
    return json.dumps(text)


def _label(name: str) -> str:
    """A declared name for a message: as written when it is an identifier
    (so of bounded length and plain characters), bounded and escaped
    otherwise."""
    return name if is_identifier(name) else _show(name)


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# ---------------------------------------------------------------------------
# Compiled structures
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Set:
    name: str
    elements: list
    texts: list[str]
    positions: dict
    order: str

    @property
    def size(self) -> int:
        return len(self.elements)


@dataclass(eq=False)
class _Param:
    index: int
    name: str
    sets: tuple[_Set, ...]
    table: dict[tuple[int, ...], float]
    default: float | None


@dataclass(eq=False)
class _Family:
    index: int
    name: str
    sets: tuple[_Set, ...]
    type: str
    lower: int | None
    upper: int | None
    description: str | None


@dataclass(frozen=True, eq=False)
class _Pos:
    """An index position of a reference: binding slot plus shift."""

    slot: int
    shift: int
    set: _Set


@dataclass(frozen=True, eq=False)
class _VarRef:
    family: _Family | None
    name: str | None  # an explicit variable
    positions: tuple[_Pos, ...] = ()


@dataclass(frozen=True, eq=False)
class _ValRef:
    path: str
    literal: float | None = None
    param: _Param | None = None
    positions: tuple[_Pos, ...] = ()


@dataclass(frozen=True, eq=False)
class _Operand:
    kind: str  # "index" or "value"
    slot: int = -1
    value: _ValRef | None = None


@dataclass(frozen=True, eq=False)
class _Cond:
    path: str
    operator: str
    left: _Operand
    right: _Operand


@dataclass(eq=False)
class _Scope:
    loops: tuple[tuple[int, _Set], ...]
    conds: tuple[_Cond, ...]


@dataclass(eq=False)
class _TermTemplate:
    source: str
    kind: str  # "linear" or "quadratic"
    scope: _Scope
    coefficient: _ValRef
    variables: tuple[_VarRef, ...]
    generated: int = 0
    skipped: int = 0


@dataclass(eq=False)
class _Member:
    scope: _Scope
    variable: _VarRef
    coefficient: _ValRef | None


@dataclass(eq=False)
class _ConstraintTemplate:
    source: str
    kind: str  # "linear" or "cardinality"
    id: str
    description: str | None
    type: str
    operator: str
    scope: _Scope
    loop_names: tuple[str, ...]
    rhs: _ValRef
    weight: _ValRef | None
    members: list[_Member]
    generated: int = 0
    skipped: int = 0
    merged: int = 0
    merged_example: str = ""


class _LimitReached(Exception):
    """Internal: the work W crossed the ceiling while generating ``source``."""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.source = source


_MISSING = object()


# ---------------------------------------------------------------------------
# Errors with ceilings
# ---------------------------------------------------------------------------


class _ErrorLog:
    """Expansion errors, at most 20 per source and 100 in total (spec §14.11).

    A source is one top-level entry (``index_sets[0]``, ``constraint_templates[2]``,
    ``version`` ...). Errors past a ceiling are only counted, per code, and
    the count is written onto the last listed error of that source. An
    error identical to one already listed is not listed again; the count
    of those past the ceiling may include repeats.
    """

    def __init__(self) -> None:
        self._kept: list[tuple[str, SolveError]] = []
        self._per_source: Counter[str] = Counter()
        self._omitted: dict[str, Counter[str]] = {}
        # Identities of the kept errors only: at most ERRORS_TOTAL entries.
        self._seen: set[tuple[str, str | None, str]] = set()

    def accepts(self, source: str) -> bool:
        """Whether an error from ``source`` would still be listed; when not,
        the caller may :meth:`count` it without building it."""
        return len(self._kept) < ERRORS_TOTAL and self._per_source[source] < ERRORS_PER_SOURCE

    def count(self, source: str, code: str) -> None:
        """Record an error past the ceilings by its code alone."""
        self._omitted.setdefault(source, Counter())[code] += 1

    def add(self, source: str, error: SolveError) -> None:
        if not self.accepts(source):
            self.count(source, error.code)
            return
        # The same missing key used by several bindings is one mistake, not
        # several: a listed error identical in code, path and message is not
        # listed again.
        identity = (error.code, error.path, error.message)
        if identity in self._seen:
            return
        self._seen.add(identity)
        self._kept.append((source, error))
        self._per_source[source] += 1

    def __bool__(self) -> bool:
        return bool(self._kept)

    def finish(self) -> tuple[SolveError, ...]:
        errors = [error for _, error in self._kept]
        for source, counts in self._omitted.items():
            listed = [i for i, (kept, _) in enumerate(self._kept) if kept == source]
            at = listed[-1] if listed else len(errors) - 1
            total = sum(counts.values())
            detail = ", ".join(f"{code} ×{count}" for code, count in counts.items())
            suffix = (
                f"; {total} more error{'s' if total != 1 else ''} from {source} "
                f"{'are' if total != 1 else 'is'} not listed ({detail})"
            )
            errors[at] = errors[at].model_copy(
                update={"message": errors[at].message + suffix}
            )
        return tuple(errors)


# ---------------------------------------------------------------------------
# Origins and the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    source: str
    loop_names: tuple[str, ...] = ()
    template_id: str = ""


@dataclass(frozen=True)
class _ListOrigin:
    """Which template generated which position of one expanded list."""

    explicit: int
    spans: tuple[_Span, ...]
    starts: tuple[int, ...]
    # For constraint lists: per generated constraint (index - explicit), the
    # number of terms / members each member template produced.
    members: tuple[tuple[int, ...], ...] = ()

    def find(self, position: int) -> _Span | None:
        if position < self.explicit:
            return None
        at = bisect.bisect_right(self.starts, position) - 1
        if at < 0:
            return None
        span = self.spans[at]
        return span if span.start <= position < span.end else None


def _list_origin(explicit: int, spans: list[_Span], members=()) -> _ListOrigin:
    spans = [span for span in spans if span.end > span.start]
    return _ListOrigin(
        explicit=explicit,
        spans=tuple(spans),
        starts=tuple(span.start for span in spans),
        members=tuple(members),
    )


@dataclass(frozen=True)
class _Origins:
    variables: _ListOrigin
    linear_terms: _ListOrigin
    quadratic_terms: _ListOrigin
    linear_constraints: _ListOrigin
    cardinality: _ListOrigin
    family_names: tuple[str, ...]
    generated_variables: int
    left_out_variables: int


@dataclass(frozen=True)
class Location:
    """Where a path of the expanded problem was written in the document."""

    source: str  # the top-level template, e.g. "constraint_templates[1]"
    path: str  # the mapped path, e.g. "constraint_templates[1].terms[0]"
    suffix: str  # " (generated by constraint_templates[1] with i=a)"


_ITEM_PATH = re.compile(
    r"(variables|objective\.linear_terms|objective\.quadratic_terms|constraints"
    r"|cardinality_constraints)\[(\d+)\](.*)",
    re.DOTALL,
)
_SUB_ITEM = re.compile(r"\.(terms|variables)\[(\d+)\](.*)", re.DOTALL)


@dataclass(frozen=True)
class ExpandedProblem:
    """The outcome of :func:`expand_problem` (spec §14.8).

    ``problem`` is the expanded problem, or ``None`` when ``errors`` is not
    empty; for a document without templates it is ``source`` itself and
    nothing else is set, so every later step behaves exactly as before.
    ``limit_exceeded`` is true when the one error is
    ``TEMPLATE_EXPANSION_LIMIT`` (reported as ``resource_limit_exceeded``
    by a solve). ``warnings`` are the expansion's own advisories.
    """

    source: OptimizationProblem
    problem: OptimizationProblem | None
    errors: tuple[SolveError, ...] = ()
    warnings: tuple[SolveError, ...] = ()
    limit_exceeded: bool = False
    _origins: _Origins | None = field(default=None, repr=False, compare=False)

    @property
    def templated(self) -> bool:
        """Whether the source carried any template field."""
        return self.source.has_templates()

    @property
    def family_names(self) -> tuple[str, ...]:
        return self._origins.family_names if self._origins else ()

    @property
    def all_generated_variables_left_out(self) -> int:
        """How many generated variables there were, when every one of them
        was left out as unused; 0 otherwise."""
        origins = self._origins
        if origins is None or origins.generated_variables == 0:
            return 0
        if origins.left_out_variables == origins.generated_variables:
            return origins.generated_variables
        return 0

    def objective_source(self, kind: str, position: int) -> str | None:
        """The template that generated objective term ``position``."""
        if self._origins is None:
            return None
        origin = (
            self._origins.linear_terms if kind == "linear" else self._origins.quadratic_terms
        )
        span = origin.find(position)
        return None if span is None else span.source

    def objective_explicit(self, kind: str) -> int:
        if self._origins is None:
            return 0
        origin = (
            self._origins.linear_terms if kind == "linear" else self._origins.quadratic_terms
        )
        return origin.explicit

    def locate(self, path: str | None) -> Location | None:
        """The template a validator path of the expanded problem points into.

        ``None`` for a path on an explicit entry, on the problem as a whole
        or outside the generated lists: such a path is already where the
        document wrote it.
        """
        if path is None or self._origins is None:
            return None
        match = _ITEM_PATH.fullmatch(path)
        if match is None:
            return None
        list_name, position, rest = match.group(1), int(match.group(2)), match.group(3)
        origins = self._origins
        origin = {
            "variables": origins.variables,
            "objective.linear_terms": origins.linear_terms,
            "objective.quadratic_terms": origins.quadratic_terms,
            "constraints": origins.linear_constraints,
            "cardinality_constraints": origins.cardinality,
        }[list_name]
        span = origin.find(position)
        if span is None:
            return None
        binding = ""
        if list_name in ("constraints", "cardinality_constraints"):
            sub = _SUB_ITEM.fullmatch(rest)
            if sub is not None:
                counts = origin.members[position - origin.explicit]
                j = int(sub.group(2))
                member = 0
                total = 0
                for member, count in enumerate(counts):
                    total += count
                    if j < total:
                        break
                rest = f".{sub.group(1)}[{member}]{sub.group(3)}"
            binding = self._binding(list_name, position, span)
        return Location(
            source=span.source,
            path=span.source + rest,
            suffix=f" (generated by {span.source}{binding})",
        )

    def _binding(self, list_name: str, position: int, span: _Span) -> str:
        """`` with i=a, p=0`` for a generated constraint, read off its id."""
        if not span.loop_names or self.problem is None:
            return ""
        items = (
            self.problem.constraints
            if list_name == "constraints"
            else self.problem.cardinality_constraints
        )
        generated_id = items[position].id
        inner = generated_id[len(span.template_id) + 1 : -1]
        texts = inner.split(",")
        if len(texts) != len(span.loop_names):
            return ""
        pairs = ", ".join(f"{name}={text}" for name, text in zip(span.loop_names, texts))
        return f" with {pairs}"


# ---------------------------------------------------------------------------
# The expander
# ---------------------------------------------------------------------------


def expand_problem(
    problem: OptimizationProblem,
    *,
    max_template_bindings: int = DEFAULT_MAX_TEMPLATE_BINDINGS,
) -> ExpandedProblem:
    """Expand ``problem``'s templates; never raises (spec §14.8).

    ``max_template_bindings`` is the ceiling on both U and W (spec §14.9);
    the service passes its policy's value, a library caller gets the
    default. A problem without template fields is returned untouched.
    """
    if not problem.has_templates():
        return ExpandedProblem(source=problem, problem=problem)
    return _Expander(problem, max_template_bindings).run()


class _Expander:
    def __init__(self, problem: OptimizationProblem, limit: int) -> None:
        self.problem = problem
        self.limit = limit
        self.log = _ErrorLog()
        self.sets: dict[str, _Set] = {}
        self.params: dict[str, _Param] = {}
        self.families: dict[str, _Family] = {}
        # Every declared symbol: name -> the path that declared it first.
        self.symbols: dict[str, str] = {}
        self.explicit_types = {variable.name: variable.type for variable in problem.variables}
        self.term_templates: list[_TermTemplate] = []
        self.constraint_templates: list[_ConstraintTemplate] = []
        self.rhs_params: dict[str, list[str]] = {}
        self.work = 0
        self.building = True
        self.slots: list[int] = [0] * (2 * MAX_ENTRIES)

    # -- driver -----------------------------------------------------------

    def run(self) -> ExpandedProblem:
        self._check_version()
        self._declare_sets()
        self._declare_parameters()
        self._declare_families()
        self._compile_objective_templates()
        self._compile_constraint_templates()
        self._check_rhs_parameters()
        if self.log:
            return self._failed()
        bound_error = self._check_bound()
        if bound_error is not None:
            return self._limit_failure(bound_error)
        try:
            generated = self._generate()
        except _LimitReached as reached:
            return self._limit_failure(
                _error(
                    code="TEMPLATE_EXPANSION_LIMIT",
                    path=reached.source,
                    message=(
                        f"Generating the templates reached {self.work} units of "
                        "work (one per generated entry plus, for each generated "
                        "constraint, one per pair of its terms or members), "
                        f"above this server's max_template_bindings of "
                        f"{self.limit}; it stopped while expanding {reached.source}"
                    ),
                )
            )
        if self.log:
            return self._failed()
        return self._assemble(generated)

    def _failed(self) -> ExpandedProblem:
        return ExpandedProblem(source=self.problem, problem=None, errors=self.log.finish())

    def _limit_failure(self, error: SolveError) -> ExpandedProblem:
        return ExpandedProblem(
            source=self.problem, problem=None, errors=(error,), limit_exceeded=True
        )

    # -- phase 1: static checks -------------------------------------------

    def _check_version(self) -> None:
        problem = self.problem
        if problem.version not in ("1.0", "1.1", "1.2"):
            return
        names = problem.template_fields()
        verb = "requires" if len(names) == 1 else "require"
        message = (
            f'{_join(names)} {verb} version "1.3" or later, but version is '
            f"{problem.version!r}"
        )
        notes: list[str] = []
        if "version" not in problem.model_fields_set:
            notes.append('version defaults to "1.0" when omitted')
        if problem.version in ("1.0", "1.1") and problem.cardinality_constraints:
            notes.append('"1.3" also allows cardinality_constraints')
        if problem.version == "1.0" and (
            any(variable.type == "integer" for variable in problem.variables)
            or any(family.type == "integer" for family in problem.variable_families)
        ):
            notes.append('"1.3" also allows the integer variables')
        if notes:
            message += "; " + "; ".join(notes)
        self.log.add(
            "version",
            _error(code="FEATURE_REQUIRES_NEWER_VERSION", path="version", message=message),
        )

    def _declare_name(self, name: str, path: str, what: str, source: str) -> bool:
        """Check a declared set / parameter / family name; True if usable."""
        if not is_identifier(name) or name in RESERVED:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=f"{path}.name",
                    message=f"The name of {what} {_show(name)} must be {_NAME_RULE}, "
                    'and not the word "in"',
                ),
            )
            return False
        if name in self.symbols:
            self.log.add(
                source,
                _error(
                    code="DUPLICATE_TEMPLATE_NAME",
                    path=f"{path}.name",
                    message=(
                        f"The name {name} is already declared at {self.symbols[name]}; "
                        "index sets, parameters and variable families need "
                        "distinct names"
                    ),
                ),
            )
            return False
        self.symbols[name] = path
        return True

    def _declare_sets(self) -> None:
        for i, index_set in enumerate(self.problem.index_sets):
            path = f"index_sets[{i}]"
            usable = self._declare_name(index_set.name, path, "index set", path)
            elements = index_set.elements
            ok = True
            if not elements:
                self.log.add(
                    path,
                    _error(
                        code="INDEX_SET_INVALID",
                        path=f"{path}.elements",
                        message=f"Index set {_show(index_set.name)} has no elements; "
                        "list at least one",
                    ),
                )
                ok = False
            kind = type(elements[0]) if elements else None
            seen: set = set()
            for j, element in enumerate(elements):
                element_path = f"{path}.elements[{j}]"
                if type(element) is not kind:
                    self.log.add(
                        path,
                        _error(
                            code="INDEX_SET_INVALID",
                            path=element_path,
                            message=(
                                f"Index set {_show(index_set.name)} mixes strings "
                                "and integers; its elements must be all strings or "
                                "all integers"
                            ),
                        ),
                    )
                    ok = False
                    continue
                if isinstance(element, int):
                    if abs(element) > INTEGER_BOUND_LIMIT:
                        self.log.add(
                            path,
                            _error(
                                code="INDEX_SET_INVALID",
                                path=element_path,
                                message=(
                                    f"The element at position {j} of index set "
                                    f"{_show(index_set.name)} lies outside "
                                    f"±{INTEGER_BOUND_LIMIT}"
                                ),
                            ),
                        )
                        ok = False
                        continue
                elif _ELEMENT.fullmatch(element) is None:
                    self.log.add(
                        path,
                        _error(
                            code="INDEX_SET_INVALID",
                            path=element_path,
                            message=(
                                f"Element {_show(element)} of index set "
                                f"{_show(index_set.name)} must be 1 to "
                                f"{ELEMENT_LIMIT} letters, digits, underscores, "
                                "dots or hyphens"
                            ),
                        ),
                    )
                    ok = False
                    continue
                if element in seen:
                    self.log.add(
                        path,
                        _error(
                            code="INDEX_SET_INVALID",
                            path=element_path,
                            message=(
                                f"Element {_show(element)} is listed more than once "
                                f"in index set {_show(index_set.name)}"
                            ),
                        ),
                    )
                    ok = False
                    continue
                seen.add(element)
            # Registered whenever its name is usable, errors or not, so the
            # templates that range over it are still checked without
            # knock-on errors; nothing is generated while any error stands.
            if usable:
                positions: dict = {}
                for j, element in enumerate(elements):
                    positions.setdefault(element, j)
                self.sets[index_set.name] = _Set(
                    name=index_set.name,
                    elements=list(elements),
                    texts=[
                        str(element) if ok else _show(element) for element in elements
                    ],
                    positions=positions,
                    order=index_set.order,
                )

    def _index_sets(self, names: list[str], path: str, source: str, owner: str):
        """Resolve a parameter's or family's ``indices``; None on error."""
        if not 1 <= len(names) <= MAX_ENTRIES:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=f"{path}.indices",
                    message=(
                        f"{owner} needs one to {MAX_ENTRIES} index sets in "
                        f"indices, got {len(names)}"
                    ),
                ),
            )
            return None
        resolved = []
        for d, name in enumerate(names):
            found = self.sets.get(name)
            if found is None:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=f"{path}.indices[{d}]",
                        message=f"{owner} names {_show(name)}, which is not a "
                        "declared (valid) index set",
                    ),
                )
                return None
            resolved.append(found)
        return tuple(resolved)

    def _declare_parameters(self) -> None:
        for p, parameter in enumerate(self.problem.parameters):
            path = f"parameters[{p}]"
            usable = self._declare_name(parameter.name, path, "parameter", path)
            owner = f"Parameter {_show(parameter.name)}"
            sets = self._index_sets(parameter.indices, path, path, owner)
            if parameter.default is not None and not math.isfinite(parameter.default):
                self.log.add(
                    path,
                    _error(
                        code="PARAMETER_TABLE_INVALID",
                        path=f"{path}.default",
                        message=f"{owner} has a default that is not a finite number",
                    ),
                )
            if sets is None:
                continue
            table: dict[tuple[int, ...], float] = {}
            for r, row in enumerate(parameter.values):
                key = self._row_key(path, r, row.key, sets, owner)
                if key is None:
                    continue
                if not math.isfinite(row.value):
                    self.log.add(
                        path,
                        _error(
                            code="PARAMETER_TABLE_INVALID",
                            path=f"{path}.values[{r}].value",
                            message=f"{owner} has a value that is not a finite "
                            f"number in row {r}",
                        ),
                    )
                    continue
                if key in table:
                    self.log.add(
                        path,
                        _error(
                            code="PARAMETER_TABLE_INVALID",
                            path=f"{path}.values[{r}].key",
                            message=(
                                f"{owner} has more than one row for the key "
                                f"({self._key_text(sets, key)})"
                            ),
                        ),
                    )
                    continue
                table[key] = float(row.value)
            # Registered, with its valid rows, whenever its name and indices
            # are usable, so references to it are checked without knock-on
            # errors; nothing is generated while any error stands.
            if usable:
                default = parameter.default
                self.params[parameter.name] = _Param(
                    index=p,
                    name=parameter.name,
                    sets=sets,
                    table=table,
                    default=float(default)
                    if default is not None and math.isfinite(default)
                    else None,
                )

    def _row_key(self, path, r, elements, sets, owner) -> tuple[int, ...] | None:
        """A parameter row's key as element positions; None (reported) if invalid."""
        if len(elements) != len(sets):
            self.log.add(
                path,
                _error(
                    code="PARAMETER_TABLE_INVALID",
                    path=f"{path}.values[{r}].key",
                    message=(
                        f"{owner} is indexed by {len(sets)} index "
                        f"set{'s' if len(sets) != 1 else ''}, but row {r} has a key "
                        f"of {len(elements)} element{'s' if len(elements) != 1 else ''}"
                    ),
                ),
            )
            return None
        key: list[int] = []
        for d, (element, index_set) in enumerate(zip(elements, sets)):
            # Keys of different types never match: 1 is not "1".
            position = index_set.positions.get(element)
            if position is None or type(element) is not type(index_set.elements[0]):
                self.log.add(
                    path,
                    _error(
                        code="PARAMETER_TABLE_INVALID",
                        path=f"{path}.values[{r}].key[{d}]",
                        message=(
                            f"{_show(element)} is not an element of index set "
                            f"{index_set.name}, which position {d} of {owner}'s key "
                            "ranges over"
                        ),
                    ),
                )
                return None
            key.append(position)
        return tuple(key)

    @staticmethod
    def _key_text(sets: tuple[_Set, ...], key: tuple[int, ...]) -> str:
        return ",".join(index_set.texts[position] for index_set, position in zip(sets, key))

    def _declare_families(self) -> None:
        for f, family in enumerate(self.problem.variable_families):
            path = f"variable_families[{f}]"
            # Registered whenever its name is declared and its indices
            # resolve, errors in its name or bounds notwithstanding, so the
            # templates that reference it are still checked without knock-on
            # errors; nothing is generated while any error stands.
            usable = self._declare_name(family.name, path, "variable family", path)
            if usable and family.name.startswith("__"):
                self.log.add(
                    path,
                    _error(
                        code="RESERVED_VARIABLE_NAME",
                        path=f"{path}.name",
                        message=(
                            f"Variable family name {family.name} is reserved: names "
                            "starting with '__' are for internal variables"
                        ),
                    ),
                )
            elif usable and family.name in self.explicit_types:
                self.log.add(
                    path,
                    _error(
                        code="DUPLICATE_TEMPLATE_NAME",
                        path=f"{path}.name",
                        message=(
                            f"Variable family {family.name} has the name of an "
                            "explicit variable, which templates could then no "
                            "longer reference; rename one of them"
                        ),
                    ),
                )
            # The family's type and bounds are checked once here, with the
            # very rule and wording an explicit variable gets (spec §14.5).
            probe = Variable.model_construct(
                name=_label(family.name),
                type=family.type,
                lower_bound=family.lower_bound,
                upper_bound=family.upper_bound,
                description=None,
            )
            bound_errors: list[SolveError] = []
            _check_variable_bounds(probe, path, bound_errors)
            for bound_error in bound_errors:
                self.log.add(path, bound_error)
            sets = self._index_sets(
                family.indices, path, path, f"Variable family {_show(family.name)}"
            )
            if sets is None:
                continue
            if usable:
                self.families[family.name] = _Family(
                    index=f,
                    name=family.name,
                    sets=sets,
                    type=family.type,
                    lower=family.lower_bound,
                    upper=family.upper_bound,
                    description=family.description,
                )

    # Scope and reference compilation --------------------------------------

    def _scope(
        self,
        source: str,
        path: str,
        for_each: list[str],
        where: list[str],
        outer: dict[str, tuple[int, _Set]],
        first_slot: int,
    ):
        """Compile one scope: ``(scope, bindings, where_names, where_ok)``.

        None when a ``for_each`` item fails, since the scope's references
        cannot be checked without its bindings; a failing ``where`` is
        reported and flagged (``where_ok``) but the references are still
        checked, so one pass reports every error it can.
        """
        ok = True
        bindings = dict(outer)
        loops: list[tuple[int, _Set]] = []
        if len(for_each) > MAX_ENTRIES:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=f"{path}.for_each",
                    message=f"for_each has {len(for_each)} items; at most "
                    f"{MAX_ENTRIES} are allowed",
                ),
            )
            return None
        for k, text in enumerate(for_each):
            item_path = f"{path}.for_each[{k}]"
            try:
                item = parse_for_each(text)
            except GrammarError as exc:
                self.log.add(
                    source,
                    _error(code="TEMPLATE_REFERENCE_INVALID", path=item_path, message=str(exc)),
                )
                ok = False
                continue
            if item.index in bindings:
                self.log.add(
                    source,
                    _error(
                        code="DUPLICATE_TEMPLATE_NAME",
                        path=item_path,
                        message=(
                            f"Index {item.index} is already bound in this scope; "
                            "use a different name"
                        ),
                    ),
                )
                ok = False
                continue
            if item.index in self.symbols or item.index in self.sets:
                self.log.add(
                    source,
                    _error(
                        code="DUPLICATE_TEMPLATE_NAME",
                        path=item_path,
                        message=(
                            f"Index {item.index} has the name of the declaration at "
                            f"{self.symbols.get(item.index, 'index_sets')}; an index "
                            "name must differ from every declared name"
                        ),
                    ),
                )
                ok = False
                continue
            loop_set = self.sets.get(item.set_name)
            if loop_set is None:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=item_path,
                        message=f"{item.set_name} is not a declared (valid) index set",
                    ),
                )
                ok = False
                continue
            slot = first_slot + len(loops)
            bindings[item.index] = (slot, loop_set)
            loops.append((slot, loop_set))
        if len(where) > MAX_ENTRIES:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=f"{path}.where",
                    message=f"where has {len(where)} conditions; at most "
                    f"{MAX_ENTRIES} are allowed",
                ),
            )
            return None
        if not ok:
            return None
        conds: list[_Cond] = []
        where_names: set[str] = set()
        for k, text in enumerate(where):
            cond_path = f"{path}.where[{k}]"
            cond = self._condition(source, cond_path, text, bindings, where_names)
            if cond is None:
                ok = False
            else:
                conds.append(cond)
        return _Scope(loops=tuple(loops), conds=tuple(conds)), bindings, where_names, ok

    def _condition(self, source, path, text, bindings, used: set[str]):
        try:
            parsed = parse_condition(text)
        except GrammarError as exc:
            self.log.add(
                source, _error(code="TEMPLATE_REFERENCE_INVALID", path=path, message=str(exc))
            )
            return None
        operands = []
        for operand in (parsed.left, parsed.right):
            if operand.kind == "number":
                operands.append(_Operand(kind="value", value=_ValRef(path=path, literal=operand.value)))
            elif operand.kind == "index":
                bound = bindings.get(operand.name)
                if bound is None:
                    self.log.add(
                        source,
                        _error(
                            code="TEMPLATE_REFERENCE_INVALID",
                            path=path,
                            message=self._unbound(operand.name),
                        ),
                    )
                    return None
                used.add(operand.name)
                operands.append(_Operand(kind="index", slot=bound[0]))
            else:
                reference = Reference(
                    head=operand.name,
                    indices=tuple(Index(name=name) for name in operand.indices),
                )
                value = self._parameter_ref(source, path, reference, bindings, used)
                if value is None:
                    return None
                operands.append(_Operand(kind="value", value=value))
        left, right = operands
        if left.kind != right.kind:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=path,
                    message=(
                        "The condition compares an index with a number; "
                        + _LITERAL_HINT
                    ),
                ),
            )
            return None
        if left.kind == "index":
            left_set = self._slot_set(bindings, left.slot)
            right_set = self._slot_set(bindings, right.slot)
            if left_set is not right_set:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=path,
                        message=(
                            f"The condition compares an index of {left_set.name} "
                            f"with an index of {right_set.name}; both sides must "
                            "range over the same index set"
                        ),
                    ),
                )
                return None
        return _Cond(path=path, operator=parsed.operator, left=left, right=right)

    @staticmethod
    def _slot_set(bindings, slot: int) -> _Set:
        for bound_slot, bound_set in bindings.values():
            if bound_slot == slot:
                return bound_set
        raise AssertionError("slot without a binding")  # pragma: no cover

    def _unbound(self, name: str) -> str:
        if name in self.sets:
            return (
                f"{name} is an index set, not an index; bind an index with "
                f'for_each ("i in {name}") and use that'
            )
        if name in self.params or name in self.families:
            return f"{name} needs its indices in brackets, e.g. {name}[i]"
        return f"{name} is not an index bound by for_each; " + _LITERAL_HINT

    def _positions(self, source, path, reference: Reference, expected, owner, bindings, used):
        """Resolve bracketed indices against ``expected`` sets; None on error."""
        indices = reference.indices or ()
        if len(indices) != len(expected):
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=path,
                    message=(
                        f"{owner} {reference.head} has {len(expected)} "
                        f"{'index' if len(expected) == 1 else 'indices'} "
                        f"({', '.join(s.name for s in expected)}), but the reference "
                        f"gives {len(indices)}"
                    ),
                ),
            )
            return None
        positions = []
        for d, (index, expected_set) in enumerate(zip(indices, expected)):
            bound = bindings.get(index.name)
            if bound is None:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=path,
                        message=self._unbound(index.name),
                    ),
                )
                return None
            slot, bound_set = bound
            if bound_set is not expected_set:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=path,
                        message=(
                            f"Index {index.name} ranges over {bound_set.name}, but "
                            f"position {d} of {owner.lower()} {reference.head} is "
                            f"{expected_set.name}"
                        ),
                    ),
                )
                return None
            if index.shift and bound_set.order == "none":
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=path,
                        message=(
                            f"Index {index.name} is shifted, but index set "
                            f"{bound_set.name} has no order; declare its order "
                            '"linear" or "cyclic" to allow shifts'
                        ),
                    ),
                )
                return None
            used.add(index.name)
            positions.append(_Pos(slot=slot, shift=index.shift, set=bound_set))
        return tuple(positions)

    def _parameter_ref(self, source, path, reference: Reference, bindings, used):
        head = reference.head
        param = self.params.get(head)
        if param is None:
            if head in self.families:
                what = "a variable family, not a parameter"
            elif head in self.sets:
                what = "an index set, not a parameter"
            elif head in self.symbols:
                what = "a parameter whose declaration has errors"
            else:
                what = "not a declared parameter"
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=path,
                    message=f"{head} is {what}",
                ),
            )
            return None
        if reference.indices is None:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=path,
                    message=f"Parameter {head} needs its indices in brackets, e.g. {head}[i]",
                ),
            )
            return None
        positions = self._positions(source, path, reference, param.sets, "Parameter", bindings, used)
        if positions is None:
            return None
        return _ValRef(path=path, param=param, positions=positions)

    def _value(self, source, path, value, bindings, used):
        """Compile a coefficient / rhs / weight: a number or a parameter."""
        if value is None:
            return None
        if not isinstance(value, str):
            return _ValRef(path=path, literal=value)
        if NUMBER.fullmatch(value.strip(" ")) is not None:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=path,
                    message="Write the number as a JSON number, not as a string",
                ),
            )
            return None
        try:
            reference = parse_reference(value)
        except GrammarError as exc:
            self.log.add(
                source, _error(code="TEMPLATE_REFERENCE_INVALID", path=path, message=str(exc))
            )
            return None
        return self._parameter_ref(source, path, reference, bindings, used)

    def _variable(self, source, path, text, bindings, used):
        """Compile a variable reference: a family with indices or an explicit name."""
        try:
            reference = parse_reference(text)
        except GrammarError as exc:
            self.log.add(
                source, _error(code="TEMPLATE_REFERENCE_INVALID", path=path, message=str(exc))
            )
            return None
        head = reference.head
        if reference.indices is None:
            if head in self.families or head in [f.name for f in self.problem.variable_families]:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=path,
                        message=f"Variable family {head} needs its indices, e.g. {head}[i]",
                    ),
                )
                return None
            if head not in self.explicit_types:
                self.log.add(
                    source,
                    _error(
                        code="UNKNOWN_VARIABLE",
                        path=path,
                        message=(
                            f"Variable {head} does not exist: a name without "
                            "brackets must be an explicit variable declared in "
                            "variables"
                        ),
                    ),
                )
                return None
            return _VarRef(family=None, name=head)
        family = self.families.get(head)
        if family is None:
            if head in self.params:
                what = "a parameter, not a variable family"
            elif head in self.sets:
                what = "an index set, not a variable family"
            elif head in self.symbols:
                what = "a variable family whose declaration has errors"
            else:
                what = "not a declared variable family"
            self.log.add(
                source,
                _error(code="TEMPLATE_REFERENCE_INVALID", path=path, message=f"{head} is {what}"),
            )
            return None
        positions = self._positions(
            source, path, reference, family.sets, "Variable family", bindings, used
        )
        if positions is None:
            return None
        return _VarRef(family=family, name=None, positions=positions)

    def _is_binary(self, ref: _VarRef) -> bool:
        if ref.family is not None:
            return ref.family.type == "binary"
        return self.explicit_types.get(ref.name) == "binary"

    def _unused(self, source, path, for_each_names, used) -> bool:
        """Report every bound index no reference uses (spec §14.5)."""
        ok = True
        for k, name in for_each_names:
            if name not in used:
                self.log.add(
                    source,
                    _error(
                        code="TEMPLATE_REFERENCE_INVALID",
                        path=f"{path}.for_each[{k}]",
                        message=(
                            f"Index {name} is bound but no reference uses it, so "
                            "every entry would be repeated once per element; use "
                            "it in a reference (or in an inner where), or remove it"
                        ),
                    ),
                )
                ok = False
        return ok

    @staticmethod
    def _loop_names(for_each: list[str]) -> list[tuple[int, str]]:
        names = []
        for k, text in enumerate(for_each):
            try:
                names.append((k, parse_for_each(text).index))
            except GrammarError:
                pass
        return names

    def _compile_objective_templates(self) -> None:
        objective = self.problem.objective
        groups = (
            ("linear", "objective.linear_term_templates", objective.linear_term_templates),
            ("quadratic", "objective.quadratic_term_templates", objective.quadratic_term_templates),
        )
        for kind, list_path, templates in groups:
            for t, template in enumerate(templates):
                source = f"{list_path}[{t}]"
                compiled = self._scope(source, source, template.for_each, template.where, {}, 0)
                if compiled is None:
                    continue
                scope, bindings, _where_names, where_ok = compiled
                used: set[str] = set()
                coefficient = self._value(
                    source, f"{source}.coefficient", template.coefficient, bindings, used
                )
                if kind == "linear":
                    texts = [("variable", template.variable)]
                else:
                    texts = [("variable1", template.variable1), ("variable2", template.variable2)]
                refs = [
                    self._variable(source, f"{source}.{name}", text, bindings, used)
                    for name, text in texts
                ]
                if coefficient is None or any(ref is None for ref in refs):
                    continue
                # Only with every reference compiled is ``used`` complete,
                # so an unused index is never reported as a knock-on error.
                if not self._unused(source, source, self._loop_names(template.for_each), used):
                    continue
                if not where_ok:
                    continue
                if (
                    kind == "quadratic"
                    and parse_reference(template.variable1) == parse_reference(template.variable2)
                    and self._is_binary(refs[0])
                ):
                    self.log.add(
                        source,
                        _error(
                            code="SELF_QUADRATIC_TERM",
                            path=source,
                            message=(
                                f"Quadratic term template multiplies "
                                f"{template.variable1.strip()} by itself; for binary "
                                "variables x*x = x, use a linear term template instead"
                            ),
                        ),
                    )
                    continue
                self.term_templates.append(
                    _TermTemplate(
                        source=source,
                        kind=kind,
                        scope=scope,
                        coefficient=coefficient,
                        variables=tuple(refs),
                    )
                )

    def _explicit_constraint_ids(self) -> set[str]:
        return {constraint.id for constraint in self.problem.constraints} | {
            constraint.id for constraint in self.problem.cardinality_constraints
        }

    def _compile_constraint_templates(self) -> None:
        explicit_ids = self._explicit_constraint_ids()
        template_ids: dict[str, str] = {}
        groups = (
            ("linear", "constraint_templates", self.problem.constraint_templates),
            ("cardinality", "cardinality_constraint_templates", self.problem.cardinality_constraint_templates),
        )
        for kind, list_path, templates in groups:
            for t, template in enumerate(templates):
                source = f"{list_path}[{t}]"
                ok = self._check_template_id(source, template.id, explicit_ids, template_ids)
                ok = self._check_weight(source, template) and ok
                compiled = self._scope(source, source, template.for_each, template.where, {}, 0)
                if compiled is None:
                    continue
                # Indices only this scope's own where uses do not count as
                # used (spec §14.5), so its where names are not collected.
                scope, outer, _outer_where, where_ok = compiled
                ok = ok and where_ok
                # Whether every use of the outer indices is known; if not, an
                # unused-index report could be a knock-on error, so none is made.
                usage_complete = True
                outer_used: set[str] = set()
                rhs = self._value(source, f"{source}.rhs", template.rhs, outer, outer_used)
                weight = self._value(source, f"{source}.weight", template.weight, outer, outer_used)
                if rhs is None or (template.weight is not None and weight is None):
                    ok = False
                    usage_complete = False
                if kind == "linear" and template.operator in ("<=", ">="):
                    if rhs is not None and rhs.literal is not None and not float(rhs.literal).is_integer():
                        self.log.add(
                            source,
                            _error(
                                code="NON_INTEGER_INEQUALITY",
                                path=f"{source}.rhs",
                                message=(
                                    f"Inequality constraint {_label(template.id)} requires "
                                    "integer coefficients and rhs for slack encoding, "
                                    f"got {rhs.literal}"
                                ),
                            ),
                        )
                        ok = False
                if kind == "cardinality" and rhs is not None and rhs.param is not None:
                    self.rhs_params.setdefault(rhs.param.name, []).append(f"{source}.rhs")
                members: list[_Member] = []
                member_list = template.terms if kind == "linear" else template.variables
                member_field = "terms" if kind == "linear" else "variables"
                first_slot = len(scope.loops)
                for m, member in enumerate(member_list):
                    member_path = f"{source}.{member_field}[{m}]"
                    inner = self._scope(
                        source, member_path, member.for_each, member.where, outer, first_slot
                    )
                    if inner is None:
                        ok = False
                        usage_complete = False
                        continue
                    member_scope, bindings, member_where, member_where_ok = inner
                    if not member_where_ok:
                        ok = False
                        usage_complete = False
                    # An outer index used by an inner where filters the
                    # inner iteration, which counts as a use (spec §14.5).
                    outer_used |= member_where & set(outer)
                    member_used: set[str] = set()
                    variable_path = (
                        f"{member_path}.variable" if kind == "linear" else member_path
                    )
                    variable = self._variable(
                        source, variable_path, member.variable, bindings, member_used
                    )
                    coefficient = None
                    if kind == "linear":
                        coefficient = self._value(
                            source, f"{member_path}.coefficient", member.coefficient,
                            bindings, member_used,
                        )
                        if coefficient is None:
                            ok = False
                            usage_complete = False
                        elif (
                            template.operator in ("<=", ">=")
                            and coefficient.literal is not None
                            and not float(coefficient.literal).is_integer()
                        ):
                            self.log.add(
                                source,
                                _error(
                                    code="NON_INTEGER_INEQUALITY",
                                    path=f"{member_path}.coefficient",
                                    message=(
                                        f"Inequality constraint {_label(template.id)} requires "
                                        "integer coefficients and rhs for slack "
                                        f"encoding, got {coefficient.literal}"
                                    ),
                                ),
                            )
                            ok = False
                    if variable is None:
                        ok = False
                        usage_complete = False
                        continue
                    if kind == "cardinality" and not self._is_binary(variable):
                        self.log.add(
                            source,
                            _error(
                                code="CARDINALITY_VARIABLE_NOT_BINARY",
                                path=member_path,
                                message=(
                                    f"Cardinality constraint template {_label(template.id)} "
                                    f"counts {member.variable.strip()}, which is an "
                                    "integer variable; only binary variables can be "
                                    "counted"
                                ),
                            ),
                        )
                        ok = False
                    outer_used |= member_used & set(outer)
                    inner_names = [
                        (k, name)
                        for k, name in self._loop_names(member.for_each)
                        if name not in outer
                    ]
                    if coefficient is not None or kind == "cardinality":
                        if not self._unused(source, member_path, inner_names, member_used):
                            ok = False
                    members.append(
                        _Member(scope=member_scope, variable=variable, coefficient=coefficient)
                    )
                if usage_complete and not self._unused(
                    source, source, self._loop_names(template.for_each), outer_used
                ):
                    ok = False
                if not ok:
                    continue
                self.constraint_templates.append(
                    _ConstraintTemplate(
                        source=source,
                        kind=kind,
                        id=template.id,
                        description=template.description,
                        type=template.type,
                        operator=template.operator,
                        scope=scope,
                        loop_names=tuple(name for _, name in self._loop_names(template.for_each)),
                        rhs=rhs,
                        weight=weight,
                        members=members,
                    )
                )

    def _check_template_id(self, source, template_id, explicit_ids, template_ids) -> bool:
        if not is_identifier(template_id) or template_id in RESERVED:
            self.log.add(
                source,
                _error(
                    code="TEMPLATE_REFERENCE_INVALID",
                    path=f"{source}.id",
                    message=f"The template id {_show(template_id)} must be {_NAME_RULE}",
                ),
            )
            return False
        if template_id in template_ids:
            self.log.add(
                source,
                _error(
                    code="DUPLICATE_TEMPLATE_NAME",
                    path=f"{source}.id",
                    message=(
                        f"Template id {template_id} is already used by "
                        f"{template_ids[template_id]}; constraint template ids must "
                        "be unique"
                    ),
                ),
            )
            return False
        if template_id in explicit_ids:
            self.log.add(
                source,
                _error(
                    code="DUPLICATE_TEMPLATE_NAME",
                    path=f"{source}.id",
                    message=(
                        f"Template id {template_id} is also the id of an explicit "
                        "constraint; template ids share the constraint id namespace"
                    ),
                ),
            )
            return False
        template_ids[template_id] = source
        return True

    def _check_weight(self, source, template) -> bool:
        if template.type == "hard":
            if template.weight is not None:
                self.log.add(
                    source,
                    _error(
                        code="HARD_CONSTRAINT_HAS_WEIGHT",
                        path=f"{source}.weight",
                        message=(
                            f"Hard constraint {_label(template.id)} must not carry a weight; "
                            "hard penalty strength is chosen by the penalty strategy"
                        ),
                    ),
                )
                return False
            return True
        weight = template.weight
        if weight is None or (not isinstance(weight, str) and weight <= 0):
            self.log.add(
                source,
                _error(
                    code="SOFT_CONSTRAINT_MISSING_WEIGHT",
                    path=f"{source}.weight",
                    message=f"Soft constraint {_label(template.id)} requires a weight > 0, got {weight}",
                ),
            )
            return False
        return True

    def _check_rhs_parameters(self) -> None:
        """A parameter used as a cardinality rhs holds whole counts (§14.10)."""
        for name in self.rhs_params:
            param = self.params[name]
            path = f"parameters[{param.index}]"
            source_param = self.problem.parameters[param.index]
            rows = [(f"{path}.values[{r}].value", row.value) for r, row in enumerate(source_param.values)]
            if source_param.default is not None:
                rows.append((f"{path}.default", source_param.default))
            for row_path, value in rows:
                if float(value).is_integer() and abs(value) <= INTEGER_BOUND_LIMIT:
                    continue
                self.log.add(
                    path,
                    _error(
                        code="PARAMETER_TABLE_INVALID",
                        path=row_path,
                        message=(
                            f"Parameter {name} is used as a cardinality rhs, so every "
                            f"value must be a whole number within ±{INTEGER_BOUND_LIMIT}, "
                            f"but it holds {value}"
                        ),
                    ),
                )

    # -- phase 2: the upper bound U ---------------------------------------

    def _bound_terms(self):
        """``(source, bindings)`` per family and template, in order (§14.9)."""
        limit = self.limit
        for family in self.families.values():
            sizes = [index_set.size for index_set in family.sets]
            yield f"variable_families[{family.index}]", _bounded_product(sizes, limit)
        for template in self.term_templates:
            sizes = [loop_set.size for _, loop_set in template.scope.loops]
            yield template.source, _bounded_product(sizes, limit)
        for template in self.constraint_templates:
            outer = _bounded_product([s.size for _, s in template.scope.loops], limit)
            inner = 1
            for member in template.members:
                inner += _bounded_product([s.size for _, s in member.scope.loops], limit)
                if inner > limit:
                    break
            yield template.source, outer * inner

    def _check_bound(self) -> SolveError | None:
        total = 0
        for source, amount in self._bound_terms():
            total += amount
            if total > self.limit:
                return _error(
                    code="TEMPLATE_EXPANSION_LIMIT",
                    path=source,
                    message=(
                        f"The templates would iterate over {total} or more "
                        f"bindings by the time {source} is expanded, above this "
                        f"server's max_template_bindings of {self.limit}; nothing "
                        "was generated"
                    ),
                )
        return None

    # -- phase 3: generation ----------------------------------------------

    def _count(self, amount: int, source: str) -> None:
        self.work += amount
        if self.work > self.limit:
            raise _LimitReached(source)

    def _bindings(self, scope: _Scope):
        slots = self.slots
        loops = scope.loops
        if not loops:
            yield
            return
        for combo in itertools.product(*(range(loop_set.size) for _, loop_set in loops)):
            for (slot, _), position in zip(loops, combo):
                slots[slot] = position
            yield

    def _position(self, pos: _Pos) -> int | None:
        position = self.slots[pos.slot] + pos.shift
        if pos.shift:
            size = pos.set.size
            if pos.set.order == "cyclic":
                position %= size
            elif not 0 <= position < size:
                return None
        return position

    def _key(self, positions: tuple[_Pos, ...]) -> tuple[int, ...] | None:
        key = []
        for pos in positions:
            position = self._position(pos)
            if position is None:
                return None
            key.append(position)
        return tuple(key)

    def _lookup(self, ref: _ValRef, key: tuple[int, ...], source: str):
        """The value of ``ref`` at ``key``; ``_MISSING`` (reported) if none."""
        if ref.param is None:
            return ref.literal
        param = ref.param
        value = param.table.get(key)
        if value is not None:
            return value
        if param.default is not None:
            return param.default
        self.building = False
        # Past the error ceilings the error is only counted, never built: a
        # table missing most of its keys must not cost memory per key.
        if not self.log.accepts(source):
            self.log.count(source, "PARAMETER_VALUE_MISSING")
            return _MISSING
        self.log.add(
            source,
            _error(
                code="PARAMETER_VALUE_MISSING",
                path=ref.path,
                message=(
                    f"Parameter {param.name} has no value for "
                    f"({self._key_text(param.sets, key)}) and no default"
                ),
            ),
        )
        return _MISSING

    def _passes(self, scope: _Scope, source: str) -> bool:
        """Evaluate ``where`` left to right, stopping at the first false."""
        for cond in scope.conds:
            values = []
            for operand in (cond.left, cond.right):
                if operand.kind == "index":
                    values.append(self.slots[operand.slot])
                    continue
                ref = operand.value
                value = self._lookup(ref, self._key(ref.positions) or (), source)
                if value is _MISSING:
                    return False
                values.append(value)
            left, right = values
            operator = cond.operator
            if operator == "==":
                holds = left == right
            elif operator == "!=":
                holds = left != right
            elif operator == "<":
                holds = left < right
            elif operator == "<=":
                holds = left <= right
            elif operator == ">":
                holds = left > right
            else:
                holds = left >= right
            if not holds:
                return False
        return True

    def _name(self, ref: _VarRef) -> str | None:
        if ref.family is None:
            return ref.name
        texts = []
        for pos in ref.positions:
            position = self._position(pos)
            if position is None:
                return None
            texts.append(pos.set.texts[position])
        return f"{ref.family.name}[{','.join(texts)}]"

    def _generated_id(self, template: _ConstraintTemplate) -> str:
        if not template.scope.loops:
            return template.id
        texts = [loop_set.texts[self.slots[slot]] for slot, loop_set in template.scope.loops]
        return f"{template.id}[{','.join(texts)}]"

    def _generate(self):
        variables: list[tuple[int, list[Variable]]] = []
        for family in self.families.values():
            source = f"variable_families[{family.index}]"
            created: list[Variable] = []
            for combo in itertools.product(*(range(s.size) for s in family.sets)):
                self._count(1, source)
                if not self.building:
                    continue
                texts = ",".join(s.texts[p] for s, p in zip(family.sets, combo))
                created.append(
                    Variable.model_construct(
                        name=f"{family.name}[{texts}]",
                        type=family.type,
                        lower_bound=family.lower,
                        upper_bound=family.upper,
                        description=family.description,
                    )
                )
            variables.append((family.index, created))

        linear: list[tuple[str, list[LinearTerm]]] = []
        quadratic: list[tuple[str, list[QuadraticTerm]]] = []
        for template in self.term_templates:
            created_terms = []
            for _ in self._bindings(template.scope):
                if not self._passes(template.scope, template.source):
                    continue
                names = [self._name(ref) for ref in template.variables]
                key = self._key(template.coefficient.positions)
                if any(name is None for name in names) or key is None:
                    template.skipped += 1
                    continue
                coefficient = self._lookup(template.coefficient, key, template.source)
                if coefficient is _MISSING:
                    continue
                self._count(1, template.source)
                template.generated += 1
                if not self.building:
                    continue
                if template.kind == "linear":
                    created_terms.append(
                        LinearTerm.model_construct(variable=names[0], coefficient=float(coefficient))
                    )
                else:
                    created_terms.append(
                        QuadraticTerm.model_construct(
                            variable1=names[0], variable2=names[1], coefficient=float(coefficient)
                        )
                    )
            (linear if template.kind == "linear" else quadratic).append(
                (template.source, created_terms)
            )

        constraints: list[tuple[_ConstraintTemplate, list, list]] = []
        for template in self.constraint_templates:
            created, member_counts = self._generate_constraints(template)
            constraints.append((template, created, member_counts))
        return variables, linear, quadratic, constraints

    def _generate_constraints(self, template: _ConstraintTemplate):
        created = []
        member_counts = []
        source = template.source
        for _ in self._bindings(template.scope):
            if not self._passes(template.scope, source):
                continue
            rhs_key = self._key(template.rhs.positions)
            weight_key = () if template.weight is None else self._key(template.weight.positions)
            if rhs_key is None or weight_key is None:
                template.skipped += 1
                continue
            self._count(1, source)
            collected = []
            counts = [0] * len(template.members)
            count = 0
            out_of_bounds = False
            for m, member in enumerate(template.members):
                for _ in self._bindings(member.scope):
                    if not self._passes(member.scope, source):
                        continue
                    name = self._name(member.variable)
                    coefficient_key = (
                        () if member.coefficient is None else self._key(member.coefficient.positions)
                    )
                    if name is None or coefficient_key is None:
                        out_of_bounds = True
                        break
                    count += 1
                    self._count(count, source)
                    collected.append((name, member.coefficient, coefficient_key))
                    counts[m] += 1
                if out_of_bounds:
                    break
            if out_of_bounds:
                template.skipped += 1
                continue
            rhs = self._lookup(template.rhs, rhs_key, source)
            weight = (
                None if template.weight is None else self._lookup(template.weight, weight_key, source)
            )
            coefficients = [
                1.0 if ref is None else self._lookup(ref, key, source)
                for _, ref, key in collected
            ]
            if rhs is _MISSING or weight is _MISSING or any(c is _MISSING for c in coefficients):
                continue
            template.generated += 1
            if not self.building:
                continue
            generated_id = self._generated_id(template)
            names = [name for name, _, _ in collected]
            if template.kind == "linear":
                if len(set(names)) != len(names):
                    template.merged += 1
                    if not template.merged_example:
                        repeated = next(n for n, c in Counter(names).items() if c > 1)
                        template.merged_example = f"{generated_id} names {repeated} more than once"
                created.append(
                    Constraint.model_construct(
                        id=generated_id,
                        description=template.description,
                        type=template.type,
                        terms=[
                            LinearTerm.model_construct(variable=name, coefficient=float(value))
                            for name, value in zip(names, coefficients)
                        ],
                        operator=template.operator,
                        rhs=float(rhs),
                        weight=None if weight is None else float(weight),
                    )
                )
            else:
                created.append(
                    CardinalityConstraint.model_construct(
                        id=generated_id,
                        description=template.description,
                        type=template.type,
                        variables=names,
                        operator=template.operator,
                        rhs=int(rhs),
                        weight=None if weight is None else float(weight),
                    )
                )
            member_counts.append(tuple(counts))
        return created, member_counts

    # -- assembly ---------------------------------------------------------

    def _assemble(self, generated) -> ExpandedProblem:
        family_variables, linear, quadratic, constraint_groups = generated
        problem = self.problem
        objective = problem.objective

        linear_terms = list(objective.linear_terms)
        linear_spans = []
        for source, terms in linear:
            linear_spans.append(_Span(len(linear_terms), len(linear_terms) + len(terms), source))
            linear_terms.extend(terms)
        quadratic_terms = list(objective.quadratic_terms)
        quadratic_spans = []
        for source, terms in quadratic:
            quadratic_spans.append(
                _Span(len(quadratic_terms), len(quadratic_terms) + len(terms), source)
            )
            quadratic_terms.extend(terms)

        constraints = list(problem.constraints)
        cardinality = list(problem.cardinality_constraints)
        constraint_spans, cardinality_spans = [], []
        constraint_members, cardinality_members = [], []
        for template, created, counts in constraint_groups:
            if template.kind == "linear":
                target, spans, members = constraints, constraint_spans, constraint_members
            else:
                target, spans, members = cardinality, cardinality_spans, cardinality_members
            spans.append(
                _Span(
                    len(target),
                    len(target) + len(created),
                    template.source,
                    loop_names=template.loop_names,
                    template_id=template.id,
                )
            )
            target.extend(created)
            members.extend(counts)

        # Generated variables no objective term or constraint uses are left
        # out (spec §14.14 item 2); one sharing an explicit name is kept so
        # the validator still reports the clash.
        referenced: set[str] = {term.variable for term in linear_terms}
        for term in quadratic_terms:
            referenced.add(term.variable1)
            referenced.add(term.variable2)
        for constraint in constraints:
            referenced.update(term.variable for term in constraint.terms)
        for constraint in cardinality:
            referenced.update(constraint.variables)
        explicit_names = set(self.explicit_types)
        variables = list(problem.variables)
        variable_spans = []
        warnings: list[SolveError] = []
        generated_total = 0
        left_out_total = 0
        for index, created in family_variables:
            source = f"variable_families[{index}]"
            kept = [v for v in created if v.name in referenced or v.name in explicit_names]
            left_out = [v.name for v in created if not (v.name in referenced or v.name in explicit_names)]
            generated_total += len(created)
            left_out_total += len(left_out)
            variable_spans.append(_Span(len(variables), len(variables) + len(kept), source))
            variables.extend(kept)
            if left_out:
                shown = ", ".join(left_out[:UNUSED_NAMES_SHOWN])
                more = len(left_out) - UNUSED_NAMES_SHOWN
                warnings.append(
                    _warning(
                        "UNUSED_TEMPLATE_VARIABLES",
                        source,
                        (
                            f"{len(left_out)} of the {len(created)} variables of family "
                            f"{problem.variable_families[index].name} "
                            f"{'appear' if len(left_out) != 1 else 'appears'} in no "
                            f"objective term or constraint and "
                            f"{'were' if len(left_out) != 1 else 'was'} left out: {shown}"
                            + (f" and {more} more" if more > 0 else "")
                        ),
                    )
                )

        for template in self.term_templates:
            self._template_warnings(template.source, template, "terms", warnings)
        for template in self.constraint_templates:
            self._template_warnings(template.source, template, "constraints", warnings)

        expanded = problem.model_copy(
            update={
                "variables": variables,
                "objective": objective.model_copy(
                    update={
                        "linear_terms": linear_terms,
                        "quadratic_terms": quadratic_terms,
                        "linear_term_templates": [],
                        "quadratic_term_templates": [],
                    }
                ),
                "constraints": constraints,
                "cardinality_constraints": cardinality,
                "index_sets": [],
                "parameters": [],
                "variable_families": [],
                "constraint_templates": [],
                "cardinality_constraint_templates": [],
            }
        )
        origins = _Origins(
            variables=_list_origin(len(problem.variables), variable_spans),
            linear_terms=_list_origin(len(objective.linear_terms), linear_spans),
            quadratic_terms=_list_origin(len(objective.quadratic_terms), quadratic_spans),
            linear_constraints=_list_origin(
                len(problem.constraints), constraint_spans, constraint_members
            ),
            cardinality=_list_origin(
                len(problem.cardinality_constraints), cardinality_spans, cardinality_members
            ),
            family_names=tuple(family.name for family in problem.variable_families),
            generated_variables=generated_total,
            left_out_variables=left_out_total,
        )
        return ExpandedProblem(
            source=problem,
            problem=expanded,
            warnings=tuple(warnings),
            _origins=origins,
        )

    @staticmethod
    def _template_warnings(source, template, noun, warnings: list[SolveError]) -> None:
        if template.skipped:
            warnings.append(
                _warning(
                    "TEMPLATE_BOUNDARY_SKIPPED",
                    source,
                    (
                        f"{source} left out {template.skipped} "
                        f"{noun if template.skipped != 1 else noun[:-1]} whose shifted "
                        "index runs past the end of a linear index set"
                    ),
                )
            )
        merged = template.merged if isinstance(template, _ConstraintTemplate) else 0
        if merged:
            warnings.append(
                _warning(
                    "TEMPLATE_TERMS_MERGED",
                    source,
                    (
                        f"{merged} constraint{'s' if merged != 1 else ''} generated by "
                        f"{source} name{'s' if merged == 1 else ''} the same variable "
                        "more than once, and the "
                        f"compiler sums its coefficients; for example "
                        f"{template.merged_example}"
                    ),
                )
            )
        if template.generated == 0:
            warnings.append(
                _warning(
                    "EMPTY_TEMPLATE_EXPANSION",
                    source,
                    f"{source} generated no {noun}",
                )
            )


def _bounded_product(sizes: list[int], limit: int) -> int:
    """Π sizes, stopping (and returning a value > limit) once past ``limit``."""
    product = 1
    for size in sizes:
        product *= size
        if product > limit:
            return product
    return product
