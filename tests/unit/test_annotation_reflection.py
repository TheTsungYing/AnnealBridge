"""Reference table for the four annotation-reflection helpers.

Recorded 2026-09-15 from HEAD 1886fad, i.e. from the implementations as
they stood *before* the batch 6 refactor:

- ``validation.problem_validator._optional_model`` / ``_optional_number``
- ``orchestration.limits._nested_model`` / ``_numeric_leaf``

Every expectation below is a literal captured from 1886fad, not a value
computed at test time. The two modules answer
overlapping but deliberately different questions (an option *block* versus
any nested model; an optional number versus any numeric leaf), so the rows
where they disagree are the interesting ones — for example
``Annotated[Block, "m"]`` is a nested model but not an option block, and
``Annotated[int, "m"] | None`` is an optional number while
``Annotated[int | None, "m"]`` is not.

Any cell that changes means the classification semantics drifted. When a
refactor makes a row fail, fix the refactor — do not edit the expected
value to make the suite pass.
"""

from typing import Annotated, Any, Literal, Optional, TypeVar, Union, get_args

import pytest
from pydantic import BaseModel

from annealbridge.models import (
    DWaveQPUOptions,
    FujitsuDAOptions,
    LeapHybridBQMOptions,
    LeapHybridCQMOptions,
    SolverPreferences,
)
from annealbridge.models.quantities import Count, Quantity
from annealbridge.models.reflection import (
    is_number_type,
    model_class,
    strip_annotated,
    union_members,
)
from annealbridge.orchestration.limits import _nested_model, _numeric_leaf
from annealbridge.validation.problem_validator import _optional_model, _optional_number


class Block(BaseModel):
    """Stand-in for a backend option block."""

    x: int = 0


class Other(BaseModel):
    """A second model, to exercise multi-model unions."""

    y: int = 0


_T = TypeVar("_T")


# (annotation, _optional_model, _optional_number, _nested_model, _numeric_leaf)
ANNOTATION_CASES = [
    pytest.param(int, None, False, None, True, id="int"),
    pytest.param(float, None, False, None, True, id="float"),
    pytest.param(bool, None, False, None, False, id="bool"),
    pytest.param(bool | None, None, False, None, False, id="bool|None"),
    pytest.param(int | float | None, None, False, None, True, id="int|float|None"),
    pytest.param(int | float, None, False, None, True, id="int|float"),
    pytest.param(
        Annotated[int, "m"] | None, None, True, None, True, id="Annotated[int]|None"
    ),
    pytest.param(
        Optional[Annotated[int, "m"]],  # noqa: UP045 - the typing.Union spelling
        None,
        True,
        None,
        True,
        id="Optional[Annotated[int]]",
    ),
    pytest.param(
        Annotated[int | None, "m"], None, False, None, True, id="Annotated[int|None]"
    ),
    pytest.param(
        Annotated[Annotated[int, "a"], "b"],
        None,
        False,
        None,
        True,
        id="Annotated[Annotated[int]]",
    ),
    pytest.param(Count, None, False, None, True, id="Count"),
    pytest.param(Quantity, None, False, None, True, id="Quantity"),
    pytest.param(Count | None, None, True, None, True, id="Count|None"),
    pytest.param(Quantity | None, None, True, None, True, id="Quantity|None"),
    pytest.param(list[Block], None, False, Block, False, id="list[Block]"),
    pytest.param(
        Annotated[Block, "m"], None, False, Block, False, id="Annotated[Block]"
    ),
    pytest.param(Block, None, False, Block, False, id="Block"),
    pytest.param(Block | None, Block, False, Block, False, id="Block|None"),
    pytest.param(
        Optional[Block],  # noqa: UP045 - the typing.Union spelling
        Block,
        False,
        Block,
        False,
        id="Optional[Block]",
    ),
    pytest.param(
        Union[Block, None],  # noqa: UP007 - the typing.Union spelling
        Block,
        False,
        Block,
        False,
        id="Union[Block,None]",
    ),
    pytest.param(
        Union[Block, Other],  # noqa: UP007 - the typing.Union spelling
        None,
        False,
        Block,
        False,
        id="Union[Block,Other]",
    ),
    pytest.param(
        Block | Other | None, None, False, Block, False, id="Block|Other|None"
    ),
    pytest.param(int | Block, None, False, Block, False, id="int|Block"),
    pytest.param(Literal["a", "b"], None, False, None, False, id="Literal"),
    pytest.param(
        Annotated[Block, "m"] | None,
        None,
        False,
        None,
        False,
        id="Annotated[Block]|None",
    ),
    pytest.param(list[int], None, False, None, False, id="list[int]"),
    pytest.param(str, None, False, None, False, id="str"),
    pytest.param(str | None, None, False, None, False, id="str|None"),
    pytest.param(type(None), None, False, None, False, id="NoneType"),
    pytest.param(
        Optional[int],  # noqa: UP045 - the typing.Union spelling
        None,
        True,
        None,
        True,
        id="Optional[int]",
    ),
    # A flag stays a flag behind Annotated, even as the single optional member.
    pytest.param(
        Annotated[bool, "m"] | None, None, False, None, False, id="Annotated[bool]|None"
    ),
    # _nested_model scans any generic alias's arguments, not only unions.
    pytest.param(dict[str, Block], None, False, Block, False, id="dict[str,Block]"),
    # An optional model behind Annotated is neither an option block nor nested.
    pytest.param(
        Annotated[Block | None, "m"],
        None,
        False,
        None,
        False,
        id="Annotated[Block|None]",
    ),
    pytest.param(
        Optional[Annotated[int | None, "m"]],  # noqa: UP045 - the typing.Union spelling
        None,
        False,
        None,
        True,
        id="Optional[Annotated[int|None]]",
    ),
    pytest.param(Any, None, False, None, False, id="Any"),
    pytest.param("Block", None, False, None, False, id="forward-ref-string"),
    pytest.param(_T, None, False, None, False, id="TypeVar"),
]


@pytest.mark.parametrize(
    ("annotation", "optional_model", "optional_number", "nested_model", "numeric_leaf"),
    ANNOTATION_CASES,
)
def test_annotation_reflection_matches_reference(
    annotation: object,
    optional_model: type[BaseModel] | None,
    optional_number: bool,
    nested_model: type[BaseModel] | None,
    numeric_leaf: bool,
) -> None:
    """Each helper answers exactly what it answered before the refactor."""
    assert _optional_model(annotation) is optional_model
    assert _optional_number(annotation) is optional_number
    assert _nested_model(annotation) is nested_model
    assert _numeric_leaf(annotation) is numeric_leaf


# (model, field name, _optional_model, _optional_number, _nested_model,
#  _numeric_leaf) for every field of the preference models the limit and
# validation code reflects over.
MODEL_FIELD_CASES = [
    pytest.param(
        SolverPreferences, "backend", None, False, None, False, id="prefs.backend"
    ),
    pytest.param(
        SolverPreferences, "num_reads", None, False, None, True, id="prefs.num_reads"
    ),
    pytest.param(
        SolverPreferences, "num_sweeps", None, False, None, True, id="prefs.num_sweeps"
    ),
    pytest.param(SolverPreferences, "seed", None, True, None, True, id="prefs.seed"),
    pytest.param(
        SolverPreferences, "top_k", None, False, None, True, id="prefs.top_k"
    ),
    pytest.param(
        SolverPreferences,
        "max_retries",
        None,
        False,
        None,
        True,
        id="prefs.max_retries",
    ),
    pytest.param(
        SolverPreferences,
        "penalty_multiplier",
        None,
        False,
        None,
        True,
        id="prefs.penalty_multiplier",
    ),
    pytest.param(
        SolverPreferences,
        "dwave_qpu",
        DWaveQPUOptions,
        False,
        DWaveQPUOptions,
        False,
        id="prefs.dwave_qpu",
    ),
    pytest.param(
        SolverPreferences,
        "leap_hybrid_bqm",
        LeapHybridBQMOptions,
        False,
        LeapHybridBQMOptions,
        False,
        id="prefs.leap_hybrid_bqm",
    ),
    pytest.param(
        SolverPreferences,
        "leap_hybrid_cqm",
        LeapHybridCQMOptions,
        False,
        LeapHybridCQMOptions,
        False,
        id="prefs.leap_hybrid_cqm",
    ),
    pytest.param(
        SolverPreferences,
        "fujitsu_da",
        FujitsuDAOptions,
        False,
        FujitsuDAOptions,
        False,
        id="prefs.fujitsu_da",
    ),
    pytest.param(
        DWaveQPUOptions,
        "annealing_time_us",
        None,
        True,
        None,
        True,
        id="qpu.annealing_time_us",
    ),
    pytest.param(
        DWaveQPUOptions,
        "chain_strength",
        None,
        True,
        None,
        True,
        id="qpu.chain_strength",
    ),
    pytest.param(
        DWaveQPUOptions, "auto_scale", None, False, None, False, id="qpu.auto_scale"
    ),
    pytest.param(
        LeapHybridBQMOptions,
        "time_limit_seconds",
        None,
        True,
        None,
        True,
        id="hybrid_bqm.time_limit_seconds",
    ),
    pytest.param(
        LeapHybridCQMOptions,
        "time_limit_seconds",
        None,
        True,
        None,
        True,
        id="hybrid_cqm.time_limit_seconds",
    ),
    pytest.param(
        FujitsuDAOptions,
        "time_limit_seconds",
        None,
        True,
        None,
        True,
        id="fujitsu.time_limit_seconds",
    ),
    pytest.param(
        FujitsuDAOptions, "num_run", None, True, None, True, id="fujitsu.num_run"
    ),
    pytest.param(
        FujitsuDAOptions, "num_group", None, True, None, True, id="fujitsu.num_group"
    ),
    pytest.param(
        FujitsuDAOptions,
        "num_output_solution",
        None,
        True,
        None,
        True,
        id="fujitsu.num_output_solution",
    ),
]

PREFERENCE_MODELS = (
    SolverPreferences,
    DWaveQPUOptions,
    LeapHybridBQMOptions,
    LeapHybridCQMOptions,
    FujitsuDAOptions,
)


def _models_reachable_from(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every BaseModel class reachable through ``root``'s field annotations.

    Walked with plain ``get_args`` recursion on purpose: this is the yardstick
    for the table's completeness, so it must not reuse the helpers under test.
    """
    found: set[type[BaseModel]] = set()
    pending: list[object] = [root]
    while pending:
        item = pending.pop()
        if isinstance(item, type) and issubclass(item, BaseModel):
            if item not in found:
                found.add(item)
                pending.extend(info.annotation for info in item.model_fields.values())
        else:
            pending.extend(get_args(item))
    return found


@pytest.mark.parametrize(
    (
        "model",
        "field_name",
        "optional_model",
        "optional_number",
        "nested_model",
        "numeric_leaf",
    ),
    MODEL_FIELD_CASES,
)
def test_model_field_reflection_matches_reference(
    model: type[BaseModel],
    field_name: str,
    optional_model: type[BaseModel] | None,
    optional_number: bool,
    nested_model: type[BaseModel] | None,
    numeric_leaf: bool,
) -> None:
    """Real preference fields classify exactly as they did before."""
    annotation = model.model_fields[field_name].annotation
    assert _optional_model(annotation) is optional_model
    assert _optional_number(annotation) is optional_number
    assert _nested_model(annotation) is nested_model
    assert _numeric_leaf(annotation) is numeric_leaf


def test_model_field_table_covers_every_preference_field() -> None:
    """A new preference field must be added to the table above.

    The reference table is only a safety net while it is complete; this
    test fails when a field is added to (or removed from) one of the
    preference models so the row gets recorded too.
    """
    recorded = {(case.values[0], case.values[1]) for case in MODEL_FIELD_CASES}
    declared = {
        (model, name) for model in PREFERENCE_MODELS for name in model.model_fields
    }
    assert recorded == declared


def test_preference_model_list_covers_every_option_block() -> None:
    """A new option block must be added to ``PREFERENCE_MODELS`` too.

    Without this, a block added to ``SolverPreferences`` would need only its
    own row in the table above, and the fields inside it would go unchecked.
    """
    assert set(PREFERENCE_MODELS) == _models_reachable_from(SolverPreferences)


UNION_CASES = [
    pytest.param(int | None, (int,), id="int|None"),
    pytest.param(
        Optional[int],  # noqa: UP045 - the typing.Union spelling
        (int,),
        id="Optional[int]",
    ),
    pytest.param(
        Union[Block, Other],  # noqa: UP007 - the typing.Union spelling
        (Block, Other),
        id="Union[Block,Other]",
    ),
    pytest.param(int | float | None, (int, float), id="int|float|None"),
]

NON_UNION_CASES = [
    pytest.param(int, id="int"),
    pytest.param(Literal["a"], id="Literal"),
    pytest.param(Annotated[int | None, "m"], id="Annotated[int|None]"),
    pytest.param(list[int], id="list[int]"),
]

STRIP_TO_TYPE_CASES = [
    pytest.param(Count, int, id="Count"),
    pytest.param(Quantity, float, id="Quantity"),
    pytest.param(Annotated[Annotated[int, "a"], "b"], int, id="Annotated[Annotated]"),
    pytest.param(int, int, id="int"),
]

STRIP_KEEPS_UNION_CASES = [
    pytest.param(
        Annotated[Block | None, "m"], Block | None, id="Annotated[Block|None]"
    ),
    pytest.param(int | None, int | None, id="int|None"),
]

MODEL_CLASS_CASES = [
    pytest.param(Block, Block, id="Block"),
    pytest.param(Block | None, None, id="Block|None"),
    pytest.param(list[Block], None, id="list[Block]"),
    pytest.param(Annotated[Block, "m"], None, id="Annotated[Block]"),
    pytest.param(int, None, id="int"),
    pytest.param(BaseModel, BaseModel, id="BaseModel"),
]

NUMBER_TYPE_CASES = [
    pytest.param(int, True, id="int"),
    pytest.param(float, True, id="float"),
    pytest.param(Count, True, id="Count"),
    pytest.param(Quantity, True, id="Quantity"),
    pytest.param(bool, False, id="bool"),
    pytest.param(Annotated[bool, "m"], False, id="Annotated[bool]"),
    pytest.param(int | None, False, id="int|None"),
    pytest.param(str, False, id="str"),
    pytest.param(Block, False, id="Block"),
]


class TestReflectionPrimitives:
    """The four primitives in ``models/reflection.py``, each on its own.

    The tables above pin the two *composed* rules; these pin the pieces
    they are composed from, so a drift shows up at the primitive that
    moved rather than only at the caller that noticed.
    """

    @pytest.mark.parametrize(("annotation", "expected"), UNION_CASES)
    def test_union_members_drops_none(
        self, annotation: object, expected: tuple[object, ...]
    ) -> None:
        """A union yields its members in order, without ``NoneType``."""
        assert union_members(annotation) == expected

    @pytest.mark.parametrize("annotation", NON_UNION_CASES)
    def test_union_members_rejects_non_unions(self, annotation: object) -> None:
        """Anything that is not a union yields None.

        ``Annotated[int | None, "m"]`` included: the wrapper is the
        outermost form, and this primitive does not see through it.
        """
        assert union_members(annotation) is None

    @pytest.mark.parametrize(("annotation", "expected"), STRIP_TO_TYPE_CASES)
    def test_strip_annotated_yields_the_underlying_type(
        self, annotation: object, expected: type
    ) -> None:
        """Every ``Annotated`` layer comes off; a bare type is returned as is."""
        assert strip_annotated(annotation) is expected

    @pytest.mark.parametrize(("annotation", "expected"), STRIP_KEEPS_UNION_CASES)
    def test_strip_annotated_keeps_the_inner_annotation(
        self, annotation: object, expected: object
    ) -> None:
        """Only the outer wrapper goes; a union inside stays a union."""
        assert strip_annotated(annotation) == expected

    @pytest.mark.parametrize(("annotation", "expected"), MODEL_CLASS_CASES)
    def test_model_class_matches_only_the_bare_class(
        self, annotation: object, expected: type[BaseModel] | None
    ) -> None:
        """Only a BaseModel subclass itself matches; aliases do not."""
        assert model_class(annotation) is expected

    @pytest.mark.parametrize(("annotation", "expected"), NUMBER_TYPE_CASES)
    def test_is_number_type(self, annotation: object, expected: bool) -> None:
        """``int`` / ``float`` and their annotated aliases, and nothing else."""
        assert is_number_type(annotation) is expected
