"""The two small grammars of schema 1.3 templates (validation/template_grammar).

Schema 1.3 spec 2026-09-25 §14.6 (v4), §14.16 item 8:

* every production -- ``for_each_item``, ``ref``, ``index``, ``condition``,
  ``operand``, ``CMP`` (longest match), ``IDENT``, ``UINT``, ``NUMBER`` (finite
  after conversion) and ``SP`` -- has positive and negative cases;
* every character class is explicit ASCII: Unicode digits, letters and the
  full-width space are refused, as are tabs, control characters and a
  trailing newline;
* ``in`` is reserved, literal elements (``x[i,0]``) are not references, a
  shift is ``+``/``-`` and one to nine digits without a leading zero
  (``p+0``, ten digits refused), and a where condition cannot shift;
* a malformed string raises :class:`GrammarError` (a ``ValueError``) whose
  message names the character position and shows at most the one offending
  character, JSON-escaped -- never more of the input.
"""

import re

import pytest

from annealbridge.validation.template_grammar import (
    IDENT_LIMIT,
    RESERVED,
    SHIFT_DIGITS,
    Condition,
    ForEachItem,
    GrammarError,
    Index,
    Operand,
    Reference,
    is_identifier,
    parse_condition,
    parse_for_each,
    parse_reference,
)

LONGEST = "n" * IDENT_LIMIT  # 64 characters: still an identifier
TOO_LONG = "n" * (IDENT_LIMIT + 1)


def _index(name: str) -> Operand:
    return Operand(kind="index", name=name)


def _number(value: float) -> Operand:
    return Operand(kind="number", value=value)


def _parameter(name: str, *indices: str) -> Operand:
    return Operand(kind="parameter", name=name, indices=indices)


def _error(parse, text: str) -> str:
    """The message of the :class:`GrammarError` ``parse(text)`` raises."""
    try:
        parse(text)
    except GrammarError as exc:
        return str(exc)
    raise AssertionError(f"{text!r} parsed without a GrammarError")


class TestIdentifier:
    """``IDENT := [A-Za-z_][A-Za-z0-9_]*`` (ASCII, at most 64 characters)."""

    @pytest.mark.parametrize("text", ["x", "_", "_x1", "Aa_09", LONGEST])
    def test_ascii_identifiers(self, text):
        assert is_identifier(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "1x",
            "x-y",
            "x.y",
            "x y",
            "x[0]",
            TOO_LONG,
            "é",  # é: a Unicode letter
            "x١",  # Arabic-Indic digit one
            "ｘ",  # full-width x
            "x\n",
        ],
    )
    def test_everything_else_is_not_an_identifier(self, text):
        assert not is_identifier(text)

    def test_in_is_the_only_reserved_word(self):
        assert RESERVED == frozenset({"in"})


class TestForEachItem:
    """``for_each_item := IDENT SP "in" SP IDENT``."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("i in s", ForEachItem(index="i", set_name="s")),
            ("city in cities_2", ForEachItem(index="city", set_name="cities_2")),
            ("i   in   s", ForEachItem(index="i", set_name="s")),
            ("  i in s  ", ForEachItem(index="i", set_name="s")),
            (f"{LONGEST} in {LONGEST}", ForEachItem(index=LONGEST, set_name=LONGEST)),
            ("inx in s", ForEachItem(index="inx", set_name="s")),
            ("i in ins", ForEachItem(index="i", set_name="ins")),
        ],
    )
    def test_valid_items(self, text, expected):
        assert parse_for_each(text) == expected

    @pytest.mark.parametrize(
        "text, message",
        [
            (
                "",
                'expected an index name (a for_each item is "index in set") at '
                "character 1, found the end of the text",
            ),
            (
                "1i in s",
                'expected an index name (a for_each item is "index in set") at '
                'character 1, found "1"',
            ),
            (
                "iin s",
                'expected "in" (a for_each item is "index in set") at character 5, '
                'found "s"',
            ),
            (
                "i s",
                'expected "in" (a for_each item is "index in set") at character 3, '
                'found "s"',
            ),
            (
                "i IN s",
                'expected "in" (a for_each item is "index in set") at character 3, '
                'found "I"',
            ),
            (
                "i in",
                'expected a space and an index set name after "in" at character 5, '
                "found the end of the text",
            ),
            (
                "i ins",
                'expected "in" (a for_each item is "index in set") at character 3, '
                'found "i"',
            ),
            ("i in s t", 'expected the end of the text at character 8, found "t"'),
            ("i in s,", 'expected the end of the text at character 7, found ","'),
        ],
    )
    def test_malformed_items(self, text, message):
        assert _error(parse_for_each, text) == message

    @pytest.mark.parametrize("text, position", [("in in s", 1), ("  in  in s", 3)])
    def test_in_is_reserved_as_an_index_name(self, text, position):
        assert _error(parse_for_each, text) == (
            f'the index name at character {position} is "in", which is '
            "reserved; choose another name"
        )

    @pytest.mark.parametrize("text, position", [("i in in", 6), ("i  in   in  ", 9)])
    def test_in_is_reserved_as_an_index_set_name(self, text, position):
        assert _error(parse_for_each, text) == (
            f'the index set name at character {position} is "in", which is '
            "reserved and cannot name an index set"
        )

    def test_names_longer_than_the_limit(self):
        assert _error(parse_for_each, f"{TOO_LONG} in s") == (
            "the name at character 1 is longer than 64 characters"
        )
        assert _error(parse_for_each, f"i in {TOO_LONG}") == (
            "the name at character 6 is longer than 64 characters"
        )

    @pytest.mark.parametrize(
        "text, message",
        [
            # SP is one or more ASCII spaces and nothing else.
            (
                "i\tin s",
                'expected a space and "in" (a for_each item is "index in set") at '
                'character 2, found "\\t"',
            ),
            (
                "i　in s",
                'expected a space and "in" (a for_each item is "index in set") at '
                'character 2, found "\\u3000"',
            ),
            (
                "i in\ns",
                'expected a space and an index set name after "in" at character 5, '
                'found "\\n"',
            ),
            ("i in s\n", 'expected the end of the text at character 7, found "\\n"'),
            (
                "i\x00in s",
                'expected a space and "in" (a for_each item is "index in set") at '
                'character 2, found "\\u0000"',
            ),
            # Unicode letters and digits are not identifier characters.
            (
                "é in s",
                'expected an index name (a for_each item is "index in set") at '
                'character 1, found "\\u00e9"',
            ),
            (
                "١ in s",
                'expected an index name (a for_each item is "index in set") at '
                'character 1, found "\\u0661"',
            ),
            ("i in s١", 'expected the end of the text at character 7, found "\\u0661"'),
        ],
    )
    def test_only_ascii_spaces_and_ascii_names(self, text, message):
        assert _error(parse_for_each, text) == message


class TestReference:
    """``ref := IDENT ("[" index ("," index)* "]")?`` and ``index``."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("x", Reference(head="x", indices=None)),
            ("x[i]", Reference(head="x", indices=(Index("i"),))),
            ("x[i,p]", Reference(head="x", indices=(Index("i"), Index("p")))),
            ("x[ i , p ]", Reference(head="x", indices=(Index("i"), Index("p")))),
            ("x [i]", Reference(head="x", indices=(Index("i"),))),
            ("x[i,p+1]", Reference(head="x", indices=(Index("i"), Index("p", 1)))),
            ("x[i,p - 2]", Reference(head="x", indices=(Index("i"), Index("p", -2)))),
            ("x[p+10]", Reference(head="x", indices=(Index("p", 10),))),
            (
                "x[p+123456789]",
                Reference(head="x", indices=(Index("p", 123_456_789),)),
            ),
            (
                "x[p-999999999]",
                Reference(head="x", indices=(Index("p", -999_999_999),)),
            ),
            (f"{LONGEST}[{LONGEST}]", Reference(head=LONGEST, indices=(Index(LONGEST),))),
            (
                "x[a,b,c,d,e,f,g,h]",
                Reference(head="x", indices=tuple(Index(n) for n in "abcdefgh")),
            ),
        ],
    )
    def test_valid_references(self, text, expected):
        assert parse_reference(text) == expected

    @pytest.mark.parametrize(
        "text, message",
        [
            ("", "expected a family, parameter or variable name at character 1, found the end of the text"),
            ("1x", 'expected a family, parameter or variable name at character 1, found "1"'),
            ("x[]", 'expected an index name (literal elements such as a or 0 are not supported; bind an index with for_each) at character 3, found "]"'),
            ("x[i", 'expected "," or "]" at character 4, found the end of the text'),
            ("x[i,]", 'expected an index name (literal elements such as a or 0 are not supported; bind an index with for_each) at character 5, found "]"'),
            ("x[i]]", 'expected the end of the text at character 5, found "]"'),
            ("x[i] y", 'expected the end of the text at character 6, found "y"'),
            ("x.y", 'expected the end of the text at character 2, found "."'),
            ("x(i)", 'expected the end of the text at character 2, found "("'),
            ("x[i*2]", 'expected "," or "]" at character 4, found "*"'),
            ("x[i]+1", 'expected the end of the text at character 5, found "+"'),
            ("x[+1]", 'expected an index name (literal elements such as a or 0 are not supported; bind an index with for_each) at character 3, found "+"'),
            ("x[i+]", 'expected a whole number after "+" at character 5, found "]"'),
            ("x[i-]", 'expected a whole number after "-" at character 5, found "]"'),
            ("x[i+-1]", 'expected a whole number after "+" at character 5, found "-"'),
        ],
    )
    def test_malformed_references(self, text, message):
        assert _error(parse_reference, text) == message

    @pytest.mark.parametrize(
        "text, position, found",
        [
            ("x[i,0]", 5, '"0"'),
            ("x[a,0]", 5, '"0"'),
            ("x[0]", 3, '"0"'),
            ("x[-1]", 3, '"-"'),
        ],
    )
    def test_literal_elements_are_not_indices(self, text, position, found):
        """``x[a,0]``: a bound index is required; the message says so."""
        assert _error(parse_reference, text) == (
            "expected an index name (literal elements such as a or 0 are not "
            "supported; bind an index with for_each) at character "
            f"{position}, found {found}"
        )

    def test_a_zero_shift_is_refused(self):
        """``UINT := [1-9][0-9]{0,8}``: ``p+0`` is not a shift."""
        assert _error(parse_reference, "x[i,p+0]") == (
            "the shift at character 7 must be a positive whole number without "
            "leading zeros"
        )

    def test_a_shift_with_a_leading_zero_is_refused(self):
        assert _error(parse_reference, "x[p+01]") == (
            "the shift at character 5 must be a positive whole number without "
            "leading zeros"
        )

    @pytest.mark.parametrize("digits", ["1234567890", "12345678901234567890"])
    def test_a_shift_has_at_most_nine_digits(self, digits):
        assert SHIFT_DIGITS == 9
        assert _error(parse_reference, f"x[p+{digits}]") == (
            "the shift at character 5 has more than 9 digits"
        )

    def test_names_longer_than_the_limit(self):
        assert _error(parse_reference, TOO_LONG) == (
            "the name at character 1 is longer than 64 characters"
        )
        assert _error(parse_reference, f"x[{TOO_LONG}]") == (
            "the name at character 3 is longer than 64 characters"
        )

    @pytest.mark.parametrize(
        "text, message",
        [
            ("x[i\t]", 'expected "," or "]" at character 4, found "\\t"'),
            ("x[i]\n", 'expected the end of the text at character 5, found "\\n"'),
            ("x[i]\r", 'expected the end of the text at character 5, found "\\r"'),
            ("x[i　]", 'expected "," or "]" at character 4, found "\\u3000"'),
            ("x[i+١]", 'expected a whole number after "+" at character 5, found "\\u0661"'),
            ("x[é]", 'expected an index name (literal elements such as a or 0 are not supported; bind an index with for_each) at character 3, found "\\u00e9"'),
            ("ｘ[i]", 'expected a family, parameter or variable name at character 1, found "\\uff58"'),
            ("x[i]\x07", 'expected the end of the text at character 5, found "\\u0007"'),
        ],
    )
    def test_only_ascii(self, text, message):
        assert _error(parse_reference, text) == message

    def test_a_python_expression_is_not_a_reference(self):
        assert _error(parse_reference, "__import__('os')") == (
            'expected the end of the text at character 11, found "("'
        )


class TestCondition:
    """``condition := operand CMP operand`` and ``operand``."""

    @pytest.mark.parametrize("operator", ["==", "!=", "<=", ">=", "<", ">"])
    def test_every_comparison(self, operator):
        assert parse_condition(f"i {operator} j") == Condition(
            left=_index("i"), operator=operator, right=_index("j")
        )

    @pytest.mark.parametrize(
        "text, operator",
        [("i<=j", "<="), ("i>=j", ">="), ("i==j", "=="), ("i!=j", "!="), ("i<j", "<"), ("i>j", ">")],
    )
    def test_the_longest_operator_wins(self, text, operator):
        """``<=`` is one operator, never ``<`` followed by ``=j``."""
        assert parse_condition(text) == Condition(
            left=_index("i"), operator=operator, right=_index("j")
        )

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("d[i] == 1", Condition(_parameter("d", "i"), "==", _number(1.0))),
            ("d[i,j] >= 0.5", Condition(_parameter("d", "i", "j"), ">=", _number(0.5))),
            ("d[ i , j ]>=0.5", Condition(_parameter("d", "i", "j"), ">=", _number(0.5))),
            ("-1 < d[i]", Condition(_number(-1.0), "<", _parameter("d", "i"))),
            ("1e3 > d[i]", Condition(_number(1000.0), ">", _parameter("d", "i"))),
            ("d[i] < 1E-3", Condition(_parameter("d", "i"), "<", _number(0.001))),
            ("d[i] < 2.5e+2", Condition(_parameter("d", "i"), "<", _number(250.0))),
            ("d[i]==-1", Condition(_parameter("d", "i"), "==", _number(-1.0))),
            ("-0 == 0", Condition(_number(-0.0), "==", _number(0.0))),
            ("1 == 2", Condition(_number(1.0), "==", _number(2.0))),
            ("d[i] < 1e308", Condition(_parameter("d", "i"), "<", _number(1e308))),
            ("i == j ", Condition(_index("i"), "==", _index("j"))),
            # A literal element parses as a number; the expander refuses
            # comparing it with an index (spec §14.6).
            ("p == 0", Condition(_index("p"), "==", _number(0.0))),
        ],
    )
    def test_valid_conditions(self, text, expected):
        assert parse_condition(text) == expected

    def test_a_single_equals_sign(self):
        assert _error(parse_condition, "i = j") == (
            'use "==" to compare (character 3 is a single "=")'
        )
        assert _error(parse_condition, "i => j") == (
            'use "==" to compare (character 3 is a single "=")'
        )

    @pytest.mark.parametrize(
        "text, message",
        [
            ("i === j", 'expected an index name, a parameter reference or a number at character 5, found "="'),
            ("i <> j", 'expected an index name, a parameter reference or a number at character 4, found ">"'),
            ("i < = j", 'expected an index name, a parameter reference or a number at character 5, found "="'),
            ("== j", 'expected an index name, a parameter reference or a number at character 1, found "="'),
            ("i ==", "expected an index name, a parameter reference or a number at character 5, found the end of the text"),
            ("i j", 'expected one of ==, !=, <, <=, >, >= at character 3, found "j"'),
            ("i", "expected one of ==, !=, <, <=, >, >= at character 2, found the end of the text"),
            ("i == j == k", 'expected the end of the text at character 8, found "="'),
            ("i == j and j == k", 'expected the end of the text at character 8, found "a"'),
            ("d[] == 1", 'expected an index name (literal elements such as a or 0 are not supported; bind an index with for_each) at character 3, found "]"'),
            ("d[i == 1", 'expected "," or "]" at character 5, found "="'),
        ],
    )
    def test_malformed_conditions(self, text, message):
        assert _error(parse_condition, text) == message

    @pytest.mark.parametrize(
        "text, message",
        [
            # NUMBER := "-"? ("0" | [1-9][0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
            ("d[i] < 01", 'expected a number such as 3, -2 or 0.5 at character 8, found "0"'),
            ("d[i] < 1.", 'expected a number such as 3, -2 or 0.5 at character 8, found "1"'),
            ("d[i] < .5", 'expected an index name, a parameter reference or a number at character 8, found "."'),
            ("d[i] < 1.5.3", 'expected a number such as 3, -2 or 0.5 at character 8, found "1"'),
            ("d[i] < 1x", 'expected a number such as 3, -2 or 0.5 at character 8, found "1"'),
            ("d[i] < 1_0", 'expected a number such as 3, -2 or 0.5 at character 8, found "1"'),
            ("d[i] < 1e", 'expected a number such as 3, -2 or 0.5 at character 8, found "1"'),
            ("d[i] < 1e+", 'expected a number such as 3, -2 or 0.5 at character 8, found "1"'),
            ("d[i] < +1", 'expected an index name, a parameter reference or a number at character 8, found "+"'),
            ("d[i] < -", 'expected a number such as 3, -2 or 0.5 at character 8, found "-"'),
            ("d[i] < --1", 'expected a number such as 3, -2 or 0.5 at character 8, found "-"'),
            ("d[i] < 0x10", 'expected a number such as 3, -2 or 0.5 at character 8, found "0"'),
        ],
    )
    def test_malformed_numbers(self, text, message):
        assert _error(parse_condition, text) == message

    @pytest.mark.parametrize(
        "text, position",
        [("1e999 > d[i]", 1), ("d[i] < 1e309", 8), ("d[i] > -1e400", 8)],
    )
    def test_a_number_must_be_finite(self, text, position):
        assert _error(parse_condition, text) == (
            f"the number at character {position} is not finite"
        )

    @pytest.mark.parametrize(
        "text, position",
        [("d[i+1] == 1", 3), ("d[i-1] == 1", 3), ("1 == d[j,i+2]", 10), ("dist[a,b+1]<0", 8)],
    )
    def test_a_where_condition_cannot_shift(self, text, position):
        """The position is where the shifted index starts; no name is echoed."""
        assert _error(parse_condition, text) == (
            f"the index at character {position} is shifted, but a where "
            "condition cannot shift an index; shifts are only allowed in the "
            "generated entry's own references"
        )

    @pytest.mark.parametrize(
        "text, message",
        [
            ("d[i] == ١", 'expected an index name, a parameter reference or a number at character 9, found "\\u0661"'),
            ("d[i] == １", 'expected an index name, a parameter reference or a number at character 9, found "\\uff11"'),
            ("i　== j", 'expected one of ==, !=, <, <=, >, >= at character 2, found "\\u3000"'),
            ("i ==\tj", 'expected an index name, a parameter reference or a number at character 5, found "\\t"'),
            ("i == j\n", 'expected the end of the text at character 7, found "\\n"'),
            ("i == j\x00", 'expected the end of the text at character 7, found "\\u0000"'),
            ("é == j", 'expected an index name, a parameter reference or a number at character 1, found "\\u00e9"'),
        ],
    )
    def test_only_ascii(self, text, message):
        assert _error(parse_condition, text) == message

    def test_names_longer_than_the_limit(self):
        assert _error(parse_condition, f"{TOO_LONG} == j") == (
            "the name at character 1 is longer than 64 characters"
        )
        assert parse_condition(f"{LONGEST} == j") == Condition(
            left=_index(LONGEST), operator="==", right=_index("j")
        )


# One malformed string per raise site of the module (spec §14.6: a syntax
# error reports the character position and that one character JSON-escaped).
_EVERY_ERROR = [
    (parse_for_each, "i  s"),
    (parse_for_each, "i\tin s"),
    (parse_for_each, "i in"),
    (parse_for_each, f"{TOO_LONG} in s"),
    (parse_for_each, "in in s"),
    (parse_for_each, "i in in"),
    (parse_reference, "x[i"),
    (parse_reference, "x[i+]"),
    (parse_reference, "x[p+0]"),
    (parse_reference, "x[p+1234567890]"),
    (parse_reference, "x[i]\n"),
    (parse_condition, "i = j"),
    (parse_condition, "i j"),
    (parse_condition, "d[i] < 1x"),
    (parse_condition, "d[i] < 1e999"),
    (parse_condition, "d[i+1] == 1"),
]


class TestErrorMessages:
    """What a :class:`GrammarError` may say (spec §14.6)."""

    def test_grammar_error_is_a_value_error(self):
        assert issubclass(GrammarError, ValueError)

    @pytest.mark.parametrize(
        "parse, text", _EVERY_ERROR, ids=[repr(text) for _, text in _EVERY_ERROR]
    )
    def test_every_grammar_error_names_a_character_position(self, parse, text):
        assert re.search(r"\bcharacter \d+\b", _error(parse, text))

    @pytest.mark.parametrize(
        "parse, text",
        [
            (parse_for_each, "i in s; SENTINEL_TEXT"),
            (parse_for_each, "i in s\nSENTINEL_TEXT"),
            (parse_reference, "x[i*SENTINEL_TEXT]"),
            (parse_reference, "x[i]('SENTINEL_TEXT')"),
            (parse_reference, "__import__('SENTINEL_TEXT')"),
            (parse_reference, "x[p+99999999999999999999SENTINEL_TEXT]"),
            (parse_condition, "i == j or SENTINEL_TEXT"),
            (parse_condition, "d[i] < 1eSENTINEL_TEXT"),
            (parse_condition, "i = SENTINEL_TEXT"),
        ],
    )
    def test_the_rest_of_the_input_is_never_echoed(self, parse, text):
        message = _error(parse, text)
        assert "SENTINEL" not in message
        assert "TEXT" not in message

    @pytest.mark.parametrize(
        "parse, text",
        [(parse, text) for parse, text in _EVERY_ERROR if "found" in _error(parse, text)],
    )
    def test_found_shows_one_json_escaped_character(self, parse, text):
        message = _error(parse, text)
        assert re.search(
            r'found (the end of the text|"(?:[^"\\]|\\[nrt"\\]|\\u[0-9a-f]{4})")$',
            message,
        ), message
