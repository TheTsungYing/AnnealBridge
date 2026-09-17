"""The capability / availability models in ``models.capabilities`` (3a spec §7, §8)."""

import pytest
from pydantic import ValidationError

from annealbridge.models import (
    AvailabilityStatus,
    ParameterLimit,
    SolverCapabilities,
)
from annealbridge.solvers import (
    SimulatedAnnealingBackend,
    SimulatedBifurcationBackend,
    SolverRegistry,
    TabuBackend,
)
from annealbridge.solvers import SolverCapabilities as ReexportedCapabilities
from annealbridge.solvers.base import AvailabilityStatus as ReexportedStatus
from annealbridge.solvers.simulated_annealing import _SEED_LIMIT
from annealbridge.solvers.simulated_bifurcation import _SEED_LIMIT as _SB_SEED_LIMIT
from annealbridge.solvers.tabu import _SEED_LIMIT as _TABU_SEED_LIMIT


def make_capabilities(**overrides) -> SolverCapabilities:
    fields = dict(
        name="fake",
        remote=False,
        heuristic=True,
        exhaustive=False,
        supports_seed=True,
        supports_num_reads=True,
        supports_time_limit=False,
        supported_model_types=["bqm"],
        returns_multiple_samples=True,
        description="A fake backend.",
    )
    fields.update(overrides)
    return SolverCapabilities(**fields)


class TestReexports:
    def test_solvers_package_reexports_the_models_classes(self):
        assert ReexportedCapabilities is SolverCapabilities
        assert ReexportedStatus is AvailabilityStatus


class TestParameterLimit:
    def test_fields_round_trip(self):
        limit = ParameterLimit(
            preference="dwave_qpu.annealing_time_us",
            limit="annealing_time_us",
            error_code="QPU_ANNEALING_TIME_LIMIT",
        )
        assert limit.preference == "dwave_qpu.annealing_time_us"
        assert limit.limit == "annealing_time_us"
        assert limit.error_code == "QPU_ANNEALING_TIME_LIMIT"

    def test_all_fields_are_required(self):
        with pytest.raises(ValidationError):
            ParameterLimit(preference="num_reads", limit="reads")  # type: ignore[call-arg]

    def test_capabilities_accept_a_list_of_limits(self):
        caps = make_capabilities(
            parameter_limits=[
                ParameterLimit(preference="num_reads", limit="reads", error_code="QPU_READS_LIMIT")
            ]
        )
        assert [limit.limit for limit in caps.parameter_limits] == ["reads"]


class TestAvailabilityStatus:
    def test_available_only_for_the_available_category(self):
        assert AvailabilityStatus(category="available").available is True
        for category in ("not_installed", "credentials_missing", "config_invalid", "unavailable"):
            assert AvailabilityStatus(category=category).available is False

    def test_detail_and_error_code_default_to_none(self):
        status = AvailabilityStatus(category="unavailable")
        assert status.detail is None
        assert status.error_code is None

    def test_backend_may_name_a_specific_error_code(self):
        status = AvailabilityStatus(
            category="config_invalid", detail="broken", error_code="DWAVE_CONFIG_INVALID"
        )
        assert status.error_code == "DWAVE_CONFIG_INVALID"

    def test_unknown_category_is_rejected(self):
        with pytest.raises(ValidationError):
            AvailabilityStatus(category="down")  # type: ignore[arg-type]


class TestSolverCapabilities:
    def test_new_fields_default_so_phase2_declarations_still_validate(self):
        caps = make_capabilities()
        assert caps.supports_num_sweeps is False
        assert caps.requires_embedding is False
        assert caps.parameter_limits == []

    def test_preferred_model_type_is_the_first_declared(self):
        assert make_capabilities(supported_model_types=["cqm", "bqm"]).preferred_model_type == "cqm"
        assert make_capabilities(supported_model_types=["bqm"]).preferred_model_type == "bqm"

    def test_empty_model_types_are_rejected(self):
        with pytest.raises(ValidationError):
            make_capabilities(supported_model_types=[])

    def test_unknown_model_type_is_rejected(self):
        with pytest.raises(ValidationError):
            make_capabilities(supported_model_types=["ising"])

    def test_parameter_limits_reject_tuples(self):
        with pytest.raises(ValidationError):
            make_capabilities(parameter_limits=[("num_reads", "reads", "QPU_READS_LIMIT")])


class TestSeedRange:
    """``seed_min`` / ``seed_max``: the inclusive seed range a backend accepts.

    Both default to None (no declared range). A declaration that contradicts
    itself is refused when it is built, so the validator never has to guess
    what a half-declared or inverted range means.
    """

    def test_no_range_is_the_default(self):
        caps = make_capabilities()
        assert caps.seed_min is None
        assert caps.seed_max is None

    def test_a_single_seed_range_is_legal(self):
        caps = make_capabilities(supports_seed=True, seed_min=7, seed_max=7)
        assert (caps.seed_min, caps.seed_max) == (7, 7)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"seed_min": 0},
            {"seed_max": 10},
            {"seed_min": 11, "seed_max": 10},
            {"supports_seed": False, "seed_min": 0, "seed_max": 10},
        ],
        ids=["only-min", "only-max", "min-above-max", "range-without-seed-support"],
    )
    def test_contradictory_declarations_are_rejected(self, overrides):
        with pytest.raises(ValidationError):
            make_capabilities(**overrides)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"seed_min": True, "seed_max": 10},
            {"seed_min": "0", "seed_max": 10},
            {"seed_min": 0, "seed_max": 10.0},
        ],
        ids=["bool-bound", "string-bound", "float-bound"],
    )
    def test_a_bound_must_be_a_plain_int(self, overrides):
        # A declaration is code: a bool, string or float bound is a typo,
        # refused instead of coerced into a range nobody wrote.
        with pytest.raises(ValidationError):
            make_capabilities(supports_seed=True, **overrides)

    def test_each_seeded_backend_declares_its_own_range(self):
        # The seeded backends wrap different samplers, whose accepted ranges
        # differ; each declares its own rather than sharing one constant.
        # Every other shipped backend declares none.
        registry = SolverRegistry.default()
        assert sorted(registry.names()) == [
            "dwave_qpu",
            "exact",
            "fujitsu_da",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "simulated_annealing",
            "simulated_bifurcation",
            "tabu",
        ]
        declared = {
            "simulated_annealing": (0, 2**31 - 1),
            "tabu": (0, 2**32 - 1),
            "simulated_bifurcation": (0, 2**32 - 1),
        }
        for name in registry.names():
            caps = registry.get(name).capabilities
            assert (caps.seed_min, caps.seed_max) == declared.get(name, (None, None)), name

    def test_simulated_annealing_range_matches_its_own_seed_check(self):
        # The backend's internal check accepts 0 <= seed < _SEED_LIMIT.
        caps = SimulatedAnnealingBackend().capabilities
        assert caps.seed_min == 0
        assert caps.seed_max == _SEED_LIMIT - 1 == 2**31 - 1

    def test_tabu_range_matches_its_own_seed_check(self):
        caps = TabuBackend().capabilities
        assert caps.seed_min == 0
        assert caps.seed_max == _TABU_SEED_LIMIT - 1 == 2**32 - 1

    def test_simulated_bifurcation_range_matches_its_own_seed_check(self):
        # Its own constant, even though it currently equals tabu's: the
        # declaration follows the backend's own check, not a neighbour's.
        caps = SimulatedBifurcationBackend().capabilities
        assert caps.seed_min == 0
        assert caps.seed_max == _SB_SEED_LIMIT - 1 == 2**32 - 1
