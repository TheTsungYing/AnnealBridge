"""The capability / availability models in ``models.capabilities`` (3a spec §7, §8)."""

import pytest
from pydantic import ValidationError

from annealbridge.models import (
    AvailabilityStatus,
    ParameterLimit,
    SolverCapabilities,
)
from annealbridge.solvers import SolverCapabilities as ReexportedCapabilities
from annealbridge.solvers.base import AvailabilityStatus as ReexportedStatus


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
