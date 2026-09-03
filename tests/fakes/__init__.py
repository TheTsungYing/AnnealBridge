"""Test-only fakes (backends, samplers); never shipped, never production.

Kept out of ``tests/unit`` so the fakes can be shared across the unit,
architecture and scenario suites without duplication. Module names do not
match ``test_*.py`` so pytest never collects them as tests.
"""
