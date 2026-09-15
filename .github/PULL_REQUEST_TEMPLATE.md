## Summary

What this change does and why. Link the issue if there is one.

## Type of change

- [ ] Bug fix
- [ ] New feature / new backend
- [ ] Documentation
- [ ] Refactor (no behaviour change)
- [ ] Problem JSON contract change (say which schema version it affects)

## Checklist

- [ ] `pytest` passes locally with no new `skip` / `xfail`.
- [ ] `ruff check .` passes (F and I rules).
- [ ] `tests/architecture/` still passes (import boundaries, no backend
      names in the core, no third-party HTTP client in the core).
- [ ] No vendor credential, token, or `dwave.conf` content appears in code,
      tests, fixtures, or logs.
- [ ] New behaviour is documented (`README.md` and/or the relevant page under
      `docs/`), and `CHANGELOG.md` has an entry under *Unreleased*.
- [ ] If a new error or warning code was added, it is in the error catalog
      with a `recommended_action` and listed in `docs/errors.md`.
- [ ] If a new backend was added, it declares its capabilities (including
      credentials for redaction) and needed **no** change to the service,
      validator or interface layers (only its name in `SolverPreferences`).

## How it was tested

Commands run and their result. For remote backends, say whether the live
tests (`pytest -m remote`) were run.
