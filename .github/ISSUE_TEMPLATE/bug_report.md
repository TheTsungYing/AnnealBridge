---
name: Bug report
about: Something behaves differently from what the documentation says
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the behaviour you saw.

## What you expected

What the documentation (or the error catalog) says should happen instead.

## Minimal reproduction

The smallest problem JSON that triggers it, plus the exact command or MCP
tool call. **Remove any vendor credential before pasting.**

```json
{
  "version": "1.0",
  "variables": [],
  "objective": {"direction": "minimize", "linear_terms": []},
  "constraints": [],
  "solver": {"backend": "exact"}
}
```

```bash
annealbridge solve problem.json --json
```

## Result

Paste the full `SolveResult` (`--json`) or the CLI output, including `status`,
`errors[].code` and `warnings[].code`.

## Environment

- AnnealBridge version:
- Python version:
- OS:
- Installed extras (`mcp`, `dwave`, `all`, none):
- Backend used:
- Relevant `ANNEALBRIDGE_*` settings (values only if they are not secrets):
