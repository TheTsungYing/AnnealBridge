[← Back to README](../README.md)

# AnnealBridge documentation

The [project README](../README.md) covers what AnnealBridge is, how to install
it and how to solve your first problem. These pages carry the detail: the exact
shape of the input and output documents, every error code, what each backend
needs, and the rules the project holds itself to.

## Using AnnealBridge

| Page | What it covers |
| --- | --- |
| [Problem format](problem-format.md) | The `OptimizationProblem` JSON — variables, objective, constraints, solver preferences, and every validation rule |
| [Output format](output-format.md) | `SolveResult` and everything inside it: ranked solutions, per-constraint evaluations, attempts and metadata |
| [Error reference](errors.md) | The full error catalog and the warning codes, grouped by result status |
| [CLI](cli.md) | The `annealbridge` commands, their options and their exit codes |
| [MCP server](mcp.md) | The four MCP tools, the three prompts and five resources, the stdio and streamable-HTTP transports, and host configuration |
| [Backends](backends.md) | The eight solver backends: what each one does, what it needs, and which compilation path it takes |
| [Configuration](configuration.md) | Every `ANNEALBRIDGE_*` setting, the vendor credential variables, and how they become an execution policy |

## About the project

| Page | What it covers |
| --- | --- |
| [Architecture](architecture.md) | Core versus interfaces, the enforced import boundaries, the solve pipeline, and the design principles |
| [Security model](security.md) | Remote opt-in, resource limits, credential redaction, and what is sent to a vendor |
| [Testing and CI](testing.md) | How to run the suite, what each test directory covers, and the three GitHub Actions workflows |
| [Limitations](limitations.md) | What is not supported, and what is out of scope |

## Contributing

Bug reports, backends and documentation fixes are all welcome — start with
[CONTRIBUTING.md](../CONTRIBUTING.md).
