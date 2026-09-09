# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Use GitHub's private vulnerability reporting on the repository
("Security" tab → "Report a vulnerability"). If that is not available to
you, contact the maintainer through the e-mail address listed on their GitHub
profile.

Please include the version, a minimal reproduction, and whether the issue
affects the CLI, the MCP server, a specific backend, or the library. You will
receive an acknowledgement, and a fix or a mitigation will be coordinated with
you before any public disclosure.

## Scope reminders

- AnnealBridge's MCP server has **no authentication**. Exposing the
  streamable-http transport beyond `127.0.0.1` without an authenticating
  reverse proxy is a deployment mistake, not a vulnerability in the project.
- Remote execution is off by default. Anything that would make a remote
  backend run without `ANNEALBRIDGE_ALLOW_REMOTE=true`, bypass a resource
  limit, silently fall back to another backend, or let a vendor credential
  reach a log line, a result, or an error message **is** in scope.

The security model is described in detail in
[docs/security.md](docs/security.md).
