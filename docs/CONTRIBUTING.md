# Contributing

Thanks for contributing to ariran.

## Local setup
1. Copy `.env.example` to `.env` and set required values.
2. Run `make install`.
3. Run `make dev-up` then `make init-db`.

## Quality gates
1. Run `make check` for lint + type checks.
2. Run `make test` (or `make test-unit` / `make test-integration`).

## Pull request expectations
1. Keep changes scoped and documented.
2. Add or update tests for behavioral changes.
3. Ensure all checks pass before requesting review.
