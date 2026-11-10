# Feedback task starting environment

Upstream: https://github.com/alisaifee/limits
Upstream commit: `0492c4fbef45c0f66c88c5feb00cb1270af238fa`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= --noconftest tests/test_limits.py tests/test_ratelimit_parser.py tests/test_limit_granularities.py -o asyncio_default_fixture_loop_scope=function
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `20 passed in 0.04s`
Whole-command wall time: 0.208 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
