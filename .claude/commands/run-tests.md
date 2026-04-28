---
description: Run the test suite via the test-runner sub-agent and report a compact summary.
---

Delegate to the `test-runner` sub-agent.

If the user named a specific test or module after the command, pass that scope to the sub-agent. Otherwise, run the default suite.

When the sub-agent returns:
- If all tests pass and ruff is clean: say so in one line and stop.
- If anything failed: report the verdict and ask whether to investigate, fix, or skip for now. Do NOT start fixing on your own — that decision is the user's.
