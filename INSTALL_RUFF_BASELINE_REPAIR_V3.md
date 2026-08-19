# Ruff Baseline Repair v3

Replace `agent/tool_driven_autopilot.py` with the included file, commit to main,
and start a new Tool-driven project autopilot workflow run.

The strategy applies Ruff safe fixes, then six exact and guarded TRY004 repairs,
formats the tree, and requires both Ruff and pytest to pass before opening a PR.
