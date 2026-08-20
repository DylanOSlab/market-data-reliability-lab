# NVIDIA Governance Fast v3.1

This patch fixes the two Ruff baseline errors in `agent/nvidia_pr_supervisor.py`:

- FURB167: use `re.DOTALL` instead of `re.S`
- TRY004: use `TypeError` for an invalid response type

Replace the supervisor Python file and workflow file, commit to main, rotate any
API key accidentally committed to the repository, and start a new Builder run.
