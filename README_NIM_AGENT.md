# NVIDIA NIM GitHub maintenance agent

This starter adds a repository-scoped automation that asks NVIDIA NIM for one small
maintenance change, validates file paths, creates a branch, commits, pushes, and opens a PR.
It does not auto-merge in v1.

## Setup

1. Copy `agent/agent.py` and `.github/workflows/nim-agent.yml` into the repository.
2. At https://build.nvidia.com, sign in, choose a model, and generate an API key.
3. In GitHub open **Settings > Secrets and variables > Actions > New repository secret**.
4. Name the secret `NVIDIA_API_KEY` and paste the key.
5. Open **Settings > Actions > General > Workflow permissions**.
6. Select **Read and write permissions** and enable **Allow GitHub Actions to create and approve pull requests**.
7. Open **Actions > NVIDIA NIM maintenance agent > Run workflow**.

## Optional model

Create an Actions repository variable named `NVIDIA_MODEL`. The default is
`meta/llama-3.1-70b-instruct`. Use the exact model ID shown by NVIDIA Build.

## Guardrails included

- Hard-coded repository allowlist
- Maximum five files per run
- Blocks `.github`, `.git`, `.env`, binary files, and parent-directory traversal
- Maximum 500 lines per replacement file
- One PR per run
- No repository settings, secrets, force push, deletion, or automatic merge

## Next phase

After the first PR succeeds, add test execution, failure repair, issue selection, and
conditional auto-merge. Do not enable auto-merge before verifying the model ID and the
repository's existing test command.
