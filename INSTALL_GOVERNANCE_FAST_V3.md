# NVIDIA Governance Fast v3

Replace the two included files in the repository and commit to `main`.

This mode automatically runs after the `ci` workflow completes, creates the
`nvidia-independent-review` commit status, repairs failed AI PRs up to three
times, performs an independent NIM review, squash-merges approved mergeable PRs,
deletes their branches, and dispatches the next Builder run.

After the first open AI PR receives the new status, add
`nvidia-independent-review` to the required checks for `main`.
