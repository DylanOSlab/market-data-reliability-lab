# v0.2.0 Builder Overlay

Upload the contents of this folder to the repository root.

Required repository settings:

1. Settings > Actions > General > Workflow permissions
2. Select Read and write permissions
3. Enable Allow GitHub Actions to create and approve pull requests

Then run Actions > build-v0.2.0 > Run workflow.

Expected result: an automation branch and a pull request titled
"Add official BLS CPI dataset for v0.2.0".

Do not merge until the generated golden CSV and provenance are reviewed.
