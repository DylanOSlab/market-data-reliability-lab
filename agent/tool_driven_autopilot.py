import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_REPO = os.getenv("ALLOWED_REPO", "DylanOSlab/market-data-reliability-lab")
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")
GH_TOKEN = os.getenv("GH_TOKEN", "")

ALLOWED_CHANGED_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "fixtures/",
    "provenance/",
    "README.md",
    "INSTALL_NEXT.md",
    "pyproject.toml",
)
BLOCKED_CHANGED_PREFIXES = (
    ".github/",
    ".git/",
    ".automation/",
    ".env",
)
MAX_CHANGED_FILES = 20
MAX_CHANGED_LINES = 1200


def run(*args, check=True, timeout=600):
    result = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    print(f"$ {' '.join(args)}", flush=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(args)}"
        )
    return result


def ensure_repository_scope():
    actual = os.getenv("GITHUB_REPOSITORY", "")
    if actual != ALLOWED_REPO:
        raise SystemExit(
            f"Repository blocked: received {actual!r}, expected {ALLOWED_REPO!r}."
        )


def install_project():
    run(sys.executable, "-m", "pip", "install", "--upgrade", "pip")
    result = run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".[dev]",
        check=False,
    )
    if result.returncode != 0:
        run(sys.executable, "-m", "pip", "install", "-e", ".")
    run(sys.executable, "-m", "pip", "install", "ruff", "pytest")


def baseline_tests():
    result = run(sys.executable, "-m", "pytest", "-q", check=False)
    if result.returncode != 0:
        create_or_update_issue(
            "[autopilot] Baseline tests are failing",
            "The tool-driven autopilot stopped because the main branch test suite "
            "is already failing. Review the latest workflow run before enabling "
            "automatic changes.\n\n"
            f"Workflow: {workflow_url()}",
            labels="autopilot,human-input-required",
        )
        raise RuntimeError("Baseline pytest failed; no automated change was attempted.")


def snapshot():
    return run("git", "status", "--porcelain", check=True).stdout


def changed_files():
    output = run("git", "diff", "--name-only").stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def validate_changed_paths(paths):
    if not paths:
        return
    if len(paths) > MAX_CHANGED_FILES:
        raise RuntimeError(f"Too many changed files: {len(paths)}")
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized.startswith(BLOCKED_CHANGED_PREFIXES):
            raise RuntimeError(f"Protected path changed: {normalized}")
        if not normalized.startswith(ALLOWED_CHANGED_PREFIXES):
            raise RuntimeError(f"Path outside allowlist changed: {normalized}")


def validate_diff_size():
    result = run("git", "diff", "--numstat")
    total = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0] == "-" or parts[1] == "-":
            raise RuntimeError("Binary change detected and blocked.")
        total += int(parts[0]) + int(parts[1])
    if total > MAX_CHANGED_LINES:
        raise RuntimeError(f"Diff is too large: {total} changed lines")


def strategy_ruff_fix():
    before = snapshot()
    run(sys.executable, "-m", "ruff", "check", ".", "--fix", check=False)
    run(sys.executable, "-m", "ruff", "format", ".", check=False)
    return before != snapshot()


def strategy_regenerate_cases():
    script = Path("scripts/generate_cases.py")
    if not script.exists():
        return False
    before = snapshot()
    run(sys.executable, str(script), check=False)
    return before != snapshot()


def strategy_normalize_text_files():
    before = snapshot()
    candidates = []
    for prefix in ("src", "tests", "scripts", "provenance"):
        root = Path(prefix)
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    candidates.extend(path for path in (Path("README.md"), Path("INSTALL_NEXT.md")) if path.exists())

    for path in candidates:
        if path.suffix not in {".py", ".md", ".json", ".csv", ".toml", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = [line.rstrip() for line in text.splitlines()]
        normalized = "\n".join(lines) + "\n"
        if normalized != text:
            path.write_text(normalized, encoding="utf-8")
    return before != snapshot()


STRATEGIES = (
    ("ruff-fix-and-format", strategy_ruff_fix, "Apply safe Ruff fixes and formatting"),
    ("regenerate-deterministic-cases", strategy_regenerate_cases, "Regenerate deterministic failure cases"),
    ("normalize-text-files", strategy_normalize_text_files, "Normalize supported text files"),
)


def attempt_safe_change():
    for key, function, description in STRATEGIES:
        print(f"Trying strategy: {key}", flush=True)
        run("git", "reset", "--hard", "HEAD")
        run("git", "clean", "-fd")
        changed = function()
        paths = changed_files()
        if not changed or not paths:
            print(f"Strategy produced no change: {key}", flush=True)
            continue

        validate_changed_paths(paths)
        validate_diff_size()
        run("git", "diff", "--check")
        result = run(sys.executable, "-m", "pytest", "-q", check=False)
        if result.returncode != 0:
            print(f"Strategy failed tests and will be reverted: {key}", flush=True)
            continue

        return {
            "key": key,
            "description": description,
            "paths": paths,
        }

    run("git", "reset", "--hard", "HEAD")
    run("git", "clean", "-fd")
    return None


def workflow_url():
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repository = os.getenv("GITHUB_REPOSITORY", ALLOWED_REPO)
    run_id = os.getenv("GITHUB_RUN_ID", "")
    return f"{server}/{repository}/actions/runs/{run_id}"


def issue_number_by_title(title):
    result = run(
        "gh", "issue", "list",
        "--repo", ALLOWED_REPO,
        "--state", "open",
        "--limit", "100",
        "--json", "number,title",
        "--jq", f'.[] | select(.title == {json.dumps(title)}) | .number',
        check=False,
    )
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else ""


def create_or_update_issue(title, body, labels="autopilot"):
    number = issue_number_by_title(title)
    if number:
        run("gh", "issue", "comment", number, "--repo", ALLOWED_REPO, "--body", body)
        return
    command = [
        "gh", "issue", "create", "--repo", ALLOWED_REPO,
        "--title", title, "--body", body,
    ]
    if labels:
        command.extend(["--label", labels])
    run(*command)


def publish_change(change):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fingerprint = hashlib.sha1(
        (change["key"] + "\n" + "\n".join(change["paths"])).encode("utf-8")
    ).hexdigest()[:7]
    branch = f"autopilot/{stamp}-{change['key']}-{fingerprint}"

    run("git", "checkout", "-b", branch)
    run("git", "add", "--", *change["paths"])
    run("git", "diff", "--cached", "--check")
    run("git", "commit", "-m", change["description"].lower())
    run("git", "push", "--set-upstream", "origin", branch)

    body = (
        "## Summary\n\n"
        f"{change['description']}.\n\n"
        "## Files\n\n"
        + "\n".join(f"- `{path}`" for path in change["paths"])
        + "\n\n## Validation\n\n"
        "- Baseline tests passed before the change.\n"
        "- Tests passed again after the change.\n"
        "- Changed paths and diff size were checked by the autopilot.\n\n"
        "This pull request was produced by deterministic repository tools. "
        "No generative model output was applied directly to source code."
    )
    url = run(
        "gh", "pr", "create",
        "--base", DEFAULT_BRANCH,
        "--head", branch,
        "--title", change["description"],
        "--body", body,
    ).stdout
    print(f"Pull request created: {url}", flush=True)


def report_no_safe_change():
    title = "[autopilot] No deterministic maintenance change available"
    body = (
        "The tool-driven autopilot completed all configured safe strategies but "
        "none produced a test-passing repository change.\n\n"
        "This is a normal result, not a repository failure. Add a new deterministic "
        "strategy or provide a stronger coding model before attempting feature work.\n\n"
        f"Workflow: {workflow_url()}"
    )
    create_or_update_issue(title, body)


def main():
    ensure_repository_scope()
    if not GH_TOKEN:
        raise SystemExit("GH_TOKEN is required.")

    install_project()
    baseline_tests()
    change = attempt_safe_change()
    if change is None:
        report_no_safe_change()
        print("No safe deterministic change was available.", flush=True)
        return
    publish_change(change)


if __name__ == "__main__":
    main()
