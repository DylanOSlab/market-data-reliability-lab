import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ALLOWED_REPO = os.getenv(
    "ALLOWED_REPO",
    "DylanOSlab/market-data-reliability-lab",
)

DEFAULT_BRANCH = os.getenv(
    "DEFAULT_BRANCH",
    "main",
)

LLAMA_CLI = os.getenv(
    "LLAMA_CLI",
    "./llama.cpp/build/bin/llama-cli",
)

MODEL_SPEC = os.getenv(
    "MODEL_SPEC",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M",
)

MAX_CONTEXT_CHARACTERS = 6_000
MAX_CONTEXT_FILES = 12
MAX_CHANGED_FILES = 1
MAX_GENERATED_LINES_PER_FILE = 200
MAX_TOTAL_GENERATED_CHARACTERS = 10_000
MAX_GENERATED_TOKENS = 600
MODEL_CONTEXT_SIZE = 4_096
MODEL_TIMEOUT_SECONDS = 1_200
HEARTBEAT_SECONDS = 30

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".bin",
    ".pyc",
    ".pyd",
    ".woff",
    ".woff2",
    ".ttf",
}

BLOCKED_PATH_PREFIXES = (
    ".github/",
    ".git/",
    ".automation/",
    ".env",
)

BLOCKED_FILE_NAMES = {
    "package-lock.json",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
}

PRIORITY_PATH_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "fixtures/",
    "provenance/",
)

SYSTEM_PROMPT = """
You are an autonomous software maintenance agent for one public experimental repository.

Choose exactly one small, useful, low-risk change that advances the project.
Prefer a missing regression test, deterministic test coverage, a small confirmed defect,
input validation, error handling, fixture or provenance validation, or a documentation correction.

Return only one valid JSON object with this exact structure:
{
  "summary": "short explanation",
  "branch": "ai/short-lowercase-slug",
  "commit_message": "short commit message",
  "pr_title": "pull request title",
  "pr_body": "pull request body",
  "files": [
    {
      "path": "relative/path",
      "content": "complete replacement content"
    }
  ]
}

Rules:
- Change exactly one text file.
- Return complete replacement content, not a diff.
- Keep the generated file below 200 lines.
- Never modify .github, .git, .automation, environment files, secrets, tokens,
  permissions, billing, repository settings, workflows, Actions configuration,
  security policies, dependency lock files, or binary files.
- Never delete files.
- Never use an absolute path or parent-directory path.
- Never claim tests passed.
- Never invent files, functions, dependencies, APIs, or existing behavior.
- Prefer modifying an existing file.
- Keep the change easy to review and revert.
""".strip()


def run_command(*args, check=True):
    """Run a command and return its standard output."""

    process = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if check and process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"Exit code: {process.returncode}\n"
            f"Standard output:\n{process.stdout}\n"
            f"Standard error:\n{process.stderr}"
        )

    return process.stdout.strip()


def is_safe_context_file(path):
    """Return True when a file is safe to include in model context."""

    if not path.is_file():
        return False

    normalized = path.as_posix()

    if any(normalized.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        return False

    if path.name in BLOCKED_FILE_NAMES:
        return False

    if path.suffix.lower() in BINARY_SUFFIXES:
        return False

    try:
        return path.stat().st_size <= 16_000
    except OSError:
        return False


def context_sort_key(path):
    """Prioritize source, tests, fixtures, and project metadata."""

    normalized = path.as_posix()

    for index, prefix in enumerate(PRIORITY_PATH_PREFIXES):
        if normalized.startswith(prefix):
            return index, normalized

    if normalized == "pyproject.toml":
        return 10, normalized

    if normalized == "README.md":
        return 11, normalized

    if normalized == "INSTALL_NEXT.md":
        return 12, normalized

    return 20, normalized


def build_repository_context():
    """Build a bounded repository snapshot for the local model."""

    candidates = [
        path
        for path in Path(".").rglob("*")
        if is_safe_context_file(path)
    ]

    candidates.sort(key=context_sort_key)
    sections = []
    current_size = 0

    for path in candidates[:MAX_CONTEXT_FILES]:
        try:
            file_content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        section = (
            f"\n--- FILE: {path.as_posix()} ---\n"
            f"{file_content}\n"
        )

        remaining = MAX_CONTEXT_CHARACTERS - current_size

        if remaining <= 0:
            break

        if len(section) > remaining:
            if remaining >= 400:
                sections.append(section[:remaining])
            break

        sections.append(section)
        current_size += len(section)

    return "".join(sections)


def build_model_prompt(repository_context):
    """Create the ChatML prompt sent to Qwen."""

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Repository: {ALLOWED_REPO}\n\n"
        "Use only the repository snapshot below. "
        "Choose one small task and return only JSON.\n\n"
        f"{repository_context}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_local_model(prompt):
    """Run Qwen through llama.cpp and emit heartbeat logs."""

    cli_path = Path(LLAMA_CLI)

    if not cli_path.exists():
        raise FileNotFoundError(
            f"llama.cpp executable was not found: {LLAMA_CLI}"
        )

    thread_count = min(
        4,
        max(1, os.cpu_count() or 1),
    )

    command = [
        str(cli_path),
        "-hf",
        MODEL_SPEC,
        "-p",
        prompt,
        "-n",
        str(MAX_GENERATED_TOKENS),
        "-c",
        str(MODEL_CONTEXT_SIZE),
        "-t",
        str(thread_count),
        "--temp",
        "0.1",
        "--top-p",
        "0.9",
        "--repeat-penalty",
        "1.05",
        "--no-display-prompt",
        "--no-mmap",
        "--simple-io",
        "--single-turn",
    ]

    print("Starting local Qwen inference.", flush=True)
    print(f"Model: {MODEL_SPEC}", flush=True)
    print(f"Prompt size: {len(prompt)} characters", flush=True)
    print(f"Maximum generated tokens: {MAX_GENERATED_TOKENS}", flush=True)
    print(f"Model context size: {MODEL_CONTEXT_SIZE}", flush=True)
    print(f"CPU threads: {thread_count}", flush=True)

    started_at = time.monotonic()

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    while process.poll() is None:
        elapsed_seconds = int(time.monotonic() - started_at)

        if elapsed_seconds >= MODEL_TIMEOUT_SECONDS:
            process.kill()
            stdout, stderr = process.communicate()

            raise RuntimeError(
                "Local model exceeded the configured inference timeout.\n"
                f"Partial output:\n{stdout}\n"
                f"Errors:\n{stderr}"
            )

        print(
            "Local inference is still running: "
            f"{elapsed_seconds} seconds elapsed.",
            flush=True,
        )

        time.sleep(HEARTBEAT_SECONDS)

    stdout, stderr = process.communicate()

    if stderr:
        print(stderr, file=sys.stderr, flush=True)

    if process.returncode != 0:
        raise RuntimeError(
            "Local model execution failed.\n"
            f"Exit code: {process.returncode}\n"
            f"Standard output:\n{stdout}\n"
            f"Standard error:\n{stderr}"
        )

    output = stdout.strip()

    if not output:
        raise ValueError("The local model returned an empty response.")

    elapsed_seconds = int(time.monotonic() - started_at)

    print(
        f"Local model completed after {elapsed_seconds} seconds.",
        flush=True,
    )
    print(
        f"Local model returned {len(output)} characters.",
        flush=True,
    )

    return output


def extract_json_object(model_output):
    """Extract the first complete JSON object from model output."""

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        model_output.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])

            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "The model did not return valid JSON.\n"
        f"Raw output:\n{model_output}"
    )


def validate_relative_path(path_value):
    """Validate a generated repository-relative path."""

    if not isinstance(path_value, str):
        raise ValueError("Every generated file must have a string path.")

    normalized = path_value.replace("\\", "/").strip()

    if not normalized:
        raise ValueError("Generated file path cannot be empty.")

    if normalized.startswith("/"):
        raise ValueError(f"Absolute path blocked: {normalized}")

    if re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError(f"Windows absolute path blocked: {normalized}")

    if any(normalized.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        raise ValueError(f"Protected path blocked: {normalized}")

    path_object = Path(normalized)

    if ".." in path_object.parts:
        raise ValueError(f"Parent-directory path blocked: {normalized}")

    if path_object.name in BLOCKED_FILE_NAMES:
        raise ValueError(f"Lock file blocked: {normalized}")

    if path_object.suffix.lower() in BINARY_SUFFIXES:
        raise ValueError(f"Binary file blocked: {normalized}")

    return normalized


def validate_generated_plan(plan):
    """Validate model output before changing repository files."""

    if not isinstance(plan, dict):
        raise ValueError("Generated plan must be a JSON object.")

    for field in (
        "summary",
        "branch",
        "commit_message",
        "pr_title",
        "pr_body",
    ):
        value = plan.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid field: {field}")

    branch = plan["branch"].strip()

    if not re.fullmatch(
        r"ai/[a-z0-9][a-z0-9._-]{2,60}",
        branch,
    ):
        raise ValueError(
            "Generated branch must match ai/<short-lowercase-slug>."
        )

    forbidden_placeholder_values = {
        "ai/short-lowercase-slug",
        "short commit message",
        "pull request title",
        "pull request body",
        "short explanation",
    }

    for field in (
        "summary",
        "branch",
        "commit_message",
        "pr_title",
        "pr_body",
    ):
        if plan[field].strip().lower() in forbidden_placeholder_values:
            raise ValueError(
                f"Model returned placeholder content in field: {field}"
            )

    files = plan.get("files")

    if not isinstance(files, list) or len(files) != MAX_CHANGED_FILES:
        raise ValueError("The generated plan must modify exactly one file.")

    total_characters = 0

    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each files entry must be an object.")

        normalized_path = validate_relative_path(item.get("path"))
        generated_content = item.get("content")

        if isinstance(generated_content, str) and generated_content.strip().lower() in {
            "complete replacement content",
            "complete replacement file content",
        }:
            raise ValueError(
                f"Model returned placeholder file content: {normalized_path}"
            )

        if not isinstance(generated_content, str) or not generated_content.strip():
            raise ValueError(
                f"Generated content is invalid: {normalized_path}"
            )

        line_count = len(generated_content.splitlines())

        if line_count > MAX_GENERATED_LINES_PER_FILE:
            raise ValueError(
                f"Generated file has {line_count} lines: {normalized_path}"
            )

        total_characters += len(generated_content)
        item["path"] = normalized_path

    if total_characters > MAX_TOTAL_GENERATED_CHARACTERS:
        raise ValueError("Generated content exceeds the total size limit.")


def ensure_branch_does_not_exist(branch):
    """Prevent accidentally reusing an existing remote branch."""

    process = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            "origin",
            branch,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if process.returncode == 0:
        raise ValueError(f"Generated branch already exists: {branch}")


def apply_generated_plan(plan):
    """Create a branch, commit the generated file, and open a PR."""

    branch = plan["branch"].strip()
    ensure_branch_does_not_exist(branch)

    run_command("git", "checkout", "-b", branch)

    changed_paths = []

    for item in plan["files"]:
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["content"], encoding="utf-8")
        changed_paths.append(item["path"])

    run_command("git", "add", "--", *changed_paths)

    if not run_command("git", "status", "--porcelain"):
        raise ValueError("The generated plan produced no repository changes.")

    subprocess.run(
        ["git", "diff", "--cached", "--check"],
        check=True,
    )

    print(
        run_command("git", "diff", "--cached", "--stat"),
        flush=True,
    )

    run_command(
        "git",
        "commit",
        "-m",
        plan["commit_message"].strip(),
    )

    run_command(
        "git",
        "push",
        "--set-upstream",
        "origin",
        branch,
    )

    pull_request_body = (
        plan["pr_body"].strip()
        + "\n\nGenerated by a local Qwen Coder model running inside GitHub Actions."
        + "\n\nNo external model API or API key was used."
    )

    pull_request_url = run_command(
        "gh",
        "pr",
        "create",
        "--base",
        DEFAULT_BRANCH,
        "--head",
        branch,
        "--title",
        plan["pr_title"].strip(),
        "--body",
        pull_request_body,
    )

    print(
        f"Pull request created: {pull_request_url}",
        flush=True,
    )


def main():
    """Run one repository-scoped maintenance cycle."""

    actual_repository = os.getenv("GITHUB_REPOSITORY", "")

    if actual_repository != ALLOWED_REPO:
        raise SystemExit(
            "Repository scope check failed. "
            f"Received {actual_repository!r}; "
            f"expected {ALLOWED_REPO!r}."
        )

    repository_context = build_repository_context()

    if not repository_context.strip():
        raise SystemExit("No eligible repository files were found.")

    print(
        f"Repository context size: {len(repository_context)} characters",
        flush=True,
    )

    prompt = build_model_prompt(repository_context)
    model_output = run_local_model(prompt)

    print("Raw local model output:", flush=True)
    print(model_output, flush=True)

    plan = extract_json_object(model_output)
    validate_generated_plan(plan)

    print("Validated maintenance plan:", flush=True)
    print(
        json.dumps(plan, indent=2, ensure_ascii=False),
        flush=True,
    )

    apply_generated_plan(plan)


if __name__ == "__main__":
    main()
