import json
import os
import re
import subprocess
import sys
from pathlib import Path


ALLOWED_REPO = os.getenv(
    "ALLOWED_REPO",
    "DylanOSlab/market-data-reliability-lab",
)

LLAMA_CLI = os.getenv(
    "LLAMA_CLI",
    "./llama.cpp/build/bin/llama-cli",
)

MODEL_SPEC = os.getenv(
    "MODEL_SPEC",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
)

DEFAULT_BRANCH = os.getenv(
    "DEFAULT_BRANCH",
    "main",
)

MAX_CONTEXT_CHARACTERS = 24_000
MAX_CONTEXT_FILES = 40
MAX_CHANGED_FILES = 3
MAX_GENERATED_LINES_PER_FILE = 400
MAX_TOTAL_GENERATED_CHARACTERS = 40_000


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
You are an autonomous software maintenance agent for one public,
disposable test repository.

Choose exactly one small, useful, low-risk maintenance change.

Preferred work, in order:

1. Add a missing regression test.
2. Improve deterministic test coverage.
3. Fix a small confirmed defect.
4. Improve input validation.
5. Improve error handling.
6. Improve fixture or provenance validation.
7. Correct documentation that clearly disagrees with source code.

Do not attempt a large refactor.

Return only one valid JSON object.

Required JSON structure:

{
  "summary": "short explanation",
  "branch": "ai/short-lowercase-slug",
  "commit_message": "short commit message",
  "pr_title": "pull request title",
  "pr_body": "pull request body",
  "files": [
    {
      "path": "relative/repository/path",
      "content": "complete replacement file content"
    }
  ]
}

Rules:

- Change one to three files only.
- Return complete replacement content, not a diff.
- Keep each generated file below 400 lines.
- Never modify anything under .github.
- Never modify anything under .git.
- Never modify anything under .automation.
- Never modify environment files.
- Never modify secrets, credentials, tokens, permissions, billing,
  repository settings, workflows, Actions configuration, or security
  policies.
- Never modify lock files.
- Never modify binary files.
- Never delete files.
- Never use an absolute path.
- Never use a parent-directory path.
- Do not claim that tests passed.
- Do not invent files, functions, dependencies, or existing behavior.
- Prefer modifying existing files.
- Any new behavior should include a focused test when practical.
- Keep the change easy to review and easy to revert.
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
    """Decide whether a repository file can be sent to the model."""

    if not path.is_file():
        return False

    normalized = path.as_posix()

    if any(
        normalized.startswith(prefix)
        for prefix in BLOCKED_PATH_PREFIXES
    ):
        return False

    if path.name in BLOCKED_FILE_NAMES:
        return False

    if path.suffix.lower() in BINARY_SUFFIXES:
        return False

    try:
        if path.stat().st_size > 30_000:
            return False
    except OSError:
        return False

    return True


def context_sort_key(path):
    """Prioritize source, tests, and project configuration."""

    normalized = path.as_posix()

    for index, prefix in enumerate(PRIORITY_PATH_PREFIXES):
        if normalized.startswith(prefix):
            return index, normalized

    if normalized == "pyproject.toml":
        return 10, normalized

    if normalized == "README.md":
        return 11, normalized

    return 20, normalized


def build_repository_context():
    """Build a bounded repository snapshot for the local model."""

    candidates = [
        path
        for path in Path(".").rglob("*")
        if is_safe_context_file(path)
    ]

    candidates.sort(key=context_sort_key)
    candidates = candidates[:MAX_CONTEXT_FILES]

    sections = []
    current_size = 0

    for path in candidates:
        try:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        section = (
            f"\n--- FILE: {path.as_posix()} ---\n"
            f"{content}\n"
        )

        if current_size + len(section) > MAX_CONTEXT_CHARACTERS:
            remaining = MAX_CONTEXT_CHARACTERS - current_size

            if remaining > 500:
                sections.append(section[:remaining])

            break

        sections.append(section)
        current_size += len(section)

    return "".join(sections)


def build_model_prompt(repository_context):
    """Create the complete instruction passed to llama.cpp."""

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Repository: {ALLOWED_REPO}\n\n"
        "Review the repository snapshot below and choose exactly one "
        "small maintenance task. Return only the required JSON object."
        "\n\n"
        f"{repository_context}"
        "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_local_model(prompt):
    """Run Qwen Coder through llama.cpp."""

    cli_path = Path(LLAMA_CLI)

    if not cli_path.exists():
        raise FileNotFoundError(
            f"llama.cpp executable was not found: {LLAMA_CLI}"
        )

    command = [
        str(cli_path),
        "-hf",
        MODEL_SPEC,
        "-p",
        prompt,
        "-n",
        "5000",
        "-c",
        "16384",
        "-t",
        str(max(1, os.cpu_count() or 1)),
        "--temp",
        "0.1",
        "--top-p",
        "0.9",
        "--repeat-penalty",
        "1.05",
        "--no-display-prompt",
        "--no-mmap",
    ]

    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=900,
    )

    print(
        process.stderr,
        file=sys.stderr,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Local model execution failed.\n"
            f"Exit code: {process.returncode}\n"
            f"Output:\n{process.stdout}\n"
            f"Errors:\n{process.stderr}"
        )

    if not process.stdout.strip():
        raise ValueError(
            "The local model returned an empty response."
        )

    return process.stdout.strip()


def extract_json_object(model_output):
    """Extract the first complete JSON object from model output."""

    cleaned = model_output.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                cleaned[index:]
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "The local model did not return a valid JSON object.\n\n"
        f"Raw model output:\n{model_output}"
    )


def validate_relative_path(path_value):
    """Validate a generated repository-relative path."""

    if not isinstance(path_value, str):
        raise ValueError(
            "Every generated file must have a string path."
        )

    normalized = path_value.replace("\\", "/").strip()

    if not normalized:
        raise ValueError(
            "Generated file path cannot be empty."
        )

    if normalized.startswith("/"):
        raise ValueError(
            f"Absolute path blocked: {normalized}"
        )

    if re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError(
            f"Windows absolute path blocked: {normalized}"
        )

    if any(
        normalized.startswith(prefix)
        for prefix in BLOCKED_PATH_PREFIXES
    ):
        raise ValueError(
            f"Protected path blocked: {normalized}"
        )

    path_object = Path(normalized)

    if ".." in path_object.parts:
        raise ValueError(
            f"Parent-directory path blocked: {normalized}"
        )

    if path_object.name in BLOCKED_FILE_NAMES:
        raise ValueError(
            f"Lock file blocked: {normalized}"
        )

    if path_object.suffix.lower() in BINARY_SUFFIXES:
        raise ValueError(
            f"Binary file blocked: {normalized}"
        )

    return normalized


def validate_generated_plan(plan):
    """Validate model output before changing repository files."""

    if not isinstance(plan, dict):
        raise ValueError(
            "Generated plan must be a JSON object."
        )

    required_text_fields = (
        "summary",
        "branch",
        "commit_message",
        "pr_title",
        "pr_body",
    )

    for field in required_text_fields:
        value = plan.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Missing or invalid generated field: {field}"
            )

    branch = plan["branch"].strip()

    if not re.fullmatch(
        r"ai/[a-z0-9][a-z0-9._-]{2,60}",
        branch,
    ):
        raise ValueError(
            "Generated branch must match ai/<short-lowercase-slug>."
        )

    files = plan.get("files")

    if not isinstance(files, list):
        raise ValueError(
            "Generated files value must be an array."
        )

    if not 1 <= len(files) <= MAX_CHANGED_FILES:
        raise ValueError(
            "The generated plan must modify between one and "
            f"{MAX_CHANGED_FILES} files."
        )

    seen_paths = set()
    total_characters = 0

    for item in files:
        if not isinstance(item, dict):
            raise ValueError(
                "Each generated files entry must be an object."
            )

        normalized_path = validate_relative_path(
            item.get("path")
        )

        content = item.get("content")

        if not isinstance(content, str):
            raise ValueError(
                f"Generated content must be text: {normalized_path}"
            )

        if not content.strip():
            raise ValueError(
                f"Generated content cannot be empty: {normalized_path}"
            )

        if normalized_path in seen_paths:
            raise ValueError(
                f"Duplicate generated path: {normalized_path}"
            )

        seen_paths.add(normalized_path)

        line_count = len(content.splitlines())

        if line_count > MAX_GENERATED_LINES_PER_FILE:
            raise ValueError(
                f"Generated file is too large: {normalized_path} "
                f"has {line_count} lines."
            )

        total_characters += len(content)

        item["path"] = normalized_path

    if total_characters > MAX_TOTAL_GENERATED_CHARACTERS:
        raise ValueError(
            "Generated content exceeds the total size limit."
        )


def ensure_branch_does_not_exist(branch):
    """Prevent accidentally reusing an existing branch."""

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
        raise ValueError(
            f"Generated branch already exists: {branch}"
        )


def apply_generated_plan(plan):
    """Create a branch, commit the generated files, and open a PR."""

    branch = plan["branch"].strip()

    ensure_branch_does_not_exist(branch)

    run_command(
        "git",
        "checkout",
        "-b",
        branch,
    )

    changed_paths = []

    for item in plan["files"]:
        path = Path(item["path"])

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            item["content"],
            encoding="utf-8",
        )

        changed_paths.append(
            item["path"]
        )

    run_command(
        "git",
        "add",
        "--",
        *changed_paths,
    )

    status = run_command(
        "git",
        "status",
        "--porcelain",
    )

    if not status:
        raise ValueError(
            "The generated plan produced no repository changes."
        )

    subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--check",
        ],
        check=True,
    )

    print("Generated staged diff:")
    print(
        run_command(
            "git",
            "diff",
            "--cached",
            "--stat",
        )
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
        + "\n\n"
        + "Generated by a local Qwen Coder model running inside "
        + "GitHub Actions."
        + "\n\n"
        + "No external model API or API key was used."
    )

    run_command(
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


def main():
    """Run one repository-scoped maintenance cycle."""

    actual_repository = os.getenv(
        "GITHUB_REPOSITORY",
        "",
    )

    if actual_repository != ALLOWED_REPO:
        raise SystemExit(
            "Repository scope check failed. "
            f"Received {actual_repository!r}; "
            f"expected {ALLOWED_REPO!r}."
        )

    repository_context = build_repository_context()

    if not repository_context.strip():
        raise SystemExit(
            "No eligible repository files were found."
        )

    print(
        "Repository context size:",
        len(repository_context),
        "characters",
    )

    prompt = build_model_prompt(
        repository_context
    )

    model_output = run_local_model(
        prompt
    )

    print("Raw local model output:")
    print(model_output)

    plan = extract_json_object(
        model_output
    )

    validate_generated_plan(
        plan
    )

    print("Validated maintenance plan:")
    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )
    )

    apply_generated_plan(
        plan
    )


if __name__ == "__main__":
    main()
