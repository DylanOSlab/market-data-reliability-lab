import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


ALLOWED_REPO = os.getenv(
    "ALLOWED_REPO",
    "DylanOSlab/market-data-reliability-lab",
)

GITHUB_MODEL = os.getenv(
    "GITHUB_MODEL",
    "openai/gpt-4.1-mini",
)

GITHUB_MODELS_URL = (
    "https://models.github.ai/inference/chat/completions"
)

MAX_CONTEXT_CHARACTERS = 120_000
MAX_FILES_IN_CONTEXT = 80
MAX_CHANGED_FILES = 5
MAX_LINES_PER_FILE = 500


SYSTEM_PROMPT = """
You are an autonomous software maintenance agent for one disposable
test repository.

Your goal is to create exactly one small, useful, low-risk change.

Prefer these task types:

1. Add missing regression tests.
2. Improve deterministic tests.
3. Improve input validation.
4. Improve error handling.
5. Fix a small confirmed defect.
6. Improve documentation when it does not match the code.
7. Improve fixture or provenance validation.
8. Add a small reliability improvement.

Return only valid JSON.

The JSON object must contain these keys:

- summary
- branch
- commit_message
- pr_title
- pr_body
- files

The files value must be an array of objects. Each file object must
contain:

- path
- content

The content value must contain the complete replacement content of the
file, not a diff.

Rules:

- Change between one and five files.
- Keep each replacement file under 500 lines.
- Do not modify files inside .github.
- Do not modify .git files.
- Do not modify environment files.
- Do not modify secrets, permissions, billing, repository settings,
  workflows, or security policies.
- Do not modify lock files.
- Do not modify binary files.
- Do not delete files.
- Do not use parent-directory paths.
- Do not include test results unless the supplied repository context
  proves those results.
- Do not invent source files, functions, dependencies, or behavior.
- Prefer small changes that can be reviewed and reverted easily.
- The branch must use this format: ai/<short-slug>.
""".strip()


BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".zip",
    ".gz",
    ".tar",
    ".pdf",
    ".exe",
    ".dll",
    ".so",
    ".bin",
    ".pyc",
}


BLOCKED_PATH_PREFIXES = (
    ".github/",
    ".git/",
    ".env",
)


def run_command(*args):
    """Run a command and return standard output."""

    return subprocess.check_output(
        args,
        text=True,
    ).strip()


def is_allowed_context_file(path):
    """Return True when a repository file is safe for model context."""

    if not path.is_file():
        return False

    normalized = path.as_posix()

    if ".git" in path.parts:
        return False

    if normalized.startswith(".github/"):
        return False

    if path.suffix.lower() in BINARY_SUFFIXES:
        return False

    if path.stat().st_size > 40_000:
        return False

    return True


def build_repository_context():
    """Read a bounded selection of repository text files."""

    allowed_files = []

    for path in Path(".").rglob("*"):
        if is_allowed_context_file(path):
            allowed_files.append(path)

    allowed_files.sort(
        key=lambda item: item.as_posix()
    )

    allowed_files = allowed_files[:MAX_FILES_IN_CONTEXT]

    chunks = []

    for path in allowed_files:
        try:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            chunks.append(
                f"\n--- FILE: {path.as_posix()} ---\n"
                f"{content}"
            )
        except OSError:
            continue

    context = "".join(chunks)

    return context[:MAX_CONTEXT_CHARACTERS]


def remove_markdown_fence(text):
    """Remove an optional JSON Markdown code fence."""

    cleaned = text.strip()

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

    return cleaned.strip()


def call_github_models(prompt):
    """Call GitHub Models using the workflow GITHUB_TOKEN."""

    token = os.environ["GITHUB_MODELS_TOKEN"]

    request_body = {
        "model": GITHUB_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
        "stream": False,
    }

    encoded_body = json.dumps(
        request_body
    ).encode("utf-8")

    request = urllib.request.Request(
        GITHUB_MODELS_URL,
        data=encoded_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=180,
        ) as response:
            response_data = json.load(response)

    except urllib.error.HTTPError as error:
        error_body = error.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            "GitHub Models request failed with "
            f"HTTP {error.code}: {error_body}"
        ) from error

    choices = response_data.get("choices", [])

    if not choices:
        raise ValueError(
            "GitHub Models returned no completion choices."
        )

    message = choices[0].get("message", {})
    generated_text = message.get("content", "")

    if not generated_text:
        raise ValueError(
            "GitHub Models returned an empty response."
        )

    json_text = remove_markdown_fence(
        generated_text
    )

    return json.loads(json_text)


def validate_generated_plan(plan):
    """Validate model output before making repository changes."""

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
                f"Missing or invalid field: {field}"
            )

    files = plan.get("files")

    if not isinstance(files, list):
        raise ValueError(
            "The files field must be an array."
        )

    if not 1 <= len(files) <= MAX_CHANGED_FILES:
        raise ValueError(
            "The plan must change between one and "
            f"{MAX_CHANGED_FILES} files."
        )

    branch = plan["branch"]

    if not re.fullmatch(
        r"ai/[a-z0-9][a-z0-9._-]{2,60}",
        branch,
    ):
        raise ValueError(
            "The branch must match ai/<short-slug>."
        )

    seen_paths = set()

    for item in files:
        if not isinstance(item, dict):
            raise ValueError(
                "Every files entry must be an object."
            )

        path_value = item.get("path")
        content = item.get("content")

        if not isinstance(path_value, str):
            raise ValueError(
                "Every file must have a path."
            )

        if not isinstance(content, str):
            raise ValueError(
                f"File content must be text: {path_value}"
            )

        normalized_path = path_value.replace(
            "\\",
            "/",
        )

        path_object = Path(normalized_path)

        if normalized_path.startswith(
            BLOCKED_PATH_PREFIXES
        ):
            raise ValueError(
                f"Blocked path: {normalized_path}"
            )

        if ".." in path_object.parts:
            raise ValueError(
                f"Parent-directory path blocked: "
                f"{normalized_path}"
            )

        if path_object.suffix.lower() in BINARY_SUFFIXES:
            raise ValueError(
                f"Binary file blocked: {normalized_path}"
            )

        if normalized_path in seen_paths:
            raise ValueError(
                f"Duplicate file path: {normalized_path}"
            )

        seen_paths.add(normalized_path)

        line_count = len(content.splitlines())

        if line_count > MAX_LINES_PER_FILE:
            raise ValueError(
                f"Generated file is too large: "
                f"{normalized_path} has {line_count} lines."
            )


def apply_generated_plan(plan):
    """Create a branch, commit files, push, and open a PR."""

    branch = plan["branch"]

    run_command(
        "git",
        "checkout",
        "-b",
        branch,
    )

    generated_paths = []

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

        generated_paths.append(
            item["path"]
        )

    run_command(
        "git",
        "add",
        "--",
        *generated_paths,
    )

    repository_status = run_command(
        "git",
        "status",
        "--porcelain",
    )

    if not repository_status:
        raise ValueError(
            "The model produced no repository changes."
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

    run_command(
        "git",
        "commit",
        "-m",
        plan["commit_message"],
    )

    run_command(
        "git",
        "push",
        "--set-upstream",
        "origin",
        branch,
    )

    pull_request_body = (
        plan["pr_body"]
        + "\n\n"
        + "Generated by the GitHub Models maintenance agent."
        + "\n"
        + "This is a repository-scoped experimental automation."
    )

    run_command(
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        plan["pr_title"],
        "--body",
        pull_request_body,
    )


def main():
    actual_repository = os.getenv(
        "GITHUB_REPOSITORY",
        "",
    )

    if actual_repository != ALLOWED_REPO:
        raise SystemExit(
            "Repository blocked. "
            f"Received {actual_repository!r}, "
            f"expected {ALLOWED_REPO!r}."
        )

    repository_context = build_repository_context()

    if not repository_context:
        raise SystemExit(
            "No eligible repository files were found."
        )

    prompt = (
        "Review the repository snapshot below. "
        "Choose exactly one small, useful maintenance task. "
        "Return one valid JSON plan according to the system "
        "instructions.\n\n"
        f"Repository: {actual_repository}\n"
        f"Target model: {GITHUB_MODEL}\n"
        f"{repository_context}"
    )

    plan = call_github_models(prompt)

    validate_generated_plan(plan)

    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )
    )

    apply_generated_plan(plan)


if __name__ == "__main__":
    main()
