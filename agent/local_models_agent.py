import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_REPO = os.getenv(
    "ALLOWED_REPO",
    "DylanOSlab/market-data-reliability-lab",
)
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")
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
MAX_SEARCH_CHARACTERS = 2_500
MAX_REPLACE_CHARACTERS = 4_000
MAX_GENERATED_TOKENS = 500
MODEL_CONTEXT_SIZE = 4_096
MODEL_TIMEOUT_SECONDS = 600
HEARTBEAT_SECONDS = 30

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".exe", ".dll", ".so", ".bin",
    ".pyc", ".pyd", ".woff", ".woff2", ".ttf",
}
BLOCKED_PATH_PREFIXES = (".github/", ".git/", ".automation/", ".env")
BLOCKED_FILE_NAMES = {
    "package-lock.json", "poetry.lock", "uv.lock", "Pipfile.lock",
}
PRIORITY_PATH_PREFIXES = (
    "src/", "tests/", "scripts/", "fixtures/", "provenance/",
)

SYSTEM_PROMPT = """
You are an autonomous software maintenance agent for one public experimental repository.
Choose exactly one small, useful, low-risk change supported by the supplied repository text.
Prefer a focused regression test, a small confirmed defect fix, input validation, error
handling, fixture/provenance validation, or a documentation correction.

Return ONLY a JSON object with exactly four keys: summary, path, search, replace.
- summary: a short factual description of the change.
- path: the exact relative path of ONE existing text file shown in the snapshot.
- search: an exact non-empty substring copied verbatim from that file.
- replace: the complete replacement text for that substring.

Mandatory rules:
- Modify exactly one existing file.
- The search text must occur exactly once in the selected file.
- Keep search below 2500 characters and replace below 4000 characters.
- Make a real code, test, fixture, provenance, or documentation improvement.
- Do not return branch names, commit messages, pull-request metadata, Markdown fences,
  explanations outside JSON, placeholders, ellipses, or comments such as TODO.
- Do not modify .github, .git, .automation, environment files, secrets, tokens,
  permissions, billing, repository settings, workflows, Actions configuration,
  security policies, lock files, or binary files.
- Never claim tests passed. Never invent files, functions, dependencies, APIs, or behavior.
""".strip()


def run_command(*args, check=True, timeout=None):
    process = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if check and process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"Exit code: {process.returncode}\n"
            f"Standard output:\n{process.stdout}\n"
            f"Standard error:\n{process.stderr}"
        )
    return process.stdout.strip()


def is_safe_path_text(path_value):
    if not isinstance(path_value, str):
        return False
    normalized = path_value.replace("\\", "/").strip()
    if not normalized or normalized.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:/", normalized):
        return False
    if any(normalized.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        return False
    path = Path(normalized)
    if ".." in path.parts:
        return False
    if path.name in BLOCKED_FILE_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
        return False
    return True


def is_safe_context_file(path):
    if not path.is_file() or not is_safe_path_text(path.as_posix()):
        return False
    try:
        return path.stat().st_size <= 16_000
    except OSError:
        return False


def context_sort_key(path):
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
    candidates = [
        path for path in Path(".").rglob("*") if is_safe_context_file(path)
    ]
    candidates.sort(key=context_sort_key)
    sections = []
    current_size = 0
    included_paths = []

    for path in candidates[:MAX_CONTEXT_FILES]:
        try:
            file_content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        section = f"\n--- FILE: {path.as_posix()} ---\n{file_content}\n"
        remaining = MAX_CONTEXT_CHARACTERS - current_size
        if remaining <= 0:
            break
        if len(section) > remaining:
            if remaining >= 400:
                sections.append(section[:remaining])
                included_paths.append(path.as_posix())
            break
        sections.append(section)
        included_paths.append(path.as_posix())
        current_size += len(section)

    return "".join(sections), set(included_paths)


def build_model_prompt(repository_context):
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Repository: {ALLOWED_REPO}\n\n"
        "Select one exact search-and-replace edit from this snapshot. Return JSON only.\n"
        f"{repository_context}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_local_model(prompt):
    cli_path = Path(LLAMA_CLI)
    if not cli_path.exists():
        raise FileNotFoundError(f"llama.cpp executable was not found: {LLAMA_CLI}")

    thread_count = min(4, max(1, os.cpu_count() or 1))
    command = [
        str(cli_path),
        "-hf", MODEL_SPEC,
        "-p", prompt,
        "-n", str(MAX_GENERATED_TOKENS),
        "-c", str(MODEL_CONTEXT_SIZE),
        "-t", str(thread_count),
        "--temp", "0.1",
        "--top-p", "0.9",
        "--repeat-penalty", "1.05",
        "--no-display-prompt",
        "--no-mmap",
        "--simple-io",
        "--single-turn",
    ]

    print("Starting local Qwen inference.", flush=True)
    print(f"Model: {MODEL_SPEC}", flush=True)
    print(f"Prompt size: {len(prompt)} characters", flush=True)
    print(f"Maximum generated tokens: {MAX_GENERATED_TOKENS}", flush=True)

    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    while process.poll() is None:
        elapsed = int(time.monotonic() - started_at)
        if elapsed >= MODEL_TIMEOUT_SECONDS:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(
                "Local model exceeded the configured inference timeout.\n"
                f"Partial output:\n{stdout}\nErrors:\n{stderr[-4000:]}"
            )
        print(f"Local inference is still running: {elapsed} seconds elapsed.", flush=True)
        time.sleep(HEARTBEAT_SECONDS)

    stdout, stderr = process.communicate()
    if stderr:
        print(stderr[-4000:], file=sys.stderr, flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Local model failed with exit code {process.returncode}.\n"
            f"Output:\n{stdout}\nErrors:\n{stderr[-4000:]}"
        )
    output = stdout.strip()
    if not output:
        raise ValueError("The local model returned an empty response.")
    print(
        f"Local model completed after {int(time.monotonic() - started_at)} seconds.",
        flush=True,
    )
    return output


def extract_json_object(model_output):
    cleaned = re.sub(
        r"^```(?:json)?\s*", "", model_output.strip(), flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    parsed_objects = []
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(parsed, dict):
                parsed_objects.append(parsed)
        except json.JSONDecodeError:
            continue
    if not parsed_objects:
        raise ValueError(f"The model did not return valid JSON.\nRaw output:\n{model_output}")
    return parsed_objects[-1]


def validate_edit(edit, included_paths):
    if not isinstance(edit, dict):
        raise ValueError("Generated edit must be a JSON object.")
    if set(edit) != {"summary", "path", "search", "replace"}:
        raise ValueError(
            "Generated JSON must contain exactly: summary, path, search, replace."
        )

    summary = edit["summary"]
    path_value = edit["path"]
    search = edit["search"]
    replacement = edit["replace"]

    if not isinstance(summary, str) or len(summary.strip()) < 8:
        raise ValueError("Generated summary is missing or too short.")
    if not is_safe_path_text(path_value):
        raise ValueError(f"Generated path is blocked: {path_value!r}")

    normalized_path = path_value.replace("\\", "/").strip()
    if normalized_path not in included_paths:
        raise ValueError(
            f"Generated path was not included in model context: {normalized_path}"
        )

    if not isinstance(search, str) or not search:
        raise ValueError("Generated search text must be non-empty.")
    if not isinstance(replacement, str) or not replacement:
        raise ValueError("Generated replacement text must be non-empty.")
    if len(search) > MAX_SEARCH_CHARACTERS:
        raise ValueError("Generated search text exceeds the size limit.")
    if len(replacement) > MAX_REPLACE_CHARACTERS:
        raise ValueError("Generated replacement text exceeds the size limit.")
    if search == replacement:
        raise ValueError("Generated search and replacement text are identical.")

    forbidden = {
        "exact substring copied from the file",
        "replacement text",
        "complete replacement content",
        "complete replacement file content",
        "relative/path",
    }
    if search.strip().lower() in forbidden or replacement.strip().lower() in forbidden:
        raise ValueError("Model returned placeholder search or replacement text.")
    if "..." in search or "..." in replacement:
        raise ValueError("Ellipses are not allowed in generated edits.")

    path = Path(normalized_path)
    if not path.is_file():
        raise ValueError(f"Generated target does not exist: {normalized_path}")
    original = path.read_text(encoding="utf-8", errors="strict")
    occurrences = original.count(search)
    if occurrences != 1:
        raise ValueError(
            f"Search text must occur exactly once in {normalized_path}; found {occurrences}."
        )

    edit["path"] = normalized_path
    return original


def make_slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:36] or "small-improvement"


def run_project_checks():
    print("Running project checks before creating a pull request.", flush=True)
    if Path("pyproject.toml").exists():
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=300,
        )
        print(process.stdout, flush=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"Project tests failed with exit code {process.returncode}."
            )


def apply_edit(edit, original):
    path = Path(edit["path"])
    updated = original.replace(edit["search"], edit["replace"], 1)
    path.write_text(updated, encoding="utf-8")

    run_project_checks()

    fingerprint = hashlib.sha1(
        f"{edit['path']}\n{edit['summary']}".encode("utf-8")
    ).hexdigest()[:7]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"ai/{stamp}-{make_slug(edit['summary'])}-{fingerprint}"

    run_command("git", "checkout", "-b", branch)
    run_command("git", "add", "--", edit["path"])
    if not run_command("git", "status", "--porcelain"):
        raise ValueError("The generated edit produced no repository change.")
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)

    title = edit["summary"].strip()[:72]
    commit_message = title[0].lower() + title[1:] if title else "apply small improvement"
    print(run_command("git", "diff", "--cached", "--stat"), flush=True)
    run_command("git", "commit", "-m", commit_message)
    run_command("git", "push", "--set-upstream", "origin", branch)

    body = (
        "## Summary\n\n"
        f"{edit['summary'].strip()}\n\n"
        "## Validation\n\n"
        "- Local repository tests completed successfully before this PR was created.\n\n"
        "Generated by a local Qwen Coder model running inside GitHub Actions.\n"
        "No external model API or API key was used."
    )
    url = run_command(
        "gh", "pr", "create",
        "--base", DEFAULT_BRANCH,
        "--head", branch,
        "--title", title,
        "--body", body,
    )
    print(f"Pull request created: {url}", flush=True)


def main():
    actual_repository = os.getenv("GITHUB_REPOSITORY", "")
    if actual_repository != ALLOWED_REPO:
        raise SystemExit(
            f"Repository scope failed: received {actual_repository!r}, "
            f"expected {ALLOWED_REPO!r}."
        )

    context, included_paths = build_repository_context()
    if not context.strip():
        raise SystemExit("No eligible repository files were found.")

    print(f"Repository context size: {len(context)} characters", flush=True)
    output = run_local_model(build_model_prompt(context))
    print("Raw local model output:", flush=True)
    print(output, flush=True)

    edit = extract_json_object(output)
    print("Extracted edit:", flush=True)
    print(json.dumps(edit, indent=2, ensure_ascii=False), flush=True)

    original = validate_edit(edit, included_paths)
    apply_edit(edit, original)


if __name__ == "__main__":
    main()
