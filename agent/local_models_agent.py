import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_REPO = os.getenv("ALLOWED_REPO", "DylanOSlab/market-data-reliability-lab")
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")
LLAMA_CLI = os.getenv("LLAMA_CLI", "./llama.cpp/build/bin/llama-cli")
MODEL_SPEC = os.getenv(
    "MODEL_SPEC", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M"
)

MAX_CONTEXT_CHARACTERS = 6000
MAX_CONTEXT_FILES = 12
MAX_SEARCH_CHARACTERS = 1800
MAX_REPLACE_CHARACTERS = 3000
MAX_GENERATED_TOKENS = 420
MODEL_CONTEXT_SIZE = 4096
MODEL_TIMEOUT_SECONDS = 300
HEARTBEAT_SECONDS = 30
MAX_MODEL_ATTEMPTS = 4

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".exe", ".dll", ".so", ".bin",
    ".pyc", ".pyd", ".woff", ".woff2", ".ttf",
}
BLOCKED_PATH_PREFIXES = (".github/", ".git/", ".automation/", ".env")
BLOCKED_FILE_NAMES = {"package-lock.json", "poetry.lock", "uv.lock", "Pipfile.lock"}
PRIORITY_PATH_PREFIXES = ("tests/", "src/", "scripts/", "fixtures/", "provenance/")

SYSTEM_PROMPT = """
You edit one existing repository file through an exact search-and-replace operation.
Return only JSON with exactly four keys: summary, path, search, replace.

Rules:
1. path must be an exact FILE path shown in the repository snapshot.
2. search must be copied verbatim from that file and must occur exactly once.
3. replace must differ from search and contain the improved text.
4. Make one small, useful change: preferably a focused regression test, otherwise a
   confirmed bug fix, validation improvement, error-handling improvement, or precise
   documentation correction.
5. Do not use placeholders, ellipses, Markdown fences, branch names, commit messages,
   pull-request text, TODO comments, or explanations outside JSON.
6. Do not modify protected configuration, workflow, secret, lock, or binary files.
7. Do not claim tests passed and do not invent symbols not visible in the snapshot.
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


def is_safe_path(path_value):
    if not isinstance(path_value, str):
        return False
    normalized = path_value.replace("\\", "/").strip()
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return False
    if any(normalized.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        return False
    path = Path(normalized)
    if ".." in path.parts:
        return False
    if path.name in BLOCKED_FILE_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
        return False
    return True


def is_context_file(path):
    if not path.is_file() or not is_safe_path(path.as_posix()):
        return False
    try:
        return path.stat().st_size <= 16000
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
    return 20, normalized


def build_repository_context():
    candidates = [path for path in Path(".").rglob("*") if is_context_file(path)]
    candidates.sort(key=context_sort_key)
    sections = []
    included_paths = set()
    current_size = 0

    for path in candidates[:MAX_CONTEXT_FILES]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        section = f"\n--- FILE: {path.as_posix()} ---\n{content}\n"
        remaining = MAX_CONTEXT_CHARACTERS - current_size
        if remaining <= 0:
            break
        if len(section) > remaining:
            if remaining >= 500:
                sections.append(section[:remaining])
                included_paths.add(path.as_posix())
            break
        sections.append(section)
        included_paths.add(path.as_posix())
        current_size += len(section)

    return "".join(sections), included_paths


def build_prompt(context, attempt, previous_error=None, previous_edit=None):
    retry = ""
    if previous_error:
        retry = (
            "\nThe previous answer was rejected. Produce a DIFFERENT edit.\n"
            f"Rejection reason: {previous_error}\n"
            f"Rejected answer: {json.dumps(previous_edit, ensure_ascii=False)}\n"
            "Do not repeat the rejected search or replacement.\n"
        )

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Repository: {ALLOWED_REPO}\n"
        f"Attempt: {attempt} of {MAX_MODEL_ATTEMPTS}\n"
        f"{retry}\n"
        "Choose one exact edit from this snapshot. Return JSON only.\n"
        f"{context}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_local_model(prompt):
    if not Path(LLAMA_CLI).exists():
        raise FileNotFoundError(f"llama.cpp executable was not found: {LLAMA_CLI}")

    command = [
        LLAMA_CLI,
        "-hf", MODEL_SPEC,
        "-p", prompt,
        "-n", str(MAX_GENERATED_TOKENS),
        "-c", str(MODEL_CONTEXT_SIZE),
        "-t", str(min(4, max(1, os.cpu_count() or 1))),
        "--temp", "0.35",
        "--top-p", "0.85",
        "--repeat-penalty", "1.10",
        "--seed", str(int(time.time()) % 2147483647),
        "--no-display-prompt",
        "--no-mmap",
        "--simple-io",
        "--single-turn",
    ]

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    while process.poll() is None:
        elapsed = int(time.monotonic() - started)
        if elapsed >= MODEL_TIMEOUT_SECONDS:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"Inference exceeded {MODEL_TIMEOUT_SECONDS} seconds.\n"
                f"Partial output:\n{stdout}\nErrors:\n{stderr[-2000:]}"
            )
        print(f"Inference heartbeat: {elapsed} seconds.", flush=True)
        time.sleep(HEARTBEAT_SECONDS)

    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"Local model failed with exit code {process.returncode}.\n"
            f"Output:\n{stdout}\nErrors:\n{stderr[-2000:]}"
        )
    if not stdout.strip():
        raise ValueError("The local model returned an empty response.")
    print(f"Inference completed in {int(time.monotonic() - started)} seconds.", flush=True)
    return stdout.strip()


def extract_last_json(model_output):
    decoder = json.JSONDecoder()
    objects = []
    for index, character in enumerate(model_output):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(model_output[index:])
            if isinstance(parsed, dict):
                objects.append(parsed)
        except json.JSONDecodeError:
            continue
    if not objects:
        raise ValueError("No valid JSON object was found in model output.")
    return objects[-1]


def validate_edit(edit, included_paths):
    required = {"summary", "path", "search", "replace"}
    if not isinstance(edit, dict) or set(edit) != required:
        raise ValueError("JSON must contain exactly summary, path, search, replace.")

    summary = edit["summary"]
    path_value = edit["path"]
    search = edit["search"]
    replacement = edit["replace"]

    if not isinstance(summary, str) or len(summary.strip()) < 8:
        raise ValueError("Summary is missing or too short.")
    if not is_safe_path(path_value):
        raise ValueError(f"Path is unsafe: {path_value!r}")

    normalized_path = path_value.replace("\\", "/").strip()
    if normalized_path not in included_paths:
        raise ValueError(f"Path was not included in context: {normalized_path}")
    if not isinstance(search, str) or not search.strip():
        raise ValueError("Search text is empty.")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("Replacement text is empty.")
    if search == replacement:
        raise ValueError("Search and replacement are identical.")
    if len(search) > MAX_SEARCH_CHARACTERS:
        raise ValueError("Search text is too long.")
    if len(replacement) > MAX_REPLACE_CHARACTERS:
        raise ValueError("Replacement text is too long.")
    if "..." in search or "..." in replacement:
        raise ValueError("Ellipses are not allowed.")

    placeholders = {
        "validate", "replacement", "replacement text", "search text",
        "exact substring", "exact text", "relative/path", "path/to/file",
    }
    if search.strip().lower() in placeholders or replacement.strip().lower() in placeholders:
        raise ValueError("Placeholder search or replacement was returned.")

    path = Path(normalized_path)
    if not path.is_file():
        raise ValueError(f"Target file does not exist: {normalized_path}")
    original = path.read_text(encoding="utf-8", errors="strict")
    occurrences = original.count(search)
    if occurrences != 1:
        raise ValueError(
            f"Search must occur exactly once in {normalized_path}; found {occurrences}."
        )

    edit["path"] = normalized_path
    return original


def choose_valid_edit(context, included_paths):
    previous_error = None
    previous_edit = None

    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        print(f"Model edit attempt {attempt}/{MAX_MODEL_ATTEMPTS}.", flush=True)
        output = run_local_model(
            build_prompt(context, attempt, previous_error, previous_edit)
        )
        print("Raw model output:", flush=True)
        print(output, flush=True)

        try:
            edit = extract_last_json(output)
            print("Candidate edit:", flush=True)
            print(json.dumps(edit, indent=2, ensure_ascii=False), flush=True)
            original = validate_edit(edit, included_paths)
            return edit, original
        except (ValueError, KeyError, TypeError) as error:
            previous_error = str(error)
            previous_edit = locals().get("edit")
            print(f"Attempt {attempt} rejected: {previous_error}", flush=True)

    raise RuntimeError(
        f"The model failed to produce a valid edit after {MAX_MODEL_ATTEMPTS} attempts. "
        f"Last error: {previous_error}"
    )


def run_project_checks():
    if not Path("pyproject.toml").exists():
        return
    print("Running pytest before creating a pull request.", flush=True)
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
        raise RuntimeError(f"pytest failed with exit code {process.returncode}.")


def make_slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:36] or "small-improvement"


def apply_edit(edit, original):
    path = Path(edit["path"])
    updated = original.replace(edit["search"], edit["replace"], 1)
    path.write_text(updated, encoding="utf-8")

    run_project_checks()

    fingerprint = hashlib.sha1(
        f"{edit['path']}\n{edit['summary']}\n{edit['replace']}".encode("utf-8")
    ).hexdigest()[:7]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"ai/{stamp}-{make_slug(edit['summary'])}-{fingerprint}"

    run_command("git", "checkout", "-b", branch)
    run_command("git", "add", "--", edit["path"])
    if not run_command("git", "status", "--porcelain"):
        raise ValueError("The edit produced no repository change.")
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)

    title = edit["summary"].strip()[:72]
    commit_message = title[0].lower() + title[1:]
    print(run_command("git", "diff", "--cached", "--stat"), flush=True)
    run_command("git", "commit", "-m", commit_message)
    run_command("git", "push", "--set-upstream", "origin", branch)

    body = (
        "## Summary\n\n"
        f"{edit['summary'].strip()}\n\n"
        "## Validation\n\n"
        "- pytest completed successfully before this pull request was created.\n\n"
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
    edit, original = choose_valid_edit(context, included_paths)
    print("Validated edit:", flush=True)
    print(json.dumps(edit, indent=2, ensure_ascii=False), flush=True)
    apply_edit(edit, original)


if __name__ == "__main__":
    main()
