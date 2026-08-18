import json
import os
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_REPO = os.getenv("ALLOWED_REPO", "DylanOSlab/market-data-reliability-lab")
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")
LLAMA_CLI = os.getenv("LLAMA_CLI", "./llama.cpp/build/bin/llama-cli")
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

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".exe", ".dll", ".so", ".bin",
    ".pyc", ".pyd", ".woff", ".woff2", ".ttf",
}
BLOCKED_PATH_PREFIXES = (".github/", ".git/", ".automation/", ".env")
BLOCKED_FILE_NAMES = {"package-lock.json", "poetry.lock", "uv.lock", "Pipfile.lock"}
PRIORITY_PATH_PREFIXES = ("src/", "tests/", "scripts/", "fixtures/", "provenance/")

SYSTEM_PROMPT = """
You are an autonomous software maintenance agent for one public experimental repository.
Choose exactly one small, useful, low-risk change that advances the project.
Prefer: a missing regression test, deterministic test coverage, a small confirmed defect,
input validation, error handling, fixture/provenance validation, or a documentation correction.
Return only one valid JSON object with this exact structure:
{
  "summary": "short explanation",
  "branch": "ai/short-lowercase-slug",
  "commit_message": "short commit message",
  "pr_title": "pull request title",
  "pr_body": "pull request body",
  "files": [{"path": "relative/path", "content": "complete replacement content"}]
}
Rules:
- Change exactly one text file.
- Return complete replacement content, not a diff.
- Keep the generated file below 200 lines.
- Never modify .github, .git, .automation, environment files, secrets, permissions,
  billing, repository settings, workflows, Actions configuration, security policies,
  lock files, or binary files.
- Never delete files or use absolute/parent-directory paths.
- Never claim tests passed.
- Never invent files, functions, dependencies, APIs, or behavior.
- Prefer modifying an existing file and keep the change easy to review and revert.
""".strip()


def run_command(*args, check=True):
    process = subprocess.run(
        args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if check and process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nExit code: {process.returncode}\n"
            f"Standard output:\n{process.stdout}\nStandard error:\n{process.stderr}"
        )
    return process.stdout.strip()


def is_safe_context_file(path):
    if not path.is_file():
        return False
    normalized = path.as_posix()
    if any(normalized.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        return False
    if path.name in BLOCKED_FILE_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
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
    candidates = [path for path in Path(".").rglob("*") if is_safe_context_file(path)]
    candidates.sort(key=context_sort_key)
    sections = []
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
            if remaining >= 400:
                sections.append(section[:remaining])
            break
        sections.append(section)
        current_size += len(section)
    return "".join(sections)


def build_model_prompt(repository_context):
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
        "<|im_start|>user\n"
        f"Repository: {ALLOWED_REPO}\n\n"
        "Use only the repository snapshot below. Choose one small task and return only JSON.\n\n"
        + repository_context
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )


def run_local_model(prompt):
    cli_path = Path(LLAMA_CLI)
    if not cli_path.exists():
        raise FileNotFoundError(f"llama.cpp executable was not found: {LLAMA_CLI}")
    thread_count = min(4, max(1, os.cpu_count() or 1))
    command = [
        str(cli_path), "-hf", MODEL_SPEC, "-p", prompt,
        "-n", str(MAX_GENERATED_TOKENS), "-c", str(MODEL_CONTEXT_SIZE),
        "-t", str(thread_count), "--temp", "0.1", "--top-p", "0.9",
        "--repeat-penalty", "1.05", "--no-display-prompt", "--no-mmap",
        "--simple-io",
    ]
    print("Starting local Qwen inference.", flush=True)
    print(f"Model: {MODEL_SPEC}", flush=True)
    print(f"Prompt size: {len(prompt)} characters", flush=True)
    print(f"Maximum generated tokens: {MAX_GENERATED_TOKENS}", flush=True)
    print(f"Model context size: {MODEL_CONTEXT_SIZE}", flush=True)
    print(f"CPU threads: {thread_count}", flush=True)
    try:
        process = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=MODEL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Local model exceeded the 20-minute inference limit.") from error
    if process.stderr:
        print(process.stderr, file=sys.stderr, flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Local model failed with exit code {process.returncode}.\n"
            f"Output:\n{process.stdout}\nErrors:\n{process.stderr}"
        )
    output = process.stdout.strip()
    if not output:
        raise ValueError("The local model returned an empty response.")
    print(f"Local model returned {len(output)} characters.", flush=True)
    return output


def extract_json_object(model_output):
    cleaned = re.sub(r"^```(?:json)?\s*", "", model_output.strip(), flags=re.I)
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
    raise ValueError(f"The model did not return valid JSON.\nRaw output:\n{model_output}")


def validate_relative_path(path_value):
    if not isinstance(path_value, str):
        raise ValueError("Every generated file must have a string path.")
    normalized = path_value.replace("\\", "/").strip()
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError(f"Invalid path: {normalized!r}")
    if any(normalized.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        raise ValueError(f"Protected path blocked: {normalized}")
    path_object = Path(normalized)
    if ".." in path_object.parts:
        raise ValueError(f"Parent-directory path blocked: {normalized}")
    if path_object.name in BLOCKED_FILE_NAMES or path_object.suffix.lower() in BINARY_SUFFIXES:
        raise ValueError(f"Blocked file: {normalized}")
    return normalized


def validate_generated_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Generated plan must be a JSON object.")
    for field in ("summary", "branch", "commit_message", "pr_title", "pr_body"):
        value = plan.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid field: {field}")
    if not re.fullmatch(r"ai/[a-z0-9][a-z0-9._-]{2,60}", plan["branch"].strip()):
        raise ValueError("Generated branch must match ai/<short-lowercase-slug>.")
    files = plan.get("files")
    if not isinstance(files, list) or len(files) != MAX_CHANGED_FILES:
        raise ValueError("The generated plan must modify exactly one file.")
    total_characters = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each files entry must be an object.")
        normalized_path = validate_relative_path(item.get("path"))
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Generated content is invalid: {normalized_path}")
        line_count = len(content.splitlines())
        if line_count > MAX_GENERATED_LINES_PER_FILE:
            raise ValueError(f"Generated file has {line_count} lines: {normalized_path}")
        total_characters += len(content)
        item["path"] = normalized_path
    if total_characters > MAX_TOTAL_GENERATED_CHARACTERS:
        raise ValueError("Generated content exceeds the total size limit.")


def ensure_branch_does_not_exist(branch):
    process = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if process.returncode == 0:
        raise ValueError(f"Generated branch already exists: {branch}")


def apply_generated_plan(plan):
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
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    print(run_command("git", "diff", "--cached", "--stat"), flush=True)
    run_command("git", "commit", "-m", plan["commit_message"].strip())
    run_command("git", "push", "--set-upstream", "origin", branch)
    body = (
        plan["pr_body"].strip()
        + "\n\nGenerated by a local Qwen Coder model running inside GitHub Actions."
        + "\n\nNo external model API or API key was used."
    )
    url = run_command(
        "gh", "pr", "create", "--base", DEFAULT_BRANCH, "--head", branch,
        "--title", plan["pr_title"].strip(), "--body", body,
    )
    print(f"Pull request created: {url}", flush=True)


def main():
    actual_repository = os.getenv("GITHUB_REPOSITORY", "")
    if actual_repository != ALLOWED_REPO:
        raise SystemExit(
            f"Repository scope failed: received {actual_repository!r}, expected {ALLOWED_REPO!r}."
        )
    repository_context = build_repository_context()
    if not repository_context.strip():
        raise SystemExit("No eligible repository files were found.")
    print(f"Repository context size: {len(repository_context)} characters", flush=True)
    model_output = run_local_model(build_model_prompt(repository_context))
    print("Raw local model output:", flush=True)
    print(model_output, flush=True)
    plan = extract_json_object(model_output)
    validate_generated_plan(plan)
    print("Validated maintenance plan:", flush=True)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    apply_generated_plan(plan)


if __name__ == "__main__":
    main()
