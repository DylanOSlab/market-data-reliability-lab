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
MODEL_SPEC = os.getenv("MODEL_SPEC", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M")

MAX_CATALOG_CHARS = 6500
MAX_FILE_CHARS = 12000
MAX_GENERATED_TOKENS = 500
MODEL_CONTEXT_SIZE = 4096
MODEL_TIMEOUT_SECONDS = 300
HEARTBEAT_SECONDS = 30
MAX_PLAN_ATTEMPTS = 3
MAX_EDIT_ATTEMPTS = 4
MAX_REPLACEMENT_LINES = 80

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
    ".gz", ".tar", ".7z", ".exe", ".dll", ".so", ".bin", ".pyc",
    ".pyd", ".woff", ".woff2", ".ttf",
}
BLOCKED_PREFIXES = (".github/", ".git/", ".automation/", ".env")
BLOCKED_NAMES = {"package-lock.json", "poetry.lock", "uv.lock", "Pipfile.lock"}
PREFERRED_PREFIXES = ("tests/", "src/", "scripts/", "fixtures/", "provenance/")

PLANNER_PROMPT = """
You are planning one small improvement to an existing repository.
Return only JSON with exactly three keys: summary, path, objective.

Rules:
- path must exactly match one file path shown in the catalog.
- Pick an existing text file, preferably under tests/ or src/.
- objective must describe one concrete, low-risk change supported by the catalog snippets.
- Prefer a regression test, then a confirmed bug fix, validation, error handling, or docs.
- Do not output code, line numbers, placeholders, Markdown, or explanations outside JSON.
""".strip()

EDITOR_PROMPT = """
You are editing exactly one existing file shown with numbered lines.
Return only JSON with exactly five keys: summary, start_line, end_line, old_text, replacement.

Rules:
- start_line and end_line are 1-based inclusive integers from the numbered file.
- old_text must exactly equal the complete text from start_line through end_line, without
  line-number prefixes. Preserve indentation and newlines exactly.
- replacement must be different and must contain the complete replacement for those lines.
- Change at most 80 lines. Never use placeholders, ellipses, TODOs, or Markdown fences.
- Implement the stated objective using only symbols visible in the file.
- Keep the change small and syntactically valid.
- Do not claim tests passed and do not output anything outside JSON.
""".strip()


def run_command(*args, check=True, timeout=None):
    result = subprocess.run(
        args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nExit code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def safe_path(path_value):
    if not isinstance(path_value, str):
        return False
    value = path_value.replace("\\", "/").strip()
    if not value or value.startswith("/") or re.match(r"^[A-Za-z]:/", value):
        return False
    if any(value.startswith(prefix) for prefix in BLOCKED_PREFIXES):
        return False
    path = Path(value)
    if ".." in path.parts:
        return False
    if path.name in BLOCKED_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
        return False
    return True


def eligible_file(path):
    if not path.is_file() or not safe_path(path.as_posix()):
        return False
    try:
        return path.stat().st_size <= MAX_FILE_CHARS
    except OSError:
        return False


def sort_key(path):
    value = path.as_posix()
    for index, prefix in enumerate(PREFERRED_PREFIXES):
        if value.startswith(prefix):
            return index, value
    if value in {"pyproject.toml", "README.md", "INSTALL_NEXT.md"}:
        return 10, value
    return 20, value


def build_catalog():
    files = [path for path in Path(".").rglob("*") if eligible_file(path)]
    files.sort(key=sort_key)
    entries = []
    paths = set()
    size = 0

    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        nonempty = [line for line in content.splitlines() if line.strip()]
        preview = "\n".join(nonempty[:18])[:900]
        entry = f"\nFILE: {path.as_posix()}\nPREVIEW:\n{preview}\n"
        if size + len(entry) > MAX_CATALOG_CHARS:
            break
        entries.append(entry)
        paths.add(path.as_posix())
        size += len(entry)

    return "".join(entries), paths


def chat_prompt(system, user):
    return (
        "<|im_start|>system\n" + system + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_model(prompt, seed_offset=0):
    if not Path(LLAMA_CLI).exists():
        raise FileNotFoundError(f"llama.cpp executable not found: {LLAMA_CLI}")

    command = [
        LLAMA_CLI, "-hf", MODEL_SPEC, "-p", prompt,
        "-n", str(MAX_GENERATED_TOKENS), "-c", str(MODEL_CONTEXT_SIZE),
        "-t", str(min(4, max(1, os.cpu_count() or 1))),
        "--temp", "0.25", "--top-p", "0.85", "--repeat-penalty", "1.12",
        "--seed", str((int(time.time()) + seed_offset * 7919) % 2147483647),
        "--no-display-prompt", "--no-mmap", "--simple-io", "--single-turn",
    ]

    started = time.monotonic()
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
            f"Model exited with {process.returncode}.\nOutput:\n{stdout}\n"
            f"Errors:\n{stderr[-2000:]}"
        )
    if not stdout.strip():
        raise ValueError("Model returned empty output.")
    print(f"Inference completed in {int(time.monotonic() - started)} seconds.", flush=True)
    return stdout.strip()


def repair_common_json_errors(output):
    """Repair common JSON mistakes made by very small local models."""

    starts = [index for index, char in enumerate(output) if char == "{"]

    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(output)):
            char = output[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = output[start:index + 1]
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

                    try:
                        return json.loads(candidate, strict=False)
                    except json.JSONDecodeError:
                        break

    return None


def extract_last_json(output):
    decoder = json.JSONDecoder(strict=False)
    objects = []
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
            if isinstance(value, dict):
                objects.append(value)
        except json.JSONDecodeError:
            pass

    if not objects:
        repaired = repair_common_json_errors(output)
        if repaired is not None:
            objects.append(repaired)

    if not objects:
        raise ValueError(
            "No JSON object found in model output, even after repairing "
            "literal newlines and trailing commas."
        )
    return objects[-1]


def choose_plan(catalog, catalog_paths):
    last_error = ""
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        correction = ""
        if last_error:
            correction = f"\nPrevious plan was rejected: {last_error}\nChoose a different valid plan.\n"
        user = (
            f"Repository: {ALLOWED_REPO}\nAttempt: {attempt}\n{correction}\n"
            f"CATALOG:\n{catalog}"
        )
        output = run_model(chat_prompt(PLANNER_PROMPT, user), attempt)
        print("Planner output:\n" + output, flush=True)
        try:
            plan = extract_last_json(output)
            if set(plan) != {"summary", "path", "objective"}:
                raise ValueError("Plan JSON keys must be summary, path, objective.")
            if not all(isinstance(plan[key], str) and plan[key].strip() for key in plan):
                raise ValueError("Plan fields must be non-empty strings.")
            path = plan["path"].replace("\\", "/").strip()
            if path not in catalog_paths:
                raise ValueError(f"Path is not in catalog: {path}")
            plan["path"] = path
            return plan
        except (ValueError, TypeError, KeyError) as error:
            last_error = str(error)
            print(f"Planner attempt {attempt} rejected: {last_error}", flush=True)
    raise RuntimeError(f"Planner failed after {MAX_PLAN_ATTEMPTS} attempts: {last_error}")


def numbered_file(content):
    lines = content.splitlines()
    width = len(str(max(1, len(lines))))
    return "\n".join(f"{index:>{width}}|{line}" for index, line in enumerate(lines, 1))


def exact_slice(lines, start_line, end_line):
    return "\n".join(lines[start_line - 1:end_line])


def choose_edit(plan, original):
    lines = original.splitlines()
    if not lines:
        raise ValueError("Target file is empty.")

    display = numbered_file(original)
    last_error = ""
    previous = None

    for attempt in range(1, MAX_EDIT_ATTEMPTS + 1):
        correction = ""
        if last_error:
            correction = (
                f"\nPrevious edit was rejected: {last_error}\n"
                f"Rejected edit: {json.dumps(previous, ensure_ascii=False)}\n"
                "Return a different, corrected edit.\n"
            )
        user = (
            f"Repository: {ALLOWED_REPO}\nFile: {plan['path']}\n"
            f"Objective: {plan['objective']}\nAttempt: {attempt}\n{correction}\n"
            f"NUMBERED FILE:\n{display}"
        )
        output = run_model(chat_prompt(EDITOR_PROMPT, user), 100 + attempt)
        print("Editor output:\n" + output, flush=True)
        try:
            edit = extract_last_json(output)
            previous = edit
            required = {"summary", "start_line", "end_line", "old_text", "replacement"}
            if set(edit) != required:
                raise ValueError("Edit JSON keys are incorrect.")
            if not isinstance(edit["summary"], str) or len(edit["summary"].strip()) < 8:
                raise ValueError("Edit summary is missing or too short.")
            start = edit["start_line"]
            end = edit["end_line"]
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("Line numbers must be integers.")
            if start < 1 or end < start or end > len(lines):
                raise ValueError(f"Line range {start}-{end} is outside the file.")
            if end - start + 1 > MAX_REPLACEMENT_LINES:
                raise ValueError("Edit changes too many lines.")
            expected_old = exact_slice(lines, start, end)
            if edit["old_text"] != expected_old:
                raise ValueError(
                    "old_text does not exactly match the selected lines. "
                    f"Expected: {expected_old!r}"
                )
            replacement = edit["replacement"]
            if not isinstance(replacement, str) or not replacement.strip():
                raise ValueError("Replacement is empty.")
            if replacement == expected_old:
                raise ValueError("Replacement is identical to old text.")
            if len(replacement.splitlines()) > MAX_REPLACEMENT_LINES:
                raise ValueError("Replacement contains too many lines.")
            if "..." in replacement or "TODO" in replacement:
                raise ValueError("Replacement contains ellipses or TODO placeholder.")
            return edit
        except (ValueError, TypeError, KeyError) as error:
            last_error = str(error)
            print(f"Editor attempt {attempt} rejected: {last_error}", flush=True)

    raise RuntimeError(f"Editor failed after {MAX_EDIT_ATTEMPTS} attempts: {last_error}")


def run_checks():
    if not Path("pyproject.toml").exists():
        return
    print("Running pytest before opening a pull request.", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, timeout=300,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"pytest failed with exit code {result.returncode}.")


def slug(text):
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:34] or "small-improvement"


def apply_and_publish(plan, edit, original):
    lines = original.splitlines()
    start = edit["start_line"]
    end = edit["end_line"]
    replacement_lines = edit["replacement"].splitlines()
    updated_lines = lines[:start - 1] + replacement_lines + lines[end:]
    updated = "\n".join(updated_lines)
    if original.endswith("\n"):
        updated += "\n"

    target = Path(plan["path"])
    target.write_text(updated, encoding="utf-8")
    run_checks()

    fingerprint = hashlib.sha1(
        f"{plan['path']}\n{edit['summary']}\n{edit['replacement']}".encode("utf-8")
    ).hexdigest()[:7]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"ai/{stamp}-{slug(edit['summary'])}-{fingerprint}"

    run_command("git", "checkout", "-b", branch)
    run_command("git", "add", "--", plan["path"])
    if not run_command("git", "status", "--porcelain"):
        raise ValueError("Edit produced no repository change.")
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)

    title = edit["summary"].strip()[:72]
    commit = title[0].lower() + title[1:]
    print(run_command("git", "diff", "--cached", "--stat"), flush=True)
    run_command("git", "commit", "-m", commit)
    run_command("git", "push", "--set-upstream", "origin", branch)

    body = (
        "## Summary\n\n"
        f"{edit['summary'].strip()}\n\n"
        "## Objective\n\n"
        f"{plan['objective'].strip()}\n\n"
        "## Validation\n\n"
        "- pytest completed successfully before this pull request was created.\n\n"
        "Generated by a local Qwen Coder model running inside GitHub Actions.\n"
        "No external model API or API key was used."
    )
    url = run_command(
        "gh", "pr", "create", "--base", DEFAULT_BRANCH, "--head", branch,
        "--title", title, "--body", body,
    )
    print(f"Pull request created: {url}", flush=True)


def main():
    actual = os.getenv("GITHUB_REPOSITORY", "")
    if actual != ALLOWED_REPO:
        raise SystemExit(f"Repository blocked: {actual!r}; expected {ALLOWED_REPO!r}.")

    catalog, paths = build_catalog()
    if not catalog:
        raise SystemExit("No eligible repository files were found.")
    print(f"Catalog size: {len(catalog)} characters across {len(paths)} files.", flush=True)

    plan = choose_plan(catalog, paths)
    print("Validated plan:\n" + json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    target = Path(plan["path"])
    original = target.read_text(encoding="utf-8", errors="strict")
    edit = choose_edit(plan, original)
    print("Validated edit:\n" + json.dumps(edit, indent=2, ensure_ascii=False), flush=True)

    apply_and_publish(plan, edit, original)


if __name__ == "__main__":
    main()
