from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = os.getenv("ALLOWED_REPO", "DylanOSlab/market-data-reliability-lab")
MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_FILES = int(os.getenv("MAX_FILES_PER_CHANGE", "12"))
MAX_LINES = int(os.getenv("MAX_CHANGED_LINES", "1200"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))
AUTO_MERGE = False
BLOCKED = (".github/", ".git/", ".automation/", ".env")
ALLOWED = ("src/", "tests/", "scripts/", "fixtures/", "provenance/", "README.md", "INSTALL_NEXT.md", "pyproject.toml")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".zip", ".pdf", ".pyc", ".exe", ".dll"}
SYSTEM = """You are the planner and coding agent for exactly one repository. Complete exactly one small, coherent, non-duplicate task that advances the supplied Project Charter. Return ONLY valid JSON with this schema:
{"task_title":"...","task_body":"...","issue_title":"...","acceptance_criteria":["..."],"files":[{"path":"...","content":"complete file content"}],"tests":["..."]}
Use only supplied repository evidence. Return complete text files, never diffs or placeholders. Preserve unrelated behavior and existing file content. Include meaningful regression tests. Do not alter workflows, automation policy, secrets, permissions, billing, branch protection, lock files, or binary files. Never claim tests passed."""


def run(*args: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    if process.stdout:
        print(process.stdout, flush=True)
    if check and process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(args)}")
    return process


def call_nim(messages: list[dict[str, str]], max_tokens: int = 7000) -> dict[str, Any]:
    key = os.environ["NVIDIA_API_KEY"]
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens, "stream": False}).encode()
    request = urllib.request.Request(API_URL, data=body, method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API HTTP {error.code}: {detail[:2000]}") from error
    text = data["choices"][0]["message"]["content"].strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("NVIDIA response did not contain a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("NVIDIA response root must be an object")
    return value


def existing_work() -> dict[str, Any]:
    issues = json.loads(run("gh", "issue", "list", "--repo", REPO, "--state", "all", "--limit", "200", "--json", "number,title,state,url").stdout or "[]")
    prs = json.loads(run("gh", "pr", "list", "--repo", REPO, "--state", "all", "--limit", "200", "--json", "number,title,state,mergedAt,headRefName,url").stdout or "[]")
    return {"issues": issues, "pull_requests": prs}


def repository_context(work: dict[str, Any]) -> str:
    charter = Path(".automation/project-charter.yml").read_text(encoding="utf-8")
    chunks = [f"PROJECT CHARTER:\n{charter}", f"\nEXISTING WORK:\n{json.dumps(work, ensure_ascii=False)}"]
    total = sum(map(len, chunks))
    candidates: list[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.as_posix().startswith(".github/"):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES or path.stat().st_size > 30000:
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: (0 if p.as_posix().startswith(("src/", "tests/")) else 1, p.as_posix()))
    for path in candidates[:80]:
        section = f"\n--- FILE: {path.as_posix()} ---\n{path.read_text(encoding='utf-8', errors='replace')}"
        if total + len(section) > 100000:
            break
        chunks.append(section)
        total += len(section)
    return "".join(chunks)


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def is_duplicate(plan: dict[str, Any], work: dict[str, Any]) -> bool:
    candidate = normalize_title(str(plan.get("issue_title") or plan.get("task_title") or ""))
    if not candidate:
        return True
    titles = [normalize_title(str(item.get("title", ""))) for group in work.values() for item in group]
    return candidate in titles


def validate(plan: dict[str, Any]) -> None:
    required_text = ("task_title", "task_body", "issue_title")
    if any(not isinstance(plan.get(key), str) or not plan[key].strip() for key in required_text):
        raise ValueError("Plan is missing required task metadata")
    if not isinstance(plan.get("acceptance_criteria"), list) or not plan["acceptance_criteria"]:
        raise ValueError("Plan must contain acceptance criteria")
    files = plan.get("files")
    if not isinstance(files, list) or not files or len(files) > MAX_FILES:
        raise ValueError("Plan must contain a permitted number of files")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid file entry")
        path = str(item.get("path", "")).replace("\\", "/")
        content = item.get("content")
        if not path or not isinstance(content, str) or "\x00" in content:
            raise ValueError("Invalid text file entry")
        if path.startswith(BLOCKED) or not path.startswith(ALLOWED) or ".." in Path(path).parts:
            raise ValueError(f"Blocked path: {path}")
        if Path(path).suffix.lower() in BINARY_SUFFIXES or path in seen:
            raise ValueError(f"Binary or duplicate path: {path}")
        seen.add(path)
        item["path"] = path


def preservation_check(plan: dict[str, Any]) -> None:
    for item in plan["files"]:
        path = Path(item["path"])
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8", errors="replace")
        new = item["content"]
        if len(old) >= 800 and len(new) < int(len(old) * 0.55):
            raise ValueError(f"Suspicious file truncation blocked: {path}")
        old_defs = set(re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)", old, flags=re.M))
        new_defs = set(re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)", new, flags=re.M))
        removed = sorted(old_defs - new_defs)
        if removed:
            raise ValueError(f"Existing symbols removed from {path}: {removed}")


def restore(paths: list[str]) -> None:
    for value in paths:
        path = Path(value)
        tracked = run("git", "ls-files", "--error-unmatch", value, check=False).returncode == 0
        if tracked:
            run("git", "checkout", "HEAD", "--", value)
        elif path.exists():
            path.unlink()
    run("git", "reset", check=False)


def apply_files(plan: dict[str, Any]) -> list[str]:
    preservation_check(plan)
    paths: list[str] = []
    for item in plan["files"]:
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["content"], encoding="utf-8")
        paths.append(item["path"])
    run("git", "add", "--", *paths)
    run("git", "diff", "--cached", "--check")
    if run("git", "diff", "--cached", "--quiet", check=False).returncode == 0:
        raise ValueError("Plan produced no repository changes")
    changed = 0
    for line in run("git", "diff", "--cached", "--numstat").stdout.splitlines():
        added, deleted, _ = line.split("\t", 2)
        if added == "-" or deleted == "-":
            raise ValueError("Binary changes are blocked")
        changed += int(added) + int(deleted)
    if changed > MAX_LINES:
        raise ValueError(f"Change too large: {changed} lines")
    return paths


def checks() -> tuple[bool, str]:
    ruff = run(sys.executable, "-m", "ruff", "check", ".", check=False)
    pytest = run(sys.executable, "-m", "pytest", "-q", check=False, timeout=600)
    output = f"RUFF\n{ruff.stdout}\nPYTEST\n{pytest.stdout}"
    return ruff.returncode == 0 and pytest.returncode == 0, output


def repair(plan: dict[str, Any], failure: str) -> dict[str, Any]:
    current = [{"path": item["path"], "content": Path(item["path"]).read_text(encoding="utf-8")} for item in plan["files"] if Path(item["path"]).exists()]
    prompt = {"failed_task": plan, "current_files": current, "validation_output": failure[-16000:], "instruction": "Return the same JSON schema with corrected complete files. Preserve scope, acceptance criteria, and unrelated content."}
    return call_nim([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(prompt)}])


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "task"


def ensure_label(name: str, color: str) -> None:
    run("gh", "label", "create", name, "--repo", REPO, "--color", color, "--force", check=False)


def main() -> None:
    if os.getenv("GITHUB_REPOSITORY", "") != REPO:
        raise SystemExit("Repository scope rejected")
    run(sys.executable, "-m", "pip", "install", "-e", ".[dev]")
    baseline_ok, baseline_output = checks()
    if not baseline_ok:
        raise RuntimeError(f"Baseline checks fail; autonomous coding stopped\n{baseline_output[-6000:]}")

    work = existing_work()
    context = repository_context(work)
    plan: dict[str, Any] | None = None
    for attempt in range(3):
        candidate = call_nim([{"role": "system", "content": SYSTEM}, {"role": "user", "content": context + f"\nPLANNING ATTEMPT: {attempt + 1}. Do not duplicate existing work."}])
        validate(candidate)
        if not is_duplicate(candidate, work):
            plan = candidate
            break
    if plan is None:
        raise RuntimeError("No non-duplicate task was produced after three planning attempts")

    paths = apply_files(plan)
    success, output = checks()
    repairs = 0
    while not success and repairs < MAX_REPAIRS:
        repairs += 1
        repaired = repair(plan, output)
        validate(repaired)
        restore(paths)
        plan = repaired
        paths = apply_files(plan)
        success, output = checks()
    if not success:
        restore(paths)
        raise RuntimeError(f"Repair limit reached; checks still fail\n{output[-6000:]}")

    ensure_label("autopilot", "1D76DB")
    ensure_label("ai-generated", "7057FF")
    issue_body = plan["task_body"] + "\n\n## Acceptance criteria\n" + "\n".join(f"- [ ] {item}" for item in plan["acceptance_criteria"])
    issue_url = run("gh", "issue", "create", "--repo", REPO, "--title", plan["issue_title"], "--body", issue_body, "--label", "autopilot,ai-generated").stdout.strip()
    issue_number = issue_url.rstrip("/").split("/")[-1]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"ai/{stamp}-{slug(plan['task_title'])}"
    run("git", "checkout", "-b", branch)
    run("git", "commit", "-m", plan["task_title"][:72])
    run("git", "push", "--set-upstream", "origin", branch)
    pr_body = issue_body + f"\n\nCloses #{issue_number}\n\nGenerated with NVIDIA NIM. Local Ruff and pytest were executed by Python, not trusted from model output.\n\nAUTO_MERGE=false"
    pr_url = run("gh", "pr", "create", "--repo", REPO, "--base", "main", "--head", branch, "--title", plan["task_title"], "--body", pr_body, "--label", "autopilot,ai-generated").stdout.strip()
    print(f"Pull request created: {pr_url}")
    if AUTO_MERGE:
        raise RuntimeError("AUTO_MERGE must remain false during quality validation")


if __name__ == "__main__":
    main()
