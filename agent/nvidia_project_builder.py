import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = os.getenv("ALLOWED_REPO", "DylanOSlab/market-data-reliability-lab")
MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_FILES = int(os.getenv("MAX_FILES_PER_CHANGE", "12"))
MAX_LINES = int(os.getenv("MAX_CHANGED_LINES", "1200"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))
AUTO_MERGE = os.getenv("AUTO_MERGE", "false").lower() == "true"
BLOCKED = (".github/", ".git/", ".automation/state", ".env")
ALLOWED = ("src/", "tests/", "scripts/", "fixtures/", "provenance/", "README.md", "INSTALL_NEXT.md", "pyproject.toml")

SYSTEM = """You are the planner and coding agent for one repository. Advance the supplied
Project Charter by completing exactly one small, coherent task. Return ONLY valid JSON:
{"task_title":"...","task_body":"...","issue_title":"...","files":[{"path":"...","content":"complete file content"}],"tests":["..."]}
Rules: use only supplied repository evidence; return complete text files, not diffs; include
tests for new behavior; do not alter workflows, secrets, permissions, billing, branch protection,
lock files, or binary files; keep the task reviewable and independently useful; never claim tests passed."""


def run(*args, check=True, timeout=600):
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    if p.stdout:
        print(p.stdout, flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(args)}")
    return p


def call_nim(messages, max_tokens=7000):
    key = os.environ["NVIDIA_API_KEY"]
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0.15, "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API HTTP {error.code}: {detail}") from error
    text = data["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S | re.I)
    return json.loads(text)


def repository_context():
    charter = Path(".automation/project-charter.yml").read_text(encoding="utf-8")
    chunks = [f"PROJECT CHARTER:\n{charter}"]
    total = len(chunks[0])
    candidates = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.as_posix().startswith(".github/"):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".gif", ".zip", ".pdf", ".pyc"}:
            continue
        if path.stat().st_size > 30000:
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: (0 if p.as_posix().startswith(("src/", "tests/")) else 1, p.as_posix()))
    for path in candidates[:60]:
        text = path.read_text(encoding="utf-8", errors="replace")
        section = f"\n--- FILE: {path.as_posix()} ---\n{text}"
        if total + len(section) > 80000:
            break
        chunks.append(section)
        total += len(section)
    issues = run("gh", "issue", "list", "--repo", REPO, "--state", "open", "--limit", "50", "--json", "number,title,body,labels").stdout
    chunks.append(f"\nOPEN ISSUES:\n{issues}")
    return "".join(chunks)


def validate(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("files"), list) or not plan["files"]:
        raise ValueError("Plan must contain non-empty files")
    if len(plan["files"]) > MAX_FILES:
        raise ValueError("Too many files")
    seen = set()
    for item in plan["files"]:
        path = item.get("path", "").replace("\\", "/")
        content = item.get("content")
        if not path or not isinstance(content, str):
            raise ValueError("Invalid file entry")
        if path.startswith(BLOCKED) or not path.startswith(ALLOWED) or ".." in Path(path).parts:
            raise ValueError(f"Blocked path: {path}")
        if path in seen:
            raise ValueError(f"Duplicate path: {path}")
        seen.add(path)
        item["path"] = path


def apply_files(plan):
    for item in plan["files"]:
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["content"], encoding="utf-8")
    paths = [item["path"] for item in plan["files"]]
    run("git", "add", "--", *paths)
    run("git", "diff", "--cached", "--check")
    numstat = run("git", "diff", "--cached", "--numstat").stdout
    changed = 0
    for line in numstat.splitlines():
        a, d, _ = line.split("\t", 2)
        if a == "-" or d == "-":
            raise ValueError("Binary changes are blocked")
        changed += int(a) + int(d)
    if changed > MAX_LINES:
        raise ValueError(f"Change too large: {changed} lines")


def checks():
    run(sys.executable, "-m", "ruff", "check", ".", check=False)
    return run(sys.executable, "-m", "pytest", "-q", check=False, timeout=600)


def repair(plan, failure):
    current = []
    for item in plan["files"]:
        path = Path(item["path"])
        current.append({"path": item["path"], "content": path.read_text(encoding="utf-8")})
    prompt = {"failed_task": plan, "current_files": current, "test_output": failure.stdout[-12000:], "instruction": "Return the same JSON schema with corrected complete files. Keep the task scope unchanged."}
    return call_nim([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(prompt)}])


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "task"


def main():
    if os.getenv("GITHUB_REPOSITORY", "") != REPO:
        raise SystemExit("Repository scope rejected")
    run(sys.executable, "-m", "pip", "install", "-e", ".[dev]", check=False)
    baseline = checks()
    if baseline.returncode != 0:
        raise RuntimeError("Baseline tests fail; autonomous coding stopped")

    plan = call_nim([{"role": "system", "content": SYSTEM}, {"role": "user", "content": repository_context()}])
    validate(plan)
    apply_files(plan)

    result = checks()
    for _ in range(MAX_REPAIRS):
        if result.returncode == 0:
            break
        run("git", "reset")
        plan = repair(plan, result)
        validate(plan)
        apply_files(plan)
        result = checks()
    if result.returncode != 0:
        raise RuntimeError("Repair limit reached; tests still fail")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"ai/{stamp}-{slug(plan['task_title'])}"
    run("git", "checkout", "-b", branch)
    run("git", "commit", "-m", plan["task_title"][:72])
    run("git", "push", "--set-upstream", "origin", branch)

    issue_url = run("gh", "issue", "create", "--repo", REPO, "--title", plan.get("issue_title", plan["task_title"]), "--body", plan["task_body"], "--label", "autopilot").stdout.strip()
    issue_number = issue_url.rstrip("/").split("/")[-1]
    body = plan["task_body"] + f"\n\nCloses #{issue_number}\n\nGenerated with NVIDIA NIM and validated by repository tests."
    pr_url = run("gh", "pr", "create", "--repo", REPO, "--base", "main", "--head", branch, "--title", plan["task_title"], "--body", body).stdout.strip()
    print(f"Pull request created: {pr_url}")
    if AUTO_MERGE:
        run("gh", "pr", "merge", pr_url, "--auto", "--squash", "--delete-branch", check=False)


if __name__ == "__main__":
    main()
