#!/usr/bin/env python3
"""
ollama_coder.py — terminal coding/command agent using the Ollama API.

What’s included
- COMMANDS with safety gate (Accept/Modify/Reject)
- FILE INGEST: capture stdout into context (e.g., `cat file.py`)
- PERSISTENT CHECKLIST: breakdown + status tracking in ~/.ollama_coder_checklist.json
- NEW: FILE EDITS with git-diff preview and $EDITOR-based modify flow

Env vars
- OLLAMA_HOST             (default: http://127.0.0.1:11434)
- OLLAMA_MODEL            (default: llama3.1)
- AGENT_MAX_TURNS         (default: 12)
- AGENT_MAX_INGEST_BYTES  (default: 204800)
- CHECKLIST_PATH          (default: ~/.ollama_coder_checklist.json)
- EDITOR                  (default: vi)  # used for “Modify” on edits
"""

import os
import re
import sys
import json
import uuid
import time
import tempfile
import difflib
import signal
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# --- HTTP client (requests if available, else urllib) ---
try:
    import requests  # type: ignore
    _USE_REQUESTS = True
except Exception:
    import urllib.request
    _USE_REQUESTS = False

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "12"))
MAX_INGEST_BYTES = int(os.getenv("AGENT_MAX_INGEST_BYTES", str(200 * 1024)))
CHECKLIST_PATH = os.path.expanduser(os.getenv("CHECKLIST_PATH", "~/.ollama_coder_checklist.json"))

CHAT_ENDPOINT = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

SYSTEM_PROMPT = """You are a careful coding & shell assistant running in a terminal.

You can:
1) Answer questions ("type": "answer"),
2) Propose a command to run ("type": "command"),
3) Propose FILE EDITS ("type": "edit") with new file contents.

FILE INGEST:
- If the user asks about a local file (e.g., "make my python.py do ..."),
  you may propose a safe read command (like `cat ./python.py`) and set "ingest_stdout": true.

CHECKLIST:
- You may create/update/remove persistent tasks via "checklist_delta".
- If a command or edit completes a task, include "task_id" so the host can auto-update status.

FILE EDITS:
- Use "type": "edit" and include "edits": [{ "path": "file.py", "new_content": "<entire file content>", "create_ok": true }]
- Provide whole-file "new_content". The host will show a unified diff and ask the user to Accept/Modify/Reject.
- Keep edits minimal and deterministic. Avoid destructive changes unless explicitly asked.

ALWAYS reply in pure JSON (no prose) with this schema:
{
  "type": "answer" | "command" | "edit",
  "answer": string,             // when type="answer"
  "command": string,            // when type="command"
  "explanation": string,        // brief reason this helps
  "cwd": string | null,         // optional working dir; null for current
  "ingest_stdout": boolean,     // optional for type="command"
  "ingest_title": string | null,// optional label for ingested content
  "task_id": string | null,     // optional: links this command or edit to a checklist item
  "edits": [                    // required when type="edit"
    { "path": string, "new_content": string, "create_ok": boolean }
  ],
  "checklist_delta": {          // optional
     "upserts": [ { "id": string|null, "title": string, "status": "pending|completed|failed" } ],
     "remove_ids": [ string, ... ]
  }
}

Rules:
- Prefer "answer" when a command/edit isn't necessary.
- For file reads, keep commands least-privilege; for edits, prefer smallest viable change.
- Output must be valid JSON only (no code fences or extra text).
"""

# ---------------- Checklist Persistence ----------------

class Checklist:
    def __init__(self, path: str):
        self.path = path
        self.items: Dict[str, Dict[str, str]] = {}  # id -> {"title": ..., "status": ...}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.items = {k: v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            self.items = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def upsert(self, item_id: Optional[str], title: str, status: Optional[str] = None) -> str:
        if not item_id:
            item_id = str(uuid.uuid4())[:8]
        status = status if status in ("pending", "completed", "failed") else "pending"
        self.items[item_id] = {"title": title, "status": status}
        self._save()
        return item_id

    def remove(self, item_id: str):
        if item_id in self.items:
            del self.items[item_id]
            self._save()

    def mark(self, item_id: str, status: str):
        if item_id in self.items and status in ("pending", "completed", "failed"):
            self.items[item_id]["status"] = status
            self._save()

    def clear(self):
        self.items = {}
        self._save()

    def pretty_print(self):
        if not self.items:
            print("(checklist empty)")
            return
        print("ID      | STATUS    | TITLE")
        print("-" * 60)
        for k, v in self.items.items():
            print(f"{k:<8} | {v.get('status','pending'):<9} | {v.get('title','')}")


# ---------------- Ollama HTTP ----------------

def http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if _USE_REQUESTS:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
        return json.loads(body.decode("utf-8"))

def ask_ollama(messages: List[Dict[str, str]]) -> str:
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    data = http_post_json(CHAT_ENDPOINT, payload)
    try:
        return data["message"]["content"]
    except Exception:
        if "messages" in data and data["messages"]:
            return data["messages"][-1].get("content", "")
        raise RuntimeError(f"Unexpected Ollama response: {data}")

# ---------------- Parsing helpers ----------------

def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
    raise ValueError("Could not parse JSON reply from the model.")

def validate_agent_reply(obj: Dict[str, Any]) -> Tuple[
    str, Optional[str], Optional[str], Optional[str], bool, Optional[str], Optional[str], Dict[str, Any], list
]:
    """
    Returns: (type, answer, command, cwd, ingest_stdout, ingest_title, task_id, checklist_delta, edits)
    """
    t = obj.get("type")
    if t not in ("answer", "command", "edit"):
        raise ValueError("Missing or invalid 'type' in model reply.")

    checklist_delta = obj.get("checklist_delta") or {}

    if t == "answer":
        ans = obj.get("answer")
        if not isinstance(ans, str):
            raise ValueError("Missing 'answer' string for type='answer'.")
        return ("answer", ans, None, None, False, None, None, checklist_delta, [])

    if t == "command":
        cmd = obj.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("Missing 'command' string for type='command'.")
        cwd = obj.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            cwd = None
        ingest_stdout = bool(obj.get("ingest_stdout", False))
        ingest_title = obj.get("ingest_title")
        if ingest_title is not None and not isinstance(ingest_title, str):
            ingest_title = None
        task_id = obj.get("task_id")
        if task_id is not None and not isinstance(task_id, str):
            task_id = None
        return ("command", None, cmd.strip(), cwd, ingest_stdout, ingest_title, task_id, checklist_delta, [])

    # type == "edit"
    edits = obj.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("Missing 'edits' array for type='edit'.")
    # Normalize edits
    normalized = []
    for e in edits:
        if not isinstance(e, dict):
            continue
        path = e.get("path")
        new_content = e.get("new_content")
        create_ok = bool(e.get("create_ok", False))
        if isinstance(path, str) and isinstance(new_content, str):
            normalized.append({"path": path, "new_content": new_content, "create_ok": create_ok})
    if not normalized:
        raise ValueError("No valid edit entries found.")
    task_id = obj.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        task_id = None
    return ("edit", None, None, None, False, None, task_id, checklist_delta, normalized)

# ---------------- Command execution ----------------

def run_command_stream(command: str, cwd: Optional[str]) -> int:
    print(f"\n[exec] {command}\n")
    try:
        shell = os.environ.get("SHELL", "/bin/bash")
        proc = subprocess.Popen(
            command, cwd=cwd or None, shell=True, executable=shell,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        return proc.returncode
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return 127
    except KeyboardInterrupt:
        print("\n[info] Command interrupted by user (Ctrl-C).")
        return 130

def run_command_capture(command: str, cwd: Optional[str]) -> Tuple[int, str]:
    print(f"\n[exec|capture] {command}\n")
    try:
        shell = os.environ.get("SHELL", "/bin/bash")
        completed = subprocess.run(
            command, cwd=cwd or None, shell=True, executable=shell,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        output = completed.stdout or ""
        return (completed.returncode, output)
    except FileNotFoundError as e:
        return (127, f"[error] {e}")
    except KeyboardInterrupt:
        return (130, "[info] Command interrupted by user (Ctrl-C).")

# ---------------- Edit helpers ----------------

def print_hr():
    print("-" * 60)

def truncate_for_ingest(s: str, max_bytes: int) -> Tuple[str, bool]:
    raw = s.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return (s, False)
    truncated = raw[:max_bytes]
    safe = truncated.decode("utf-8", errors="ignore")
    return (safe, True)

def read_text_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        # crude binary guard
        if "\x00" in data:
            return None
        return data
    except Exception:
        return None

def backup_path(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{path}.bak.{ts}"

def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def unified_diff_for(path: str, old: str, new: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        lineterm=""
    )
    return "\n".join(diff)

def edit_in_editor(initial_text: str) -> str:
    editor = os.environ.get("EDITOR", "vim")
    with tempfile.NamedTemporaryFile("w+", suffix=".tmp", delete=False, encoding="utf-8") as tf:
        tmp_path = tf.name
        tf.write(initial_text)
        tf.flush()
    try:
        subprocess.run([editor, tmp_path], check=False)
        with open(tmp_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ---------------- Main ----------------

def main():
    checklist = Checklist(CHECKLIST_PATH)

    print_hr()
    print("Ollama Coder — terminal agent (ingest + checklist + git-diff edits)")
    print(f"Model: {OLLAMA_MODEL}   Endpoint: {CHAT_ENDPOINT}")
    print("Type 'exit' or 'quit' to leave. Press Ctrl-C to cancel a running command.")
    print(f"Ingest cap: {MAX_INGEST_BYTES} bytes")
    print("Meta commands: ':checklist', ':done <id>', ':fail <id>', ':pending <id>', ':del <id>', ':clearchecklist'")
    print_hr()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    def apply_checklist_delta(delta: Dict[str, Any]):
        if not isinstance(delta, dict):
            return []
        upserts = delta.get("upserts") or []
        remove_ids = delta.get("remove_ids") or []
        created_ids = []
        for item in upserts:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            title = item.get("title")
            status = item.get("status")
            if isinstance(title, str) and title.strip():
                new_id = checklist.upsert(item_id if isinstance(item_id, str) else None, title.strip(), status if isinstance(status, str) else None)
                created_ids.append(new_id)
        for rid in remove_ids:
            if isinstance(rid, str):
                checklist.remove(rid)
        if upserts or remove_ids:
            print_hr()
            print("[checklist updated]")
            checklist.pretty_print()
            print_hr()
        return created_ids

    def show_checklist():
        print_hr()
        checklist.pretty_print()
        print_hr()

    def handle_local_command(line: str) -> bool:
        parts = line.strip().split()
        if not parts:
            return True
        cmd = parts[0].lower()
        if cmd == ":checklist":
            show_checklist(); return True
        if cmd in (":done", ":fail", ":pending", ":del") and len(parts) >= 2:
            _id = parts[1]
            if cmd == ":del":
                checklist.remove(_id)
            else:
                checklist.mark(_id, {":done":"completed", ":fail":"failed", ":pending":"pending"}[cmd])
            show_checklist(); return True
        if cmd == ":clearchecklist":
            checklist.clear(); show_checklist(); return True
        return False

    def add_turn(user_msg: str, assistant_msg: str):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        while len(messages) > (2 * MAX_TURNS + 1):
            del messages[1:3]

    while True:
        try:
            user_in = input("\nagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if user_in.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        if not user_in:
            continue
        if user_in.startswith(":"):
            if handle_local_command(user_in):
                continue

        # Add the user's prompt (temporarily) for the request
        messages.append({"role": "user", "content": user_in})

        try:
            raw = ask_ollama(messages)
        except Exception as e:
            print(f"[error] Failed to contact Ollama: {e}")
            messages.pop()
            continue

        try:
            obj = extract_json_object(raw)
            t, answer, command, cwd, ingest_stdout, ingest_title, task_id, checklist_delta, edits = validate_agent_reply(obj)
        except Exception:
            print("[warn] Model did not return valid JSON as instructed.")
            print("Raw reply:\n")
            print(raw)
            messages.append({"role": "assistant", "content": raw})
            continue

        # Apply any checklist updates first
        apply_checklist_delta(checklist_delta or {})

        # -------- Answer --------
        if t == "answer":
            print_hr(); print(answer); print_hr()
            messages.append({"role": "assistant", "content": json.dumps(obj)})
            continue

        # -------- Command --------
        if t == "command":
            explanation = obj.get("explanation") or ""
            print_hr()
            if explanation: print(f"Proposed command rationale: {explanation}")
            print("Proposed command:"); print(f"  {command}")
            if cwd: print(f"Working directory: {cwd}")
            if ingest_stdout:
                print("Note: stdout will be captured and added to chat context (subject to size cap).")
                if ingest_title: print(f"Ingest title: {ingest_title}")
            if task_id: print(f"Linked checklist task_id: {task_id}")
            print_hr()

            def prompt_user_choice(prompt: str = "Accept [a] / Modify [m] / Reject [r]? ") -> str:
                while True:
                    choice = input(prompt).strip().lower()
                    if choice in ("a", "m", "r"): return choice
                    print("Please type 'a' to accept, 'm' to modify, or 'r' to reject.")

            choice = prompt_user_choice()
            if choice == "r":
                print("[info] Command rejected.")
                messages.append({"role": "assistant", "content": json.dumps({"type": "answer", "answer": "User rejected the proposed command."})})
                continue
            if choice == "m":
                new_cmd = input("Enter modified command: ").strip()
                if not new_cmd:
                    print("[info] Empty command; skipping.")
                    messages.append({"role": "assistant", "content": json.dumps({"type": "answer","answer": "User canceled after choosing modify."})})
                    continue
                command = new_cmd

            if ingest_stdout:
                exit_code, output = run_command_capture(command, cwd)
                print_hr()
                if output:
                    preview, was_trunc = truncate_for_ingest(output, MAX_INGEST_BYTES)
                    print("[preview]" + (" (truncated)" if was_trunc else ""))
                    print(preview if preview.strip() else "(no output)")
                else:
                    print("[preview] (no output)")
                print_hr(); print(f"[exit status] {exit_code}"); print_hr()

                title = ingest_title or "command-output"
                content_block = f"<<INGEST:{title}>>\n{preview}\n<<END-INGEST>>"
                messages.append({"role": "assistant", "content": content_block})
            else:
                exit_code = run_command_stream(command, cwd)
                print_hr(); print(f"[exit status] {exit_code}"); print_hr()

            if task_id and isinstance(task_id, str):
                checklist.mark(task_id, "completed" if exit_code == 0 else "failed")
                print("[checklist updated from command result]")
                checklist.pretty_print(); print_hr()

            run_summary = {"type": "answer","answer": f"Executed command: {command}\nExit code: {exit_code}\n"}
            messages.append({"role": "assistant", "content": json.dumps(run_summary)})
            continue

        # -------- Edit (git-diff preview) --------
        if t == "edit":
            explanation = obj.get("explanation") or ""
            print_hr()
            if explanation: print(f"Proposed edits rationale: {explanation}")
            print("[preview] unified diff(s):")
            all_diffs = []
            old_texts = []
            new_texts = []

            for e in edits:
                path = e["path"]
                new_content = e["new_content"]
                create_ok = bool(e.get("create_ok", False))
                old_content = read_text_file(path)
                if old_content is None:
                    # Treat as new file
                    if not create_ok:
                        print(f"\n[warn] {path} does not exist (and create_ok is false). This edit will be skipped unless modified.")
                        old_content = ""
                    else:
                        old_content = ""
                diff = unified_diff_for(path, old_content, new_content)
                if not diff.strip():
                    diff = f"--- {path}\n+++ {path}\n# (no changes)"
                print("\n" + diff)
                all_diffs.append(diff)
                old_texts.append(old_content)
                new_texts.append(new_content)

            print_hr()

            def prompt_user_choice(prompt: str = "Apply all [a] / Modify [m] / Reject [r]? ") -> str:
                while True:
                    choice = input(prompt).strip().lower()
                    if choice in ("a", "m", "r"): return choice
                    print("Please type 'a' to apply, 'm' to modify, or 'r' to reject.")

            choice = prompt_user_choice()
            if choice == "r":
                print("[info] Edits rejected.")
                messages.append({"role": "assistant", "content": json.dumps({"type":"answer","answer":"User rejected proposed edits."})})
                continue

            if choice == "m":
                # Let the user choose a file to modify, open $EDITOR on temp, then re-preview and ask again.
                for idx, e in enumerate(edits):
                    print(f"{idx}: {e['path']}")
                try:
                    sel = int(input("Select index to modify (number): ").strip())
                    assert 0 <= sel < len(edits)
                except Exception:
                    print("[info] Invalid selection; aborting modify.")
                    messages.append({"role":"assistant","content":json.dumps({"type":"answer","answer":"User canceled modify for edits."})})
                    continue
                proposed = new_texts[sel]
                edited = edit_in_editor(proposed)
                new_texts[sel] = edited
                # Show updated diff for that file
                path = edits[sel]["path"]
                diff = unified_diff_for(path, old_texts[sel], edited)
                print_hr(); print(f"[updated diff] {path}\n{diff if diff.strip() else '(no changes)'}"); print_hr()
                # Ask apply again
                choice2 = input("Apply ALL edits with current contents? [y/N]: ").strip().lower()
                if choice2 != "y":
                    print("[info] Edits not applied.")
                    messages.append({"role":"assistant","content":json.dumps({"type":"answer","answer":"User reviewed but did not apply edits."})})
                    continue

            # Apply all edits
            success = True
            for i, e in enumerate(edits):
                path = e["path"]
                new_content = new_texts[i]
                old_content = old_texts[i]
                exists = os.path.exists(path)
                if not exists:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                else:
                    # backup
                    try:
                        bpath = backup_path(path)
                        with open(bpath, "w", encoding="utf-8") as bf:
                            bf.write(old_content)
                    except Exception as be:
                        print(f"[warn] Could not create backup for {path}: {be}")
                try:
                    write_text_file(path, new_content)
                    print(f"[applied] {path}")
                except Exception as we:
                    print(f"[error] Failed to write {path}: {we}")
                    success = False

            print_hr()
            print("[result] Edits applied." if success else "[result] Some edits failed.")
            print_hr()

            # Add diff context for the model (truncated)
            combined = "\n\n".join(all_diffs)
            ingest, was_trunc = truncate_for_ingest(combined, MAX_INGEST_BYTES)
            messages.append({"role":"assistant","content": f"<<PATCH:multiple>>\n{ingest}\n<<END-PATCH>>"})

            # Auto-update checklist task if provided
            if task_id and isinstance(task_id, str):
                checklist.mark(task_id, "completed" if success else "failed")
                print("[checklist updated from edit result]")
                checklist.pretty_print(); print_hr()

            # Summarize to conversation
            messages.append({"role":"assistant","content":json.dumps({"type":"answer","answer":("Applied file edits successfully." if success else "Some file edits failed.")})})
            continue

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
