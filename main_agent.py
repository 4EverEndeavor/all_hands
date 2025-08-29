#!/usr/bin/env python3
"""
main_agent.py — terminal coding/command agent using the Ollama API with
built‑in **tool calling** (no manual JSON parsing from the model).

What’s included
- TOOL CALLING via `tools=[...]` compatible with Ollama's function calling
- SAFETY GATE for executing shell commands and sed edits (Accept/Modify/Skip)
- FILE INGEST tools (cat file, find files)
- CHECKLIST
- Cross‑platform sed edit helper (BSD/macOS vs GNU)

Env vars
- OLLAMA_HOST
- OLLAMA_MODEL
- AGENT_MAX_TURNS         # used to trim history
- AGENT_MAX_INGEST_BYTES
- CHECKLIST_PATH
- EDITOR
- LOG_LEVEL
"""

from __future__ import annotations

import os
import re
import sys
import json
import uuid
import shutil
import tempfile
import logging
import subprocess
import platform
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "12"))
MAX_INGEST_BYTES = int(os.getenv("AGENT_MAX_INGEST_BYTES", str(200 * 1024)))
CHECKLIST_PATH = os.path.expanduser(os.getenv("CHECKLIST_PATH", "~/.ollama_coder_checklist.json"))

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("main_agent")

client = ollama.Client(host=OLLAMA_HOST)
CHAT_ENDPOINT = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

current_directory = os.getcwd()
SYSTEM_PROMPT = (
    "You are a careful coding & shell assistant running in a terminal.\n"
    "When possible, call the provided TOOLS instead of writing shell commands directly.\n"
    "General rules:\n"
    "- Prefer reading files before proposing edits.\n"
    "- Keep edits minimal; avoid collateral changes.\n"
    "- If a task seems risky (e.g., destructive commands), explain risks and ask for confirmation.\n"
    "Output plain helpful text unless you need to call tools."
    "When you are given a task, break the task into any number of smaller tasks.\n"
    "Create a checklist for these tasks in the following list-of-maps-style format: \n"
    "[\n"
    "    {<pending|completed|failed>: <sub_task_1_description>},\n"
    "    {<pending|completed|failed>: <sub_task_2_description>},\n"
    "    ...\n"
    "]\n"
    "When a task has been updated, use the update_checklist tool.\n"
    "When all tasks have been completed, update the checklist to and empty list\n"
    "For any failed tasks, output a possible explanation of why the task failed.\n"
    "Propose possible solutions. If no solutions are available,\n"
    "suggest some approaches on how to investigate or troubleshoot the issue.\n"
    "These can include terminal commands, checking software versions, ect\n"
    f"Your current working directory is {current_directory}"
    "Any mentioned files will be at or below this directory.\n"
    "You may need to use commands to find files, move around between\n"
    "directories, or list the contents of directories.\n"
)


# ---------------- Helpers ----------------

def shlex_quote(s: str) -> str:
    import shlex as _shlex
    return _shlex.quote(s)

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

def print_hr():
    print("-" * 60)

# ---------------- Tool Registry ----------------


# Global state that some tools use
CURRENT_CWD: Optional[str] = None
CHECKLIST = []

# ---- Tool implementations ----

def _tool_result_message(name: str, content: Dict[str, Any], tool_call_id: Optional[str]) -> Dict[str, Any]:
    """Shape a message carrying tool output. Some runtimes expect `role='tool'` and a `tool_call_id`.
    We include both the name and a JSON string content. Adjust if your Ollama build expects a slightly different shape.
    """
    payload = {
        "role": "tool",
        "content": json.dumps(content, ensure_ascii=False),
        "name": name,
    }
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


# 1) edit a file
def edit_file(args: Dict[str, Any]) -> Dict[str, Any]:
    file = args.get("file")
    command = args.get("command")
    if not file:
        return {"ok": False, "error": "file is required"}
    if not command:
        return {"ok": False, "error": "command is required"}
    print(f"proposed file edit: {command}")
    choice = input(f"Does this look safe?  y/n\n")
    if choice != "y":
        return {"ok": False, "error": "command does not appear to be safe to run"}

    # show a diff
    try:
        # 1. Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        logger.debug(f"Created temporary file {temp_file_path}")

        # 2. Copy the contents of the input file to the temp file
        shutil.copy2(file, temp_file_path)
        logger.debug(f"Copied contents from {file} to {temp_file_path}")

        # 4. Execute the sed command
        logger.debug(f"Executing command {command} in shell")
        subprocess.run(command, shell=True, check=True)

        # 5. Diff the original and modified files
        logger.debug("displaying diff now")
        print("\n--- Diff ---")
        subprocess.run(["diff -y", file, temp_file_path], check=True)
        print("--- End Diff ---")

        # 6. Excecute changes
        choice = input('Apply the changes? y/n')
        if choice == 'y':
            return {'ok': True, "message": "Successfully executed command"}
        else:
            # copy the original contents back into the original file
            shutil.copy2(temp_file_path, file)
            reason = input('What is the reason for rejection?')
            return {'ok': False, "reasonn for rejection": reason}
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



# 2) read file
def read_file(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path")
    start = args.get("start_line")
    end = args.get("end_line")
    if not path:
        return {"ok": False, "error": "path is required"}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}"}

    print(f"Reading the contents from {path}")

    s = int(start) if start else 1
    e = int(end) if end else len(lines)
    s = max(1, s); e = min(len(lines), e)
    slice_lines = lines[s-1:e]

    # Apply ingest cap
    text = "".join(slice_lines)
    raw = text.encode("utf-8", errors="replace")
    truncated = False
    if len(raw) > MAX_INGEST_BYTES:
        raw = raw[:MAX_INGEST_BYTES]
        text = raw.decode("utf-8", errors="ignore")
        truncated = True

    numbered = []
    for i, l in enumerate(text.splitlines(), start=s):
        numbered.append(f"{i:>6}  {l}")
    text = "\n".join(numbered)

    return {
        "ok": True, "path": path, "start": s, "end": e,
        "truncated": truncated, "content": text
    }


# 3) find_file_by_name_tool
def find_file_by_name_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern")
    max_results = int(args.get("max_results", 50))
    ignore_dirs = set(args.get("ignore_dirs", [".git", "node_modules", "venv", ".venv", "__pycache__"]))
    if not pattern:
        return {"ok": False, "error": "pattern is required"}

    print(f"Searching for file with name {pattern}")

    matches = []
    print(f"Searching for file: {pattern}")
    for root, dirs, files in os.walk(CURRENT_CWD or os.getcwd()):
        # prune ignored dirs in place
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for fn in files:
            if re.search(pattern, fn):
                matches.append(os.path.join(root, fn))
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break
    return {"ok": True, "pattern": pattern, "results": matches}


# 4) change_directory_tool
def change_directory_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    global CURRENT_CWD
    path = args.get("path")
    if not path:
        return {"ok": False, "error": "path is required"}
    try:
        os.chdir(path)
        CURRENT_CWD = os.getcwd()
        return {"ok": True, "cwd": CURRENT_CWD}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# 5) run_shell_command_tool (for read‑only or safe commands)
def run_shell_command_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    command = args.get("command")
    mode = args.get("mode", "capture")  # capture|stream
    if not isinstance(command, str) or not command.strip():
        return {"ok": False, "error": "command (string) is required"}

    print_hr()
    print("Proposed command:")
    print(f"  {command}")
    if input("Run this command? [y/N]: ").strip().lower() != 'y':
        return {"ok": True, "ran": False, "message": "user declined"}

    if mode == "stream":
        code = run_command_stream(command, CURRENT_CWD)
        return {"ok": True, "ran": True, "exit_code": code}
    else:
        code, out = run_command_capture(command, CURRENT_CWD)
        return {"ok": True, "ran": True, "exit_code": code, "output": out}

# 6) checklist tools

def update_checklist(args: Dict[str, Any]) -> Dict[str, Any]:
    checklist = args.get("checklist")
    if not checklist:
        return {"ok": False, "error": "checklist is required"}
    CHECKLIST = checklist
    return {"ok": True, "message": "checklist successfully updated"}

# ---- Tool registry ----

TOOLS = [
    {
        "type": "function",
        "function":
        {
            "name": "edit_file",
            "description": "Edit a single file with a shell command.",
            "parameters":
            {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "command": {"type": "string"}
                },
                "required": ["file", "command"],
            },
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "read_file",
            "description": "Read a file (optionally by line range) and return content (may be truncated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "find_file_by_name_tool",
            "description": "Find files whose basename matches a regex pattern. Prunes common noisy dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1},
                    "ignore_dirs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["pattern"],
            },
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "change_directory_tool",
            "description": "Change current working directory for future operations.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "run_shell_command_tool",
            "description": "Run a shell command after explicit user confirmation. Use for diagnostics, git, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "mode": {"type": "string", "enum": ["capture", "stream"], "default": "capture"},
                },
                "required": ["command"],
            },
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "update_checklist",
            "description": "Mark an existing checklist item as pending/completed/failed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checklist": {"type": "string"},
                },
                "required": ["checklist"],
            },
        }
    },
]


# name -> handler
TOOL_HANDLERS = {}
for tool in TOOLS:
    if tool["type"] == "function":
        name = tool["function"]["name"]
        TOOL_HANDLERS[name] = (globals()[name])

# ---------------- Chat Loop with Tool Dispatch ----------------


def ask_ollama(messages: List[Dict[str, Any]]):
    logger.debug("Sending to Ollama model=%s", OLLAMA_MODEL)
    try:
        logger.debug("Outgoing messages: %s", json.dumps(messages, ensure_ascii=False))
    except Exception:
        logger.debug("Outgoing messages: %r", messages)
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=TOOLS,
        stream=False,
    )
    logger.debug("Raw Ollama response: %s", resp)
    return resp


def handle_tool_calls(resp: Dict[str, Any], messages: List[Dict[str, Any]]) -> bool:
    """Execute any tool calls; append tool results as messages. Return True if any tool was executed."""
    msg = resp.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    executed = False
    for tc in tool_calls:
        func = (tc.get("function") or {})
        name = func.get("name")
        args_raw = func.get("arguments")
        tool_call_id = tc.get("id") or tc.get("tool_call_id")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception as e:
            args = {"_parse_error": str(e), "raw": args_raw}
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            result = {"ok": False, "error": f"unknown tool: {name}"}
        else:
            try:
                result = handler(args)
            except Exception as e:
                result = {"ok": False, "error": f"tool raised: {e}"}
        messages.append(_tool_result_message(name, result, tool_call_id))
        executed = True
    return executed


def trim_history(messages: List[Dict[str, Any]]):
    # Keep system + last N user/assistant/tool exchanges
    max_len = 2 * MAX_TURNS + 1
    while len(messages) > max_len:
        # Don't drop the first system message
        del messages[1:3]


# ---------------- Main ----------------

def main():
    global CURRENT_CWD
    CURRENT_CWD = os.getcwd()

    print_hr()
    print("Ollama Coder (tools edition) — ingest + checklist + safe edits")
    print(f"Model: {OLLAMA_MODEL}   Endpoint: {CHAT_ENDPOINT}")
    print("Type 'exit' or 'quit' to leave. Press Ctrl-C to cancel a running command.")
    print(f"Ingest cap: {MAX_INGEST_BYTES} bytes")
    print("Meta commands: ':checklist', ':cd <path>' (local), ':pwd' (local)")
    print_hr()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    def show_checklist():
        print_hr(); print(CHECKLIST); print_hr()

    while True:
        try:
            user_in = input("\nagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); return

        if user_in.lower() in {"exit", "quit"}:
            print("Goodbye!"); return
        if not user_in:
            user_in = "Proceed with checklist"

        # Simple local meta commands (non-AI)
        if user_in.startswith(":"):
            parts = user_in.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""
            if cmd == ":checklist":
                show_checklist(); continue
            if cmd == ":messages":
                print(messages); continue
            if cmd == ":cd":
                path = arg or os.path.expanduser("~")
                try:
                    os.chdir(path); CURRENT_CWD = os.getcwd()
                    print(f"cwd -> {CURRENT_CWD}")
                except Exception as e:
                    print(f"[error] {e}")
                continue
            if cmd == ":pwd":
                print(os.getcwd()); continue
            # fallthrough: treat as normal user input

        messages.append({"role": "user", "content": user_in})

        # Chat/tool loop: keep invoking until the model stops calling tools
        step_guard = 0
        while True:
            step_guard += 1
            if step_guard > 12:
                print("[warn] too many tool steps; stopping")
                break

            resp = ask_ollama(messages)
            messages.append(resp.get("message", {}))

            if handle_tool_calls(resp, messages):
                trim_history(messages)
                # model asked for tools; continue the loop to let it see results
                continue

            # No tool calls -> print assistant content
            content = resp.get("message", {}).get("content", "")
            if content:
                print_hr(); print(content); print_hr()
            break

        trim_history(messages)


if __name__ == "__main__":
    main()
