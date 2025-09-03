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
import difflib
import pprint
import tiktoken
import argparse
from datetime import datetime
from colorama import Fore, Style
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "12"))
MAX_INGEST_BYTES = int(os.getenv("AGENT_MAX_INGEST_BYTES", str(200 * 1024)))
CHECKLIST_PATH = '.all_hands/checklist.txt'

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("main_agent")

client = ollama.Client(host=OLLAMA_HOST)
CHAT_ENDPOINT = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

SYSTEM_PROMPT = (
    "You are an aggressive coding & shell assistant running in a terminal.\n"
    "I will give you a task to complete.\n"
    "I am not going to help in the completion of this task.\n"
    "You must complete this task by calling tools.\n"
    "If you complete a task, such as creating a file,\n"
    "you must verify that the file was created by calling a shell command.\n"
    "General rules:\n"
    "- Read files before making edits.\n"
    "- Keep edits minimal; avoid collateral changes.\n"
    "- Make sure to check the contents of a directory before attempting to add, edit or remove files.\n"
    "- Always try using tools first, unless a task needs further clarification.\n"
    "When you are given a task, break the task into any number of smaller tasks.\n"
    "Create a checklist for these tasks in the following list-of-maps-style format: \n"
    "[\n"
    "    {<pending|completed|failed>: <sub_task_1_description>},\n"
    "    {<pending|completed|failed>: <sub_task_2_description>},\n"
    "    ...\n"
    "]\n"
    "When a task has been updated, use the update_checklist tool.\n"
    "When all tasks have been completed, update the checklist to and empty list\n"
    "Your first course of action should either be to create the checklist by calling\n"
    "update_checklist or asking a question to the user to further clarify the task.\n"
    "For any failed tasks, output a possible explanation of why the task failed.\n"
    "Propose possible solutions. If no solutions are available,\n"
    "suggest some approaches on how to investigate or troubleshoot the issue.\n"
    "These can include terminal commands, checking software versions, ect\n"
    "You may need to use commands to find files, move around between\n"
    "directories, or list the contents of directories.\n"
    "You may propose suggestions for new tools if you think one would be helpful or necessary\n"
    "Editing files:\n"
    "    The only way you may edit the contents of a file is by creating a git patch.\n"
    "    Here are the necessary steps:\n"
    "    1. Make sure you have the whole contents of the file in context. That is, it must be visible in the chat history.\n"
    "    2. If the file contents are not inside the chat history, read the contents of the file by calling a command like \"cat\".\n"
    "    3. Generate a git patch file using terminal commands with the proposed changes.\n"
    "    4. Create a temp file called <file_name>.<extension>.bak\n"
    "    5. Copy the contents of the original file into the temporary file.\n"
    "    6. Run apply the git patch to the original file.\n"
    "    7. Run \"git diff\" from the terminal to show the output to the user.\n"
    "    8. Ask the user if the changes are ok.\n"
    "    9. If the user approves the changes, delete the temp file.\n"
    "   10. If the user rejects the changes, copy the contents from the temp file back into the original file.\n"
)

# ---------------- Helpers ----------------
def create_all_hands_folder():
    """Creates the folder '~/.all_hands' if it doesn't already exist."""
    home_dir = os.path.expanduser("~")
    folder_path = os.path.join(home_dir, ".all_hands")

    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print("Folder '~/.all_hands' created successfully.")
        except OSError as e:
            print(f"Error creating folder: {e}")
    else:
        print("Folder '~/.all_hands' already exists.")


def create_checklist_file():
    """Creates the file 'checklist.txt' if it doesn't already exist."""
    print('creating checklist file')
    home_dir = os.path.expanduser("~")
    print(f"home_dir: {home_dir}")
    file_path = os.path.join(home_dir, CHECKLIST_PATH)
    print(f"creating checklist file at: {file_path}")
    initial_checklist = (
      "Checklist:\n"
      "[\n"
      "  {pending: Clarify your first task},\n"
      "  {pending: Break this task into smaller tasks and call update_checklist}\n"
      "]"
    )

    if not os.path.exists(file_path):
        try:
            with open(file_path, "w") as checklist_file:
                checklist_file.write(initial_checklist)
                print("File 'checklist.txt' created successfully.")
        except OSError as e:
            print(f"Error creating file: {e}")
    else:
        print("File 'checklist.txt' already exists.")


def get_checklist_from_file():
    """Retrieves the checklist from the file."""
    home_dir = os.path.expanduser("~")
    file_path = os.path.join(home_dir, CHECKLIST_PATH)
    checklist = "[]"
    with open(file_path, 'r') as checklist_file:
        checklist = checklist_file.read()
        return checklist

def get_project_context():
    """
    Gathers contextual information for an AI agent to start working on a project.

    Args:
        project_name (str, optional): The name of the project. Defaults to "".

    Returns:
        dict: A dictionary containing the project context.
    """

    context = {
        "current_working_directory": os.getcwd(),
        "current_directory_name": os.path.basename(os.getcwd()),
        "current_directory_path": os.path.abspath(os.getcwd()),
        "list_of_files": [
            f for f in os.listdir(os.getcwd())
            if os.path.isfile(os.path.join(os.getcwd(), f))]
    }

    return json.dumps(context)

def get_additional_context() -> str:
    """
    Reads and returns the contents of AGENTS.md file in the same directory.
    
    Returns:
        str: The contents of the AGENTS.md file.
    """

    # Get the absolute path to the current directory
    current_dir = os.getcwd()
    
    # Construct the full path to the AGENTS.md file
    agents_path = current_dir + '/AGENTS.md'
    print(f"agents_path: {agents_path}")
    
    try:
        # Read and return the contents of the AGENTS.md file
        with open(agents_path, 'r') as f:
            additional_context = f.read()
        return additional_context
    
    except FileNotFoundError:
        print("The AGENTS.md file was not found in the current directory.")
        return None


# Read and display the contents of AGENTS.md file on initial agent prompt
additional_context = get_additional_context()
if additional_context:
    SYSTEM_PROMPT += "Directory specific context:\n" \
        + additional_context + "\n"


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


# edit a file
def edit_file(args: Dict[str, Any]) -> Dict[str, Any]:
    file = args.get("file")
    contents = args.get("contents")
    if not file:
        return {"ok": False, "error": "file is required"}
    if not contents:
        return {"ok": False, "error": "contents is required"}
    print(f"proposed file edit:\n{contents}")
    print("Does this look right?  y/n\n")
    choice = prompt_agent()
    if choice != "y":
        print("What is wrong with this suggestion?\n")
        reason = prompt_agent()
        return {"ok": False, "message": reason}

    # show a diff
    try:
        # 1. Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        print(f"Created temporary file {temp_file_path}")

        # 2. Copy the contents of the input file to the temp file
        shutil.copy2(file, temp_file_path)
        print(f"Copied contents from {file} to {temp_file_path}")

        # 3. Put new changes inside original file
        with open(file, 'w') as replace_contents:
            replace_contents.write(contents)

        # 4. show diff
        print("\n--- Diff ---")
        with open(file, 'r') as f1, open(temp_file_path, 'r') as f2:
            diff = difflib.unified_diff(
                f1.readlines(), f2.readlines(),
                fromfile=temp_file_path, tofile=file)
            for line in diff:
                print(line)
        print("--- End Diff ---")

        # 6. Excecute changes
        print('Apply the changes? y/n\n')
        choice = prompt_agent()
        if choice == 'y':
            return {'ok': True, "message": "Successfully executed contents"}
        else:
            # copy the original contents back into the original file
            shutil.copy2(temp_file_path, file)
            print('What is the reason for rejection?')
            reason = prompt_agent()
            return {'ok': False, "reasonn for rejection": reason}
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing contents: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# edit a file
def edit_file_with_git_patch(args: Dict[str, Any]) -> Dict[str, Any]:
    file = args.get("file")
    git_patch = args.get("git_patch")
    if not file:
        return {"ok": False, "error": "file is required"}
    if not git_patch:
        return {"ok": False, "error": "git_patch is required"}
    print(f"proposed file edit:\n{git_patch}")
    print("Does this look right?  y/n\n")
    choice = prompt_agent()
    if choice != "y":
        print("What is wrong with this suggestion?\n")
        reason = prompt_agent()
        return {"ok": False, "message": reason}

    # show a diff
    try:
        # 1. Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        print(f"Created temporary file {temp_file_path}")

        # 2. Copy the git_patch of the input file to the temp file
        shutil.copy2(file, temp_file_path)
        print(f"Copied git_patch from {file} to {temp_file_path}")

        # 3. Put new changes inside original file
        with open(file, 'w') as replace_contents:
            replace_contents.write(git_patch)

        # 4. show diff
        print("\n--- Diff ---")
        with open(file, 'r') as f1, open(temp_file_path, 'r') as f2:
            diff = difflib.unified_diff(
                f1.readlines(), f2.readlines(),
                fromfile=temp_file_path, tofile=file)
            for line in diff:
                print(line)
        print("--- End Diff ---")

        # 6. Excecute changes
        print('Apply the changes? y/n\n')
        choice = prompt_agent()
        if choice == 'y':
            return {'ok': True, "message": "Successfully executed git_patch"}
        else:
            # copy the original git_patch back into the original file
            shutil.copy2(temp_file_path, file)
            print('What is the reason for rejection?')
            reason = prompt_agent()
            return {'ok': False, "reasonn for rejection": reason}
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing contents: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# read file
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
    # print(f"lines: {lines}")

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

    print(f"read text:\n\n {text}")

    return {
        "ok": True, "path": path, "start": s, "end": e,
        "truncated": truncated, "content": text
    }


# create a new file
def create_file(args: Dict[str, Any]) -> Dict[str, Any]:
    filepath = args.get("filepath")
    contents = args.get("contents")
    if not filepath:
        return {"ok": False, "error": "filepath is required"}
    if not contents:
        return {"ok": False, "error": "contents is required"}

    print(f"{contents}\n\n{filepath}\n\nCreate new file?  y/n\n")
    choice = prompt_agent()
    if choice == 'y':
        try:
            with open(filepath, "w") as f:
                f.write(contents)
                print(f"File '{filepath}' created successfully.")
            return {"ok": True, "message": "File successfully created"}
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"ok": False, "error": f"exception creating file: {e}"}
    else:
        return {"ok": False, "message": "User chose not to create file"}



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


def command_requires_approval(agent_command):
    allowed_commands = ['sed', 'ls', 'cat', 'grep']
    for cmd in allowed_commands:
        if agent_command.startswith(cmd):
            return False
    return True


# 5) run_shell_command_tool (for read‑only or safe commands)
def run_shell_command_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    command = args.get("command")
    # mode = args.get("mode", "capture")  # capture|stream
    if not isinstance(command, str) or not command.strip():
        return {"ok": False, "error": "command (string) is required"}

    if command_requires_approval(command):
        print("Proposed command:")
        print(f"  {command}")
        print("Run this command? [y/N]: ")
        choice = prompt_agent()
        if choice.strip().lower() != 'y':
            print('What is the reason for rejecting the command')
            reason = prompt_agent()
            return {"ok": True, "ran": False, "message": reason}
    else:
        print(f"Running allowed command: {command}")

    # if mode == "stream":
    # code = run_command_stream(command, CURRENT_CWD)
    # return {"ok": True, "ran": True, "exit_code": code}
    # else:
    code, out = run_command_capture(command, CURRENT_CWD)
    return {"shell command": command, "output": out}

# 6) checklist tools

def update_checklist(args: Dict[str, Any]) -> Dict[str, Any]:
    checklist = args.get("checklist")
    print(f"Updating checklist to {checklist}")
    if not checklist:
        return {"ok": False, "error": "checklist is required"}
    checklist_header = "Here is your checklist of tasks:\n"
    home_dir = os.path.expanduser("~")
    file_path = os.path.join(home_dir, CHECKLIST_PATH)
    with open(file_path, 'w') as checklist_file:
        checklist_file.write(checklist_header + checklist)
    return {"ok": True, "message": "checklist successfully updated"}

# ---- Tool registry ----


TOOLS = [
    {
        "type": "function",
        "function":
        {
            "name": "edit_file",
            "description": "Edit a single file. If you need to make any edits to a file, this is your only way to do it. Do not call this tool unless you have read the contents of the file first.",
            "parameters":
            {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "contents": {
                        "type": "string",
                        "description": "This replaces the ENTIRE contents of the file."
                    }
                },
                "required": ["file", "contents"],
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
            "name": "create_file",
            "description": "Create a new file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["filepath", "contents"]
            }
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
            "name": "update_checklist",
            "description": "Updates the entire contents of the checklist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checklist": {
                        "type": "string",
                        "description": "json string representation of the entire checklist"},
                },
                "required": ["checklist"],
            },
        }
    },
]

TERMINAL_TOOLS = [
    {
        "type": "function",
        "function":
        {
            "name": "update_checklist",
            "description": "Updates the entire contents of the checklist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checklist": {
                        "type": "string",
                        "description": "json string representation of the entire checklist"},
                },
                "required": ["checklist"],
            },
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "run_shell_command_tool",
            "description": "Run a shell command. Some commands will require user approval. The following list of commands do not require approval: sed, ls, cat, grep",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
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

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def choose_num_ctx(messages, max_output_tokens=512, model_max_ctx=128000):
    input_text = " ".join(m["content"] for m in messages)
    input_tokens = count_tokens(input_text)
    
    num_ctx = input_tokens + max_output_tokens
    ctx_pct = (num_ctx / model_max_ctx) * 100
    return min(num_ctx, model_max_ctx), ctx_pct


def notify(title, message):
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])


def ask_ollama(messages: List[Dict[str, Any]]):
    # print("Sending to Ollama model=%s", OLLAMA_MODEL)
    try:
        logger.debug("Outgoing messages: %s", json.dumps(messages, ensure_ascii=False))
    except Exception:
        logger.debug("Outgoing messages: %r", messages)
    num_token_ctx, ctx_pct = choose_num_ctx(messages)
    print(f"Number token context: {num_token_ctx}")
    print(f"Percent context: {ctx_pct}")
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=TERMINAL_TOOLS,
        keep_alive=0,
        options={"num_ctx": num_token_ctx},
        # think='high',
        stream=False,
    )
    # print("Raw Ollama response: %s", resp)

    notify("Python Script", "Your task finished successfully!")

    return resp


def handle_tool_calls(resp: Dict[str, Any], messages: List[Dict[str, Any]]) -> bool:
    """Execute any tool calls; append tool results as messages. Return True if any tool was executed."""
    msg = resp.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    executed = False
    # print(f"handling tool calls:\n{msg}\n{tool_calls}")
    for tc in tool_calls:
        print(f"handling tool call: {tc}")
        func = (tc.get("function") or {})
        name = func.get("name")
        args_raw = func.get("arguments")
        tool_call_id = tc.get("id") or tc.get("tool_call_id")
        print(f"Calling function {func}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            print(f"Function args: {args}")
        except Exception as e:
            args = {"_parse_error": str(e), "raw": args_raw}
        handler = None
        try:
            handler = globals()[name]
        except Exception as e:
            print(e)
            print(f"Unable to find handler for tool: {name}")
        print(f"handler: {handler}")
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
    sys_prompt = messages[0]
    checklist_msg = {"role": "system", "content": get_checklist_from_file()}
    dir_context = {"role": "system", "content": get_project_context()}
    messages = messages[3:]
    while len(messages) > max_len:
        del messages[0]
    trimmed_msgs = [sys_prompt] + [checklist_msg] + [dir_context] + messages
    return trimmed_msgs


def parse_resp_obj(message):
    simp_msg = {
        "role": message['role'],
        "content": message.get('content')
    }
    return simp_msg


# ---------------- Style----------------
def show_messages(messages):
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'cyan': '\033[96m',
        'red': '\033[91m'
    }
    reset_color = '\033[0m'

    for msg in messages:
        if not msg: continue
        color = None
        if msg['role'] == 'user':
            color = 'red'
        elif msg['role'] == 'system':
            color = 'cyan'
        elif msg['role'] == 'tool':
            color = 'blue'
        elif msg['role'] == 'assistant':
            color = 'green'
        else:
            raise Exception(f"Message without role: {msg}")
        try:
            print(colors[color] + json.dumps(msg) + reset_color)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    

def prompt_agent():
    try:
        user_in = input("\n" + Fore.RED + "agent> " + Style.RESET_ALL).strip()
        if user_in.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            exit()
        return user_in
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); return
    

# --------------Command line args-------

def handle_arguments():
    """
    Handles any initial command line arguments the tool is called with.
    """
    parser = argparse.ArgumentParser(
        description="Manage ~/.all_hands/context.txt"
    )
    parser.add_argument(
        "-c", "--context", metavar="FILE",
        help="Local file to append to context file"
    )
    parser.add_argument(
        "-d", "--delete", action="store_true",
        help="Clear the contents of the context file"
    )
    args = parser.parse_args()

    # Expand ~ and build context file path
    context_dir = os.path.expanduser("~/.all_hands")
    context_file = os.path.join(context_dir, "context.txt")

    # Ensure directory exists
    os.makedirs(context_dir, exist_ok=True)

    if args.delete:
        try:
            with open(context_file, "w") as outfile:
                pass  # truncate the file
            print(f"Cleared contents of '{context_file}'")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        exit()

    if args.context:
        if not os.path.isfile(args.context):
            print(f"Error: File '{args.context}' not found", file=sys.stderr)
            sys.exit(1)

        try:
            abs_path = os.path.abspath(args.context)
            file_name = os.path.basename(abs_path)

            with open(args.context, "r") as infile:
                contents = infile.read()

            with open(context_file, "a") as outfile:
                outfile.write("`````````````````````````````````````````Begin File Context```````````````````````````````````````````")
                outfile.write(f"File name: {file_name}\n")
                outfile.write(f"File path: {abs_path}\n")
                outfile.write("\n\n")  # add spacing between entries
                outfile.write(contents)
                outfile.write("`````````````````````````````````````````End File Context```````````````````````````````````````````")
                outfile.write("\n\n")  # add spacing between entries

            print(f"Appended contents of '{args.context}' to '{context_file}'")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        exit()


# ---------------- Main ----------------

def main():

    handle_arguments()

    global CURRENT_CWD
    CURRENT_CWD = os.getcwd()
    create_all_hands_folder()
    create_checklist_file()

    print_hr()
    print("Ollama Coder (tools edition) — ingest + checklist + safe edits")
    print(f"Model: {OLLAMA_MODEL}   Endpoint: {CHAT_ENDPOINT}")
    print("Type 'exit' or 'quit' to leave. Press Ctrl-C to cancel a running command.")
    print(f"Ingest cap: {MAX_INGEST_BYTES} bytes")
    print("Meta commands: ':checklist', ':cd <path>' (local), ':pwd' (local)")
    print_hr()

    thinking = []
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": get_checklist_from_file()},
        {"role": "system", "content": get_project_context()},
    ]

    def show_checklist():
        print_hr(); print(get_checklist_from_file()); print_hr()

    while True:
        user_in = prompt_agent()

        if not user_in:
            user_in = "Proceed with executing checklist tasks"

        # Simple local meta commands (non-AI)
        if user_in.startswith(":"):
            parts = user_in.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""
            if cmd == ":think":
                pprint.pprint(thinking); continue
            if cmd == ":checklist":
                show_checklist(); continue
            if cmd == ":messages":
                show_messages(messages); continue
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
            messages.append(parse_resp_obj(resp.get("message", {})))
            thinking.append(resp.get("message").get("thinking"))

            if handle_tool_calls(resp, messages):
                # messages = trim_history(messages)
                # model asked for tools; continue the loop to let it see results
                continue

            # No tool calls -> print assistant content
            content = resp.get("message", {}).get("content", "")
            if content:
                print_hr(); print(content); print_hr()
            break

        # messages = trim_history(messages)


if __name__ == "__main__":
    main()
