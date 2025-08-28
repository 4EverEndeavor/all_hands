#!/usr/bin/env python3
"""
ollama_coder.py — a tiny terminal-based coding/command agent powered by the Ollama API.

Features
- Simple REPL: type a prompt, get a response.
- The model can propose a shell command (with an explanation) OR just answer questions.
- Before any command executes, you're prompted to Accept, Modify, or Reject.
- Shows command output (stdout/stderr) after execution.
- Keeps short conversation context for better follow-ups.

Requirements
- Python 3.8+
- Ollama running locally (default: http://127.0.0.1:11434) and a chat-capable model pulled (e.g., `llama3.1`).
  See: https://github.com/ollama/ollama

Optional:
- pip install requests  (recommended). If not available, script will fall back to urllib.

Usage
  $ python ollama_coder.py
  agent> list files in this folder
  …
  Proposed command: ls -la
  Accept [a] / Modify [m] / Reject [r]? a
  (runs command, prints output)

Environment variables
- OLLAMA_HOST     (default: http://127.0.0.1:11434)
- OLLAMA_MODEL    (default: llama3.1)
- AGENT_MAX_TURNS (default: 12) – how many past turns to keep in context
"""

import os
import re
import sys
import json
import shlex
import signal
import subprocess
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
MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "12"))  # keep context lean

CHAT_ENDPOINT = f"{OLLAMA_HOST.rstrip('/')}/api/chat"


SYSTEM_PROMPT = """You are a careful coding & shell assistant running in a terminal.
Decide whether the user wants:
1) A direct answer with code or explanation, OR
2) A proposed shell command to run locally.

ALWAYS reply in pure JSON (no prose) with this schema:
{
  "type": "answer" | "command",
  "answer": string,          // required when type = "answer"
  "command": string,         // required when type = "command"
  "explanation": string,     // brief reason why this command helps
  "cwd": string | null       // optional working dir for the command; null for current
}

Rules:
- Prefer "answer" when a command isn't necessary.
- If proposing a command, keep it minimal, safe, and deterministic.
- Never include destructive operations unless the user explicitly asked (and even then, explain the risk).
- Do NOT wrap JSON in markdown fences. Output must be valid JSON only.
"""

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
    """Call Ollama chat endpoint and return the assistant's message.content string."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # You can tune temperature / top_p here if desired
        # "options": {"temperature": 0.2}
    }
    data = http_post_json(CHAT_ENDPOINT, payload)
    # Standard Ollama /api/chat returns {"message": {"role": "assistant", "content": "..."}}
    try:
        return data["message"]["content"]
    except Exception:
        # Fallbacks for odd responses
        if "messages" in data and data["messages"]:
            return data["messages"][-1].get("content", "")
        raise RuntimeError(f"Unexpected Ollama response: {data}")

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Be forgiving: try to parse a JSON object from the model's output even if extra text sneaks in.
    Strategy: find the first '{' ... last '}' and parse.
    """
    text = text.strip()
    # Fast path: direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Heuristic extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # Attempt to fix stray trailing commas
            snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj

    raise ValueError("Could not parse JSON reply from the model.")

def validate_agent_reply(obj: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Validate and normalize the model's JSON.
    Returns: (type, answer, command, cwd)
    """
    t = obj.get("type")
    if t not in ("answer", "command"):
        raise ValueError("Missing or invalid 'type' in model reply.")

    if t == "answer":
        ans = obj.get("answer")
        if not isinstance(ans, str):
            raise ValueError("Missing 'answer' string for type='answer'.")
        return ("answer", ans, None, None)

    # type == "command"
    cmd = obj.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("Missing 'command' string for type='command'.")
    # explanation is optional to display; don't crash if missing
    cwd = obj.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        cwd = None
    return ("command", None, cmd.strip(), cwd)

def prompt_user_choice(prompt: str = "Accept [a] / Modify [m] / Reject [r]? ") -> str:
    while True:
        choice = input(prompt).strip().lower()
        if choice in ("a", "m", "r"):
            return choice
        print("Please type 'a' to accept, 'm' to modify, or 'r' to reject.")

def run_command(command: str, cwd: Optional[str]) -> int:
    """
    Execute the command via the user's shell. Streams output.
    Returns the exit code.
    """
    print(f"\n[exec] {command}\n")
    try:
        # Use the user's default shell
        shell = os.environ.get("SHELL", "/bin/bash")
        proc = subprocess.Popen(
            command,
            cwd=cwd or None,
            shell=True,
            executable=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Stream output line by line
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

def print_hr():
    print("-" * 60)

def main():
    print_hr()
    print("Ollama Coder — terminal agent")
    print(f"Model: {OLLAMA_MODEL}   Endpoint: {CHAT_ENDPOINT}")
    print("Type 'exit' or 'quit' to leave. Press Ctrl-C to cancel a running command.")
    print_hr()

    # Conversation state: keep a small rolling window
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    def add_turn(user_msg: str, assistant_msg: str):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        # Trim old turns to keep context small
        # (2 msgs per turn, plus the system message)
        while len(messages) > (2 * MAX_TURNS + 1):
            # keep system message, drop oldest user+assistant pair
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

        # Add the user's prompt (but don't add assistant yet)
        messages.append({"role": "user", "content": user_in})

        try:
            raw = ask_ollama(messages)
        except Exception as e:
            print(f"[error] Failed to contact Ollama: {e}")
            # remove the last user message so context stays consistent
            messages.pop()
            continue

        # We won't add the assistant message to context until we've parsed it
        try:
            obj = extract_json_object(raw)
            t, answer, command, cwd = validate_agent_reply(obj)
        except Exception as e:
            print("[warn] Model did not return valid JSON as instructed.")
            print("Raw reply:\n")
            print(raw)
            # Add the raw reply so the model can self-correct next turn
            messages.append({"role": "assistant", "content": raw})
            continue

        if t == "answer":
            # Show the answer plainly
            print_hr()
            print(answer)
            print_hr()
            # Add parsed JSON (as text) so the model sees what we showed
            messages.append({"role": "assistant", "content": json.dumps(obj)})
            continue

        # t == "command"
        explanation = obj.get("explanation") or ""
        print_hr()
        if explanation:
            print(f"Proposed command rationale: {explanation}")
        print("Proposed command:")
        print(f"  {command}")
        if cwd:
            print(f"Working directory: {cwd}")
        print_hr()

        choice = prompt_user_choice()
        if choice == "r":
            print("[info] Command rejected.")
            # Let the model see that we rejected, to adjust future proposals
            feedback = {
                "type": "answer",
                "answer": "User rejected the proposed command.",
            }
            messages.append({"role": "assistant", "content": json.dumps(feedback)})
            continue

        if choice == "m":
            new_cmd = input("Enter modified command: ").strip()
            if not new_cmd:
                print("[info] Empty command; skipping.")
                messages.append({"role": "assistant", "content": json.dumps({
                    "type": "answer",
                    "answer": "User canceled after choosing modify."
                })})
                continue
            command = new_cmd

        # Accept or modified: run it
        exit_code = run_command(command, cwd)
        print_hr()
        print(f"[exit status] {exit_code}")
        print_hr()

        # Add what we did to context
        run_summary = {
            "type": "answer",
            "answer": f"Executed command: {command}\nExit code: {exit_code}"
        }
        messages.append({"role": "assistant", "content": json.dumps(run_summary)})

if __name__ == "__main__":
    # Graceful Ctrl-C behavior in the main REPL
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")

