import json
import os
import subprocess
import sys
import re
import requests
from typing import Optional

# Configuration ---------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"  # change if you are using a different model
# -----------------------------------------------------------------------------


def ollama_chat(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Send a prompt to the Ollama chat endpoint and return the model's reply.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can answer code related questions "
                    "and, if appropriate, propose a single terminal command that "
                    "solves the user's request.  When you propose a command, wrap it "
                    "in a fenced code block with the language \"sh\" (````sh ...````). "
                    "If no command is needed, just reply with plain text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # We want a plain text reply – no structured format needed here.
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()
        # The reply text is under resp["message"]["content"]
        content = resp["message"]["content"]
        return content.strip()
    except Exception as e:
        print(f"[ERROR] Failed to contact Ollama: {e}")
        return ""


def extract_command(reply: str) -> Optional[str]:
    """
    Try to extract a command from the model's reply.

    - First, look for a fenced sh block.
    - If not found, treat the reply as a single command if it contains
      typical shell characters.
    """
    # Look for ```sh ... ```
    code_block = re.search(r"```sh\s*(.*?)\s*```", reply, re.DOTALL)
    if code_block:
        cmd = code_block.group(1).strip()
        if cmd:
            return cmd

    # As a fallback, try to parse the reply as a single line command.
    # We assume that if the reply starts with a known command keyword or
    # contains spaces or pipe/redirect symbols, it is a command.
    first_line = reply.splitlines()[0].strip()
    if first_line and (
        re.search(r"^\w+(\s|$)", first_line) or
        re.search(r"[|&;<>\$]", first_line)
    ):
        return first_line

    return None


def run_command(cmd: str) -> None:
    """
    Execute the command in a shell and print its output (stdout and stderr).
    """
    print(f"\nExecuting: {cmd}\n{'-' * (len(cmd) + 20)}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}", file=sys.stderr)
    print("-" * 80 + "\n")


def main() -> None:
    print("Ollama Coding Assistant")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Ask Ollama
        reply = ollama_chat(user_input)
        if not reply:
            print("[WARN] No reply from Ollama. Try again.")
            continue

        # Try to extract a command
        cmd = extract_command(reply)

        if cmd:
            # Show the proposed command
            print(f"Command proposed: {cmd}")
            while True:
                choice = input("Accept (a), modify (m), reject (r)? ").strip().lower()
                if choice == "a":
                    run_command(cmd)
                    break
                elif choice == "m":
                    new_cmd = input("Enter the modified command: ").strip()
                    if new_cmd:
                        cmd = new_cmd
                        run_command(cmd)
                        break
                    else:
                        print("[WARN] Empty command. Try again.")
                elif choice == "r":
                    print("Command rejected. No action taken.\n")
                    break
                else:
                    print("Please choose 'a', 'm', or 'r'.")
        else:
            # Just a textual answer
            print("\nAnswer:")
            print(reply)
            print()


if __name__ == "__main__":
    main()

