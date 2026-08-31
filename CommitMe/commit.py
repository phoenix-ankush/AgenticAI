from dotenv import load_dotenv

load_dotenv()

import json
import os
import subprocess

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant that reviews staged changes in a git repository and commits them with a well-written commit message.
You will be given a staged diff and you should generate a concise and descriptive commit message in the conventional commit format: type(scope): summary. Use imperative mood and be clear and concise.
If the diff is empty or there are no staged changes, respond with "No staged changes to commit."
Do not make up information. Be concise and clear in your response.

You always work in a loop to finish the task pipeline as below.

Rules:
1. You will always follow the steps in the order provided below.
2. You will always return the json response strictly.


Steps: Initialise | Think | Call Tool | Receive Tool Output |  Finish

Tools: staged_diff | write_commit_message | commit_with_message | push_changes

"staged_diff": Get the staged diff of the current git repository.
"write_commit_message": Generate a commit message based on the staged diff using OpenAI's API.
"commit_with_message": Commit the staged changes with the provided commit message.
"push_changes": Push the committed changes to the remote repository.

EXAMPLE

User: Commit my staged changes.

{"step": "Think", "content": "I need to see what's staged before I can describe it. I'll call staged_diff first."}
{"step": "Call Tool", "tool": "staged_diff", "input": ""}
{"step": "Receive Tool Output", "tool": "staged_diff", "output": "diff --git a/app.py b/app.py\n+ added retry logic to the API client"}
{"step": "Think", "content": "The diff adds retry logic to the API client. I'll generate a conventional commit message for that."}
{"step": "Call Tool", "tool": "write_commit_message", "input": "diff --git a/app.py ... added retry logic"}
{"step": "Receive Tool Output", "tool": "write_commit_message", "output": "feat(api): add retry logic to API client"}
{"step": "Think", "content": "I have a message. Now I'll commit the staged changes with it."}
{"step": "Call Tool", "tool": "commit_with_message", "input": "feat(api): add retry logic to API client"}
{"step": "Receive Tool Output", "tool": "commit_with_message", "output": "committed as a1b2c3d"}
{"step": "Think", "content": "Commit succeeded. Finally I'll push to the remote."}
{"step": "Call Tool", "tool": "push_changes", "input": ""}
{"step": "Receive Tool Output", "tool": "push_changes", "output": "pushed to origin/main"}
{"step": "Finish", "content": "Staged changes committed as a1b2c3d and pushed to origin/main."}

"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "staged_diff",
            "description": "Get the staged diff of the current git repository.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_commit_message",
            "description": "Generate a commit message based on the staged diff using OpenAI's API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                        "description": "The staged diff to generate a commit message for.",
                    }
                },
                "required": ["diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_with_message",
            "description": "Commit the staged changes with the provided commit message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The commit message to use for the commit.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_changes",
            "description": "Push the committed changes to the remote repository.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def staged_diff() -> str:
    """Get the staged diff of the current git repository."""
    result = subprocess.run(
        ["git", "diff", "--staged"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Error getting staged diff: {result.stderr}")
    return result.stdout


def write_commit_message(diff: str) -> str:
    """Generate a commit message based on the staged diff using OpenAI's API."""
    prompt = f"Write a concise and descriptive git commit message for the following diff:\n\n{diff}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Write a concise conventional-commit message for this diff. "
                "Format: type(scope): summary. One line, imperative mood.",
            },
            {"role": "user", "content": diff},
        ],
    )
    return response.choices[0].message.content.strip()


def commit_with_message(message):
    """Commit the staged changes with the provided commit message."""
    result = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Error committing changes: {result.stderr}")
    return result.stdout


def push_changes():
    """Push the committed changes to the remote repository."""
    result = subprocess.run(
        ["git", "push"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Error pushing changes: {result.stderr}")
    return result.stdout



def call_Model(messages, tools, tool_choice="auto"):
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice=tool_choice
    )
    return response.choices[0].message

def run_agent():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Review my staged changes and commit them well."}
    ]
    step = "Think"  # Initialize the first step
    while step != "Finish":
        if step == "Think":
            print("Thinking:", messages[-1].content)
        elif step == "Call Tool":
            print(f"Calling tool: {messages[-1].tool} with input: {messages[-1].input}")
        elif step == "Receive Tool Output":
            print(f"Received output from tool: {messages[-1].tool}: {messages[-1].output}")
        else:
            raise ValueError(f"Unknown step: {step}")

        message = call_Model(messages, tools=tools)
        messages.append(message)
        step = message.step


if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"Error: {e}")
