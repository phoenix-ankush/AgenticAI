from dotenv import load_dotenv

load_dotenv()

import json
import os
import subprocess

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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


def run_agent():
    messages = [
        {"role": "user", "content": "Review my staged changes and commit them well."}
    ]
    while True:
        print("calling API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
        )
        print("got response")
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")

            if function_name == "staged_diff":
                result = staged_diff()
            elif function_name == "write_commit_message":
                result = write_commit_message(arguments.get("diff", ""))
            elif function_name == "commit_with_message":
                result = commit_with_message(arguments.get("message", ""))
            elif function_name == "push_changes":
                result = push_changes()
            else:
                raise ValueError(f"Unknown function: {function_name}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"Error: {e}")
