from dotenv import load_dotenv

load_dotenv()

import os
import subprocess

from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

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
    response = get_client().chat.completions.create(
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
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI did not return a commit message.")
    return content.strip()



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
    diff = staged_diff()
    if not diff.strip():
        print("No staged changes to commit.")
        return

    print("Generating commit message from staged diff...")
    message = write_commit_message(diff)
    print(f"Generated commit message: {message}")

    print("Creating commit...")
    commit_output = commit_with_message(message).strip()
    print(commit_output)

    print("Pushing changes...")
    push_output = push_changes().strip()
    print(push_output)


if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"Error: {e}")
