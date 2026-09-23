from fastmcp import FastMCP
import subprocess

mcp = FastMCP("git-agent")

@mcp.tool
def staged_diff() -> str:
    """Get the diff of currently staged changes."""
    return subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True).stdout

@mcp.tool
def commit(message: str) -> str:
    """Commit staged changes with the given message."""
    subprocess.run(["git", "commit", "-m", message])
    return "committed"

if __name__ == "__main__":
    mcp.run()