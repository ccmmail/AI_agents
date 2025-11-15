"""Code editing Agent example."""

from openai import OpenAI
from typing import Callable, Tuple, List, Dict, Any
from dotenv import load_dotenv
from dataclasses import dataclass
import os, sys
import json
from pathlib import Path

DEBUG_TOOLS=True  # set to false to disable debug prints

# load environment variables from .env file
load_dotenv(".env", override=False)

def dprint(*args, **kwargs):
    """Debug print to stderr if DEBUG_TOOLS=1 is set."""
    if DEBUG_TOOLS:
        print("[debug]", *args, file=sys.stderr, **kwargs)


def get_user_message() -> Tuple[str, bool]:
    """Read one line from stdin, and returns (text, true/false)."""
    try:
        line = input()
        return line, True
    except EOFError:
        return "", False


@dataclass
class ToolDefinition:
    """Defines a tool the agent can use."""
    type: str
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for parameters
    function: Callable[[Dict[str, Any]], str]  # executes and returns string

    def to_openai(self) -> Dict[str, Any]:
        """Convert dataclass model to OpenAI tool definition."""
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            # "strict": True,
        }


def read_file_tool(args: Dict[str, Any]) -> str:
    """Read and return the contents of a *relative* file path in the CWD."""
    MAX_READ_BYTES = 200_000  # safety cap to avoid dumping huge files

    raw = args.get("path", "")
    if not isinstance(raw, str) or not raw:
        return "read_file error: 'path' must be a non-empty string"

    # Resolve path safely within the current working directory
    base = Path.cwd().resolve()
    target = (base / raw).resolve()

    # Disallow leaving the base directory or absolute paths
    try:
        target.relative_to(base)
    except ValueError:
        return "read_file error: path must be relative to the working directory"

    if not target.exists():
        return f"read_file error: no such file: {raw}"
    if not target.is_file():
        return "read_file error: path refers to a directory, not a file"

    data = target.read_bytes()
    if len(data) > MAX_READ_BYTES:
        snippet = data[:MAX_READ_BYTES].decode(errors="replace")
        return (
            f"read_file warning: file larger than {MAX_READ_BYTES} bytes; returning first chunk."
            + "\n"
            + snippet
        )
    return data.decode(errors="replace")



def list_files_tool(args: Dict[str, Any]) -> str:
    """List files and directories under an optional relative path, recursively.
    Returns a JSON array string of relative paths (directories end with '/').
    Mirrors the Go ListFiles tool.
    """
    # Resolve base directory (CWD) and target (default '.')
    base = Path.cwd().resolve()
    raw = args.get("path", "")
    target = (base / raw).resolve() if raw else base

    # Disallow leaving the base directory
    try:
        target.relative_to(base)
    except ValueError:
        return "list_files error: path must be relative to the working directory"

    if not target.exists():
        return f"list_files error: no such path: {raw or '.'}"
    if not target.is_dir():
        return "list_files error: path must be a directory"

    results: List[str] = []
    # Walk recursively; collect entries relative to 'target'
    for root, dirs, files in os.walk(target):
        root_path = Path(root)
        # Sort for stable order
        dirs.sort()
        files.sort()
        # For directories (excluding the root itself), append with trailing '/'
        rel_dir = root_path.relative_to(target)
        if str(rel_dir) != ".":
            results.append(str(rel_dir) + "/")
        # For files at this level
        for fname in files:
            full = root_path / fname
            rel = full.relative_to(target)
            results.append(str(rel))

    # Return JSON array as string (not Python object), like the Go implementation
    try:
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"list_files error: could not serialize results: {e}"


def edit_file_tool(args: Dict[str, Any]) -> str:
    """Edit (or create) a text file.
      - Requires a non-empty 'path'.
      - 'old_str' must not equal 'new_str'.
      - If file exists: replace all occurrences of old_str with new_str (Python's replace, count=-1).
      - If file does NOT exist and old_str == "": create the file with new_str as content.
      - If no replacement occurred in an existing file and old_str != "": return error.
    Returns "OK" on success, or an error string prefixed with 'edit_file error: ...'.
    """
    # Validate inputs
    path = args.get("path", "")
    old_str = args.get("old_str", None)
    new_str = args.get("new_str", None)

    if not isinstance(path, str) or path.strip() == "":
        return "edit_file error: 'path' must be a non-empty string"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "edit_file error: 'old_str' and 'new_str' must be strings"
    if old_str == new_str:
        return "edit_file error: invalid input parameters (old_str equals new_str)"

    # Resolve path safely within current working directory (like read_file)
    base = Path.cwd().resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return "edit_file error: path must be relative to the working directory"

    # If file does not exist
    if not target.exists():
        if old_str == "":
            try:
                target.write_text(new_str, encoding="utf-8")
                return "OK"
            except Exception as e:
                return f"edit_file error: {e}"
        else:
            return f"edit_file error: no such file: {path}"

    # File exists -> read, replace, validate, write
    try:
        old_content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"edit_file error: {e}"

    # Follow Go semantics: Replace all occurrences. Note: replacing with old_str=="" in Python
    # will insert new_str between characters (mirrors Go's strings.Replace with old="").
    new_content = old_content.replace(old_str, new_str)

    if old_content == new_content and old_str != "":
        return "edit_file error: old_str not found in file"

    try:
        target.write_text(new_content, encoding="utf-8")
        return "OK"
    except Exception as e:
        return f"edit_file error: {e}"


class Agent:
    """Represents tool-using conversational agent."""
    def __init__(self, client: OpenAI,
                 get_user_message: Callable[[], Tuple[str, bool]],
                 tools: List[ToolDefinition] | None = None):
        self.client = client
        self.get_user_message = get_user_message
        self.tools = tools or []
        self.openai_tools = [t.to_openai() for t in self.tools]
        self.history = [
            {"role": "system",
             "content": "You are a helpful coding assistant. You have access to tools. "
            }
        ]

    def run(self):
        """Runs the agent conversation loop."""
        print("Chat with GPT (use 'ctrl-c' to quit)")
        while True:
            # Prompt prefix with blue label
            print("\u001b[34mYou\u001b[0m: ", end="", flush=True)
            user_text, ok = self.get_user_message()
            if not ok:
                break  # EOF/Ctrl-D

            # Append user input to history
            self.history.append({"role": "user", "content": user_text})
            dprint("user said:", user_text)

            # Call the model and save assistant text (if any) to history
            # tool history is updated from inside _run_inference()
            assistant_text = self._run_inference()
            if assistant_text:
                self.history.append({"role": "assistant", "content": assistant_text})

            # Print assistant reply with yellow label
            print("\u001b[31mGPT\u001b[0m: ", assistant_text, flush=True)

    def _run_inference(self) -> str:
        """Calls the model with the current conversation state."""
        loop = 0
        tool_calls = 0
        while True:
            # call client
            resp = self.client.responses.create(
                model="gpt-5-mini",
                input=self.history,
                tools=self.openai_tools if self.openai_tools else None,
                max_output_tokens=1024,
                # stream=True,
            )

            # debug prints of inference calls
            loop += 1
            dprint(f"=== Inference {loop=}, {tool_calls=} ===")
            for i, item in enumerate(getattr(resp, "output", [])):
                typ = getattr(item, "type", None)
                name = getattr(item, "name", None)
                arguments = getattr(item, "arguments", None)
                dprint(f"output[{i}]: {typ=}, {name=}, {arguments=}")

            made_tool_call = False
            for item in resp.output:
                if item.type == "function_call":
                    # save LLM's request for function call to history
                    self.history.append({
                        "type": item.type,
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })

                    # call the requested tool
                    for tool in self.tools:
                        if tool.name == item.name:
                            try:
                                tool_output = tool.function(json.loads(item.arguments))
                            except Exception as e:
                                tool_output = f"Tool '{tool.name}' error: {e}"
                            dprint("tool result length:", len(tool_output))

                            # append tool output to history
                            self.history.append({
                                "type" : "function_call_output",
                                "call_id" : item.call_id,
                                "output": json.dumps(tool_output),
                            })
                            tool_calls += 1
                            made_tool_call = True
                            break

            if made_tool_call:
                continue

            # no more tool calls, return the user-facing message
            return resp.output_text or ""


def main():
    tools: List[ToolDefinition] = [
        ToolDefinition(
            type="function",
            function=read_file_tool,
            name="read_file",
            description=(
                "Read the contents of a given relative file path in the working directory. "
                "Use this when you want to see what's inside a file. Do not use directory names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path from the current working directory.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            type="function",
            function=list_files_tool,
            name="list_files",
            description=(
                "List files and directories at a given path. If no path is provided, "
                "list files in the current directory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": ("optional relative path to list files from."
                                       "Defaults to current directory if not provided."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            type="function",
            function=edit_file_tool,
            name="edit_file",
            description=(
                "Make edits to a text file." 
                "Replaces 'old_str' with 'new_str' in the given file."
                "'old_str' and 'new_str' MUST be different from each other."
                "If the file specified with path doesn't exist and 'old_str' is empty,"
                "the file will be created with 'new_str' as its content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path from the current working directory.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Text to search for - must match exactly and must only have one match exactly.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Text to replace old_str with",
                    },
                },
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
            },
        ),
    ]

    # create agent with tools
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    agent = Agent(client, get_user_message, tools=tools)
    try:
        agent.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
