def extract_json_payload(response: str) -> str:
    """Remove an optional Markdown JSON fence from an LLM response."""
    stripped = response.strip()
    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    opening = lines[0].strip().casefold()
    closing = lines[-1].strip()
    if opening in {"```", "```json"} and closing == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped
