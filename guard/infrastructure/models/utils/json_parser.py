import re
import json

def parse_json_markdown(text: str):
    if not text:
        raise ValueError("The input string is empty.")

    text = text.strip()

    markdown_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    match = markdown_pattern.search(text)

    if match:
        json_content = match.group(1).strip()
    else:
        json_content = text

    json_start_pattern = re.compile(r"([\{\[])")
    start_match = json_start_pattern.search(json_content)
    
    if start_match:
        start_index = start_match.start()
        closer = "}" if json_content[start_index] == "{" else "]"
        end_index = json_content.rfind(closer)
        
        if end_index != -1:
            json_content = json_content[start_index : end_index + 1]

    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            msg=f"Failed to decode the extracted JSON. Raw content extracted: {json_content}. Original error: {e.msg}",
            doc=e.doc,
            pos=e.pos
        )
