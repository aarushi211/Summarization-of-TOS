import re

_RE_SANITISE = re.compile(r'[^\w\s\.\,\-\(\)\/\&]')

def sanitise_label(value: str, max_length: int = 100) -> str:
    """
    Strip characters that could be used for prompt injection from user-supplied
    labels (service_name, doc_type) before they are interpolated into LLM prompts.
    """
    if not value or not value.strip():
        return "Unknown"
    cleaned = _RE_SANITISE.sub('', value).strip()
    return cleaned[:max_length] if cleaned else "Unknown"

def clean_text(text: str) -> str:
    """Standard cleaning for legal text."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()

# ── Markdown Conversion Regexes ──────────────────────────────────────────────
_RE_PAGE       = re.compile(r'^---\s*PAGE\s+(\d+)\s*---$', re.IGNORECASE)
_RE_SOURCE_TAG = re.compile(r'<source_id>(.+?)</source_id>')
_RE_DEEP       = re.compile(r'^(\d+\.\d+\.\d+)\s+([A-Z].+)$')
_RE_SUB        = re.compile(r'^(\d+\.\d+)\s+([A-Z][^\n]{2,})$')
_RE_TOP        = re.compile(r'^(\d+)\.\s+([A-Z][^\n]{3,})$')
_RE_LETTERED   = re.compile(r'^\(([a-z])\)\s+([A-Z].+)$')
_RE_ARTICLE    = re.compile(r'^(Article\s+[IVXLCDM\d]+)[:\.\s]*(.*)', re.IGNORECASE)
_RE_SECTION_KW = re.compile(r'^(Section\s+\d+(?:\.\d+)*)[:\.\s]*(.*)', re.IGNORECASE)
_RE_APPENDIX   = re.compile(r'^(Appendix\s+[A-Z\d])[:\.\s]*(.*)', re.IGNORECASE)
_RE_SYMBOL     = re.compile(r'^(§\s*\d+(?:\.\d+)*)\s+(.*)')
_RE_ALLCAPS    = re.compile(r'^([A-Z][A-Z\s\-\&\/]{4,})(?::|\.|\s*$)')
_RE_TITLE_CASE = re.compile(r'^((?:[A-Z][a-z]+\s){2,6}(?:[A-Z][a-z]+))(?::|$)')

def convert_to_markdown(raw_text: str) -> str:
    output: list[str] = []
    for line in raw_text.split('\n'):
        stripped = line.strip()

        m = _RE_PAGE.match(stripped)
        if m:
            output.append(f'<!-- page:{m.group(1)} -->')
            continue

        if _RE_SOURCE_TAG.search(stripped):
            output.append(stripped)
            continue

        m = _RE_DEEP.match(stripped)
        if m:
            output.append(f'#### {m.group(1)} {m.group(2)}')
            continue

        m = _RE_SUB.match(stripped)
        if m:
            output.append(f'### {m.group(1)} {m.group(2)}')
            continue

        m = _RE_TOP.match(stripped)
        if m:
            output.append(f'## {m.group(1)}. {m.group(2)}')
            continue

        m = _RE_LETTERED.match(stripped)
        if m:
            output.append(f'- **({m.group(1)})** {m.group(2)}')
            continue

        m = _RE_ARTICLE.match(stripped)
        if m:
            output.append(f'# {m.group(1)}: {m.group(2)}' if m.group(2) else f'# {m.group(1)}')
            continue

        m = _RE_SECTION_KW.match(stripped)
        if m:
            output.append(f'## {m.group(1)}: {m.group(2)}' if m.group(2) else f'## {m.group(1)}')
            continue

        m = _RE_APPENDIX.match(stripped)
        if m:
            output.append(f'# {m.group(1)}: {m.group(2)}' if m.group(2) else f'# {m.group(1)}')
            continue

        m = _RE_SYMBOL.match(stripped)
        if m:
            output.append(f'## {m.group(1)} {m.group(2)}')
            continue

        m = _RE_ALLCAPS.match(stripped)
        if m and len(stripped) < 100:
            output.append(f'## {stripped}')
            continue

        m = _RE_TITLE_CASE.match(stripped)
        if m and len(stripped) < 120:
            output.append(f'### {stripped}')
            continue

        output.append(line)
    return '\n'.join(output)
