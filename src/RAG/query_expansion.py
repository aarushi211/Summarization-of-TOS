"""Lightweight query expansion for legal-document retrieval (dense + BM25)."""
import re

# (pattern, extra terms) — applied only when the pattern matches the user query
_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bprompts?\b", re.I), "input output submission content"),
    (re.compile(r"\bhousehold\b", re.I), "account sharing family residence membership"),
    (re.compile(r"\bshare\b.*\baccount\b|\baccount\b.*\bshare\b", re.I), "household sharing membership"),
    (re.compile(r"\bterminat(e|ion|ed|ing)\b", re.I), "suspend cancellation cancel end"),
    (re.compile(r"\brefunds?\b", re.I), "cancellation billing payment charge"),
    (re.compile(r"\bcancell?(ation|ed|ing)\b", re.I), "refund terminate subscription billing"),
    (re.compile(r"\bplaylist(s)?\b", re.I), "library saved content music collection"),
    (re.compile(r"\bdispute\s+resolution\b", re.I), "governing law jurisdiction limitation legal action court claims"),
    (re.compile(r"\bdispute\b", re.I), "arbitration complaint resolution governing law jurisdiction"),
    (re.compile(r"\bcomplaint\b", re.I), "dispute resolution arbitration governing law"),
    (re.compile(r"\barbitration\b", re.I), "dispute resolution binding waiver"),
    (re.compile(r"\bwithout\s+notice\b", re.I), "notice termination suspension"),
    (re.compile(r"\bpersonal\s+data\b", re.I), "information privacy processing collection"),
    (re.compile(r"\bhandle(s|d)?\b.*\b(data|information)\b", re.I), "content input output processing storage retention privacy"),
    (re.compile(r"\bhandle(s|d)?\b.*\bprompts?\b", re.I), "content input output services provide receive"),
    (re.compile(r"\bsubmitted\b", re.I), "provide input content output"),
    (re.compile(r"\bapi\b", re.I), "services content input application programming interface"),
]


def expand_search_query(query: str) -> str:
    """
    Append domain synonyms so retrieval matches legal wording (e.g. prompt → input).
    The original query is always kept; expansions are additive for embedding/BM25 only.
    """
    if not query or not query.strip():
        return query
    extras: list[str] = []
    seen: set[str] = set()
    for pattern, terms in _EXPANSIONS:
        if pattern.search(query):
            for term in terms.split():
                key = term.lower()
                if key not in seen:
                    seen.add(key)
                    extras.append(term)
    if not extras:
        return query
    return f"{query.strip()} {' '.join(extras)}"
