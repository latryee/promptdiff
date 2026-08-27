"""Text & Word-Level Diffing Engine."""

from __future__ import annotations

import difflib
import re
from typing import List, Tuple
from promptdiff.core.models import DiffChunk


def compute_word_diff(text1: str, text2: str) -> List[DiffChunk]:
    """Compute word/token level diff chunks between two texts."""
    # Tokenize preserving whitespaces & punctuation
    words1 = re.findall(r"\S+|\s+", text1)
    words2 = re.findall(r"\S+|\s+", text2)

    matcher = difflib.SequenceMatcher(None, words1, words2)
    chunks: List[DiffChunk] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        sub1 = "".join(words1[i1:i2])
        sub2 = "".join(words2[j1:j2])

        if tag == "equal":
            chunks.append(DiffChunk(kind="equal", v1_text=sub1, v2_text=sub2))
        elif tag == "insert":
            chunks.append(DiffChunk(kind="insert", v1_text="", v2_text=sub2))
        elif tag == "delete":
            chunks.append(DiffChunk(kind="delete", v1_text=sub1, v2_text=""))
        elif tag == "replace":
            chunks.append(DiffChunk(kind="replace", v1_text=sub1, v2_text=sub2))

    return chunks


def compute_line_diff(text1: str, text2: str) -> List[Tuple[str, str, str]]:
    """Compute line-by-line diff with tags (' ', '+', '-', '?')."""
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.ndiff(lines1, lines2))
    result = []
    for line in diff:
        tag = line[0]
        content = line[2:]
        result.append((tag, content, line))
    return result
