import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

_FILE_BOUNDARY = re.compile(r"(?=\n\n=== FILE: .+ ===)")
_PAGE_BOUNDARY = re.compile(r"\n(?=--- PAGE \d+ ---)")
def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

def _coarse_segments(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if "\n\n=== FILE:" in text:
        file_parts = [p.strip() for p in _FILE_BOUNDARY.split(text) if p.strip()]
    else:
        file_parts = [text]

    segments: List[str] = []

    for part in file_parts:
        if re.search(r"--- PAGE \d+ ---", part):
            pages = [p.strip() for p in _PAGE_BOUNDARY.split(part) if p.strip()]
        else:
            pages = [part]
            
        for page in pages:
            segments.append(page)

    return segments


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks: List[str] = []

    for segment in _coarse_segments(text):
        if len(segment) <= chunk_size:
            chunks.append(segment)
        else:
            chunks.extend(splitter.split_text(segment))

    return chunks