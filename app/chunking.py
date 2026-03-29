from functools import lru_cache
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@lru_cache(maxsize=8)
def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    return _splitter(chunk_size, chunk_overlap).split_text(text)
