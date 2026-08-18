"""
Text Splitter - splits documents into smaller chunks for embedding.

WHY SPLIT?
LLMs have token limits. We split documents into overlapping chunks so:
1. Each chunk fits in the embedding model's context window
2. Overlap ensures we don't lose context at chunk boundaries
3. Smaller chunks = more precise retrieval

PYTHON CONCEPTS FOR JAVA DEVS:
- List comprehension: [x for x in items] = like Java's stream().map().collect()
- enumerate() = like IntStream with index
"""

import hashlib
import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with its metadata - like a Java record."""
    content: str
    metadata: dict
    chunk_index: int  # position WITHIN its source document, not across the corpus
    chunk_id: str  # stable identity, derived from source + position + content


def _document_key(metadata: dict) -> str:
    """
    Identify the document a chunk came from.

    PDFs produce one Document per page, so the page number is part of the
    identity - otherwise page 1 and page 2 would both claim chunk_index 0.
    """
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    return f"{source}#p{page}" if page is not None else source


def _make_chunk_id(doc_key: str, chunk_index: int, content: str) -> str:
    """
    Build a stable ID for a chunk.

    Derived from content, so the same chunk always gets the same ID no matter
    what else is in the corpus or what order documents were loaded in. This is
    what makes re-ingestion idempotent rather than destructive.
    """
    digest = hashlib.sha256(f"{doc_key}:{chunk_index}:{content}".encode("utf-8"))
    return digest.hexdigest()[:16]


class TextSplitter:
    """
    Splits documents into overlapping chunks - like a Java @Service.

    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Paragraphs (\\n\\n)
    2. Lines (\\n)
    3. Sentences (. ! ?)
    4. Words (space)
    5. Characters (last resort)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_documents(self, documents: list[Document]) -> list[TextChunk]:
        """
        Split a list of documents into chunks.

        This is like Java's:
            documents.stream()
                .flatMap(doc -> splitOne(doc).stream())
                .collect(Collectors.toList())
        """
        all_chunks: list[TextChunk] = []

        for doc in documents:
            texts = self._splitter.split_text(doc.content)
            doc_key = _document_key(doc.metadata)

            # Index restarts per document. A global counter would shift every
            # downstream chunk's identity whenever a document was added or removed.
            for chunk_index, text in enumerate(texts):
                all_chunks.append(TextChunk(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": chunk_index},
                    chunk_index=chunk_index,
                    chunk_id=_make_chunk_id(doc_key, chunk_index, text),
                ))

        logger.info(
            "Split %d documents into %d chunks",
            len(documents), len(all_chunks)
        )
        return all_chunks
