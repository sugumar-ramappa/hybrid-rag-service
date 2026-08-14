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
    chunk_index: int


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
        chunk_index = 0

        for doc in documents:
            texts = self._splitter.split_text(doc.content)

            for text in texts:
                all_chunks.append(TextChunk(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": chunk_index},
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        logger.info(
            "Split %d documents into %d chunks",
            len(documents), len(all_chunks)
        )
        return all_chunks
