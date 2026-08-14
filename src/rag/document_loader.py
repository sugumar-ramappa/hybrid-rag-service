"""
Document Loader - loads PDFs and text files from a directory.

PYTHON CONCEPTS FOR JAVA DEVS:
- List[Document] = like Java's List<Document>
- dataclass = like a Java POJO/record
- yield = lazy evaluation (like Java's Stream)
- Path.glob() = like Files.walk() with filter
- with open() = like Java's try-with-resources
- logging = like SLF4J/Logback
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)  # Like LoggerFactory.getLogger(ClassName.class)


@dataclass
class Document:
    """
    Represents a loaded document - like a Java POJO/DTO.

    Attributes:
        content: The text content of the document
        metadata: Key-value pairs about the document (source file, page number, etc.)
    """
    content: str
    metadata: dict = field(default_factory=dict)  # default_factory avoids mutable default gotcha


class DocumentLoader:
    """
    Loads documents from a directory - like a Spring @Service.

    Supports: .txt, .md, .pdf files
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(self, documents_dir: str) -> None:
        """
        Constructor - same concept as Java constructor.
        'self' is like Java's 'this' but must be explicit.
        """
        self._documents_dir = Path(documents_dir)
        if not self._documents_dir.exists():
            self._documents_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created documents directory: %s", self._documents_dir)

    def load_all(self) -> list[Document]:
        """
        Load all supported documents from the directory.
        Returns: list of Document objects (like Java's List<Document>)
        """
        documents: list[Document] = []

        for file_path in sorted(self._documents_dir.iterdir()):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self._load_file(file_path)
                    documents.extend(docs)
                    logger.info("Loaded %d document(s) from: %s", len(docs), file_path.name)
                except Exception:
                    logger.exception("Failed to load file: %s", file_path.name)

        logger.info("Total documents loaded: %d", len(documents))
        return documents

    def _load_file(self, file_path: Path) -> list[Document]:
        """
        Load a single file. Private method (underscore prefix = convention, like Java's private).
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix in {".txt", ".md"}:
            return self._load_text(file_path)
        else:
            return []

    def _load_text(self, file_path: Path) -> list[Document]:
        """Load a text or markdown file."""
        # 'with' statement = try-with-resources in Java
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return []

        return [Document(
            content=content,
            metadata={"source": file_path.name, "type": file_path.suffix}
        )]

    def _load_pdf(self, file_path: Path) -> list[Document]:
        """Load a PDF file, one Document per page."""
        documents: list[Document] = []
        reader = PdfReader(str(file_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    content=text,
                    metadata={
                        "source": file_path.name,
                        "type": ".pdf",
                        "page": page_num,
                    }
                ))

        return documents
