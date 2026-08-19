
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

    def __init__(self, documents_dir: str, max_files: int = 0) -> None:
        """
        Constructor - same concept as Java constructor.
        'self' is like Java's 'this' but must be explicit.

        Args:
            documents_dir: root of the corpus
            max_files: stop after this many files; 0 means no limit
        """
        self._documents_dir = Path(documents_dir)
        self._max_files = max_files
        if not self._documents_dir.exists():
            self._documents_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created documents directory: %s", self._documents_dir)

    def load_all(self) -> list[Document]:
        """
        Load every supported document beneath the documents directory.

        Walks the tree recursively: a real documentation corpus is nested
        (docs/concepts/storage/persistent-volumes.md), and reading only the top
        level would silently load almost nothing.

        Returns: list of Document objects (like Java's List<Document>)
        """
        documents: list[Document] = []
        root = self._documents_dir.resolve()
        failures = 0
        files_read = 0
        capped = False

        # rglob walks the whole tree; iterdir would only yield direct children.
        for file_path in sorted(self._documents_dir.rglob("*")):
            if self._max_files and files_read >= self._max_files:
                capped = True
                break

            if not file_path.is_file():
                continue  # rglob yields directories too
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            relative = file_path.relative_to(self._documents_dir)

            # The hidden check has to cover EVERY path component, not just the
            # filename. Once you recurse into a cloned repository, .git/ and
            # .github/ contain thousands of files whose own names look
            # perfectly ordinary.
            if any(part.startswith(".") for part in relative.parts):
                continue

            # Guards against symlinks pointing outside the corpus.
            if not file_path.resolve().is_relative_to(root):
                logger.warning("Skipping file outside documents dir: %s", relative)
                continue

            files_read += 1

            try:
                docs = self._load_file(file_path, relative)
                documents.extend(docs)
                logger.debug("Loaded %d document(s) from: %s", len(docs), relative)
            except Exception:
                failures += 1
                logger.exception("Failed to load file: %s", relative)

        # Say so loudly when a cap truncated the corpus. A silent limit reads
        # exactly like "that is all there was", which would quietly invalidate
        # any recall number measured against it.
        if capped:
            logger.warning(
                "MAX_FILES=%d reached - corpus truncated. Metrics measured now "
                "describe this subset, not the full corpus.",
                self._max_files,
            )

        # Surface failures in the summary. Without a count, a systematic problem
        # is indistinguishable from a small corpus - the per-file stack traces
        # scroll past and the final total simply looks lower than expected.
        if failures:
            logger.warning(
                "Loaded %d document(s); %d file(s) failed", len(documents), failures
            )
        else:
            logger.info("Total documents loaded: %d", len(documents))

        return documents

    def _load_file(self, file_path: Path, relative: Path) -> list[Document]:
        """
        Load a single file. Private method (underscore prefix = convention, like Java's private).

        Args:
            file_path: absolute path, used to read the file
            relative: path relative to the documents directory, used as `source`
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path, relative)
        elif suffix in {".txt", ".md"}:
            return self._load_text(file_path, relative)
        else:
            return []

    @staticmethod
    def _source_of(relative: Path) -> str:
        """
        The `source` recorded on every chunk.

        Uses the path relative to the corpus root, not the bare filename: a docs
        tree contains dozens of files called index.md, so a filename alone makes
        citations useless and weakens chunk IDs, which derive identity partly
        from source. as_posix() keeps separators consistent across platforms.
        """
        return relative.as_posix()

    def _load_text(self, file_path: Path, relative: Path) -> list[Document]:
        """Load a text or markdown file."""
        # 'with' statement = try-with-resources in Java
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return []

        return [Document(
            content=content,
            metadata={"source": self._source_of(relative), "type": file_path.suffix}
        )]

    def _load_pdf(self, file_path: Path, relative: Path) -> list[Document]:
        """Load a PDF file, one Document per page."""
        documents: list[Document] = []
        reader = PdfReader(str(file_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    content=text,
                    metadata={
                        "source": self._source_of(relative),
                        "type": ".pdf",
                        "page": page_num,
                    }
                ))

        return documents
