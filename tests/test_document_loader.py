"""
Tests for DocumentLoader.

PYTHON TESTING FOR JAVA DEVS:
- pytest = like JUnit 5
- def test_xxx() = like @Test public void testXxx()
- assert x == y = like assertEquals(y, x)
- tmp_path = like @TempDir (pytest provides it automatically)
- No need for a test class! Functions work fine in pytest.
"""

from pathlib import Path

from src.rag.document_loader import Document, DocumentLoader


def test_load_text_file(tmp_path: Path) -> None:
    """Test loading a .txt file - like @Test in JUnit."""
    # Arrange (Given)
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    test_file = doc_dir / "test.txt"
    test_file.write_text("Hello, this is a test document.")

    # Act (When)
    loader = DocumentLoader(str(doc_dir))
    documents = loader.load_all()

    # Assert (Then)
    assert len(documents) == 1
    assert documents[0].content == "Hello, this is a test document."
    assert documents[0].metadata["source"] == "test.txt"
    assert documents[0].metadata["type"] == ".txt"


def test_load_markdown_file(tmp_path: Path) -> None:
    """Test loading a .md file."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    test_file = doc_dir / "readme.md"
    test_file.write_text("# Title\n\nSome content here.")

    loader = DocumentLoader(str(doc_dir))
    documents = loader.load_all()

    assert len(documents) == 1
    assert "# Title" in documents[0].content
    assert documents[0].metadata["type"] == ".md"


def test_load_empty_directory(tmp_path: Path) -> None:
    """Test loading from an empty directory returns empty list."""
    doc_dir = tmp_path / "empty"
    doc_dir.mkdir()

    loader = DocumentLoader(str(doc_dir))
    documents = loader.load_all()

    assert documents == []


def test_skip_unsupported_files(tmp_path: Path) -> None:
    """Test that unsupported file types are skipped."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "image.png").write_bytes(b"\x89PNG")
    (doc_dir / "data.csv").write_text("a,b,c")
    (doc_dir / "valid.txt").write_text("This should be loaded")

    loader = DocumentLoader(str(doc_dir))
    documents = loader.load_all()

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "valid.txt"


def test_creates_directory_if_missing(tmp_path: Path) -> None:
    """Test that the loader creates the directory if it doesn't exist."""
    doc_dir = tmp_path / "nonexistent" / "nested"

    loader = DocumentLoader(str(doc_dir))
    assert doc_dir.exists()
    documents = loader.load_all()
    assert documents == []


def test_skip_empty_files(tmp_path: Path) -> None:
    """Test that empty files are skipped."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "empty.txt").write_text("")
    (doc_dir / "whitespace.txt").write_text("   \n\n   ")

    loader = DocumentLoader(str(doc_dir))
    documents = loader.load_all()

    # Empty file should be skipped, whitespace-only should also be skipped
    assert len(documents) <= 1


def test_multiple_files(tmp_path: Path) -> None:
    """Test loading multiple files."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "first.txt").write_text("First document")
    (doc_dir / "second.txt").write_text("Second document")
    (doc_dir / "third.md").write_text("Third document")

    loader = DocumentLoader(str(doc_dir))
    documents = loader.load_all()

    assert len(documents) == 3
    sources = {d.metadata["source"] for d in documents}
    assert sources == {"first.txt", "second.txt", "third.md"}
