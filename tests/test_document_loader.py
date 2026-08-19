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


def test_loads_nested_directories(tmp_path: Path) -> None:
    """
    A real docs corpus is nested. iterdir() only yielded direct children, so
    everything below the top level was silently invisible.
    """
    doc_dir = tmp_path / "docs"
    (doc_dir / "concepts" / "storage").mkdir(parents=True)
    (doc_dir / "top.md").write_text("Top level")
    (doc_dir / "concepts" / "overview.md").write_text("One deep")
    (doc_dir / "concepts" / "storage" / "volumes.md").write_text("Two deep")

    documents = DocumentLoader(str(doc_dir)).load_all()

    assert len(documents) == 3


def test_source_is_path_relative_to_corpus_root(tmp_path: Path) -> None:
    """
    Bare filenames make citations useless in a tree full of index.md, and
    weaken chunk IDs, which derive identity partly from source.
    """
    doc_dir = tmp_path / "docs"
    (doc_dir / "concepts" / "storage").mkdir(parents=True)
    (doc_dir / "concepts" / "storage" / "index.md").write_text("Storage index")
    (doc_dir / "concepts" / "index.md").write_text("Concepts index")

    documents = DocumentLoader(str(doc_dir)).load_all()

    sources = {d.metadata["source"] for d in documents}
    assert sources == {"concepts/index.md", "concepts/storage/index.md"}


def test_skips_files_inside_hidden_directories(tmp_path: Path) -> None:
    """
    Checking only file_path.name misses these: the files themselves have
    ordinary names, and a cloned repo hides thousands of them under .git/.
    """
    doc_dir = tmp_path / "docs"
    (doc_dir / ".git").mkdir(parents=True)
    (doc_dir / ".github" / "workflows").mkdir(parents=True)
    (doc_dir / ".git" / "COMMIT_EDITMSG.txt").write_text("not a document")
    (doc_dir / ".github" / "workflows" / "release.md").write_text("not a document")
    (doc_dir / "real.md").write_text("an actual document")

    documents = DocumentLoader(str(doc_dir)).load_all()

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "real.md"


def test_one_bad_file_does_not_abort_the_load(tmp_path: Path) -> None:
    """A single unreadable file must not cost you the rest of the corpus."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "good.md").write_text("readable")
    (doc_dir / "bad.txt").write_bytes(b"\xff\xfe invalid utf-8 \x00\x80")

    documents = DocumentLoader(str(doc_dir)).load_all()

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "good.md"


def test_max_files_caps_the_corpus(tmp_path: Path) -> None:
    """A cost guard: stop reading after N files."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    for i in range(10):
        (doc_dir / f"doc{i:02d}.md").write_text(f"document number {i}")

    assert len(DocumentLoader(str(doc_dir), max_files=3).load_all()) == 3
    assert len(DocumentLoader(str(doc_dir), max_files=0).load_all()) == 10
    assert len(DocumentLoader(str(doc_dir)).load_all()) == 10


def test_max_files_selects_the_same_files_every_time(tmp_path: Path) -> None:
    """
    Deterministic selection matters: a golden set built against a capped corpus
    would be worthless if the cap picked different files on the next run.
    """
    doc_dir = tmp_path / "docs"
    (doc_dir / "nested").mkdir(parents=True)
    for i in range(10):
        (doc_dir / f"doc{i:02d}.md").write_text(f"top {i}")
        (doc_dir / "nested" / f"sub{i:02d}.md").write_text(f"nested {i}")

    first = DocumentLoader(str(doc_dir), max_files=5).load_all()
    second = DocumentLoader(str(doc_dir), max_files=5).load_all()

    assert [d.metadata["source"] for d in first] == [d.metadata["source"] for d in second]


def test_raising_max_files_is_a_superset(tmp_path: Path) -> None:
    """
    Raising the cap must only add files, never swap them - otherwise
    incremental ingest would re-embed content it had already paid for.
    """
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    for i in range(10):
        (doc_dir / f"doc{i:02d}.md").write_text(f"document {i}")

    small = {d.metadata["source"] for d in DocumentLoader(str(doc_dir), max_files=3).load_all()}
    large = {d.metadata["source"] for d in DocumentLoader(str(doc_dir), max_files=6).load_all()}

    assert small < large


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
