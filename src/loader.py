"""Load lecture files (PDF/TXT) and attach source metadata."""
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document

from src.config import DATA_DIR


def load_pdf(path: Path, subject: str) -> List[Document]:
    """Read a PDF page by page. Returns [] if the file is unreadable or empty."""
    docs: List[Document] = []
    try:
        reader = PdfReader(str(path))
        pages = reader.pages
    except Exception as e:
        print(f"[WARN] Cannot open {path.name}: {e}")
        return docs

    for page_num, page in enumerate(pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:
            print(f"[WARN] {path.name} page {page_num} unreadable: {e}")
            continue
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": path.name, "subject": subject, "page": page_num},
        ))

    if not docs:
        print(f"[WARN] No extractable text in {path.name} (maybe scanned images?)")
    return docs


def load_txt(path: Path, subject: str) -> List[Document]:
    """Read a plain text file as a single document."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[WARN] Cannot read {path.name}: {e}")
        return []
    if not text:
        print(f"[WARN] {path.name} is empty")
        return []
    return [Document(
        page_content=text,
        metadata={"source": path.name, "subject": subject, "page": 1},
    )]


def load_all_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Walk data_dir/<subject>/ and load every PDF and TXT file."""
    all_docs: List[Document] = []
    if not data_dir.exists():
        print(f"[ERROR] Data folder not found: {data_dir}")
        return all_docs

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        subject = path.parent.name
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            all_docs.extend(load_pdf(path, subject))
        elif suffix == ".txt":
            all_docs.extend(load_txt(path, subject))
    return all_docs


if __name__ == "__main__":
    docs = load_all_documents()
    files = {d.metadata["source"] for d in docs}
    print(f"\nLoaded {len(docs)} pages from {len(files)} files.")
    if docs:
        sample = docs[0]
        print("\n--- Sample ---")
        print("Metadata:", sample.metadata)
        print(sample.page_content[:300])