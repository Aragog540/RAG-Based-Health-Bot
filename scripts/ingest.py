"""
scripts/ingest.py
─────────────────
Ingest a medical PDF into ChromaDB for RAG.

Usage:
    python scripts/ingest.py --pdf data/medical_book.pdf

What it does:
    1. Loads the PDF with pypdf
    2. Splits it into overlapping chunks
    3. Embeds each chunk using Ollama (free, local)
    4. Stores chunks in ChromaDB (persisted to disk)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import track

from app.config import settings
from app.retriever import add_documents, collection_count

console = Console()


def load_pdf(pdf_path: str) -> list[Document]:
    """Load a PDF and return one Document per page."""
    reader = PdfReader(pdf_path)
    docs = []
    total = len(reader.pages)
    console.print(f"[cyan]📄 Found {total} pages in '{Path(pdf_path).name}'[/cyan]")

    for page_num, page in enumerate(
        track(reader.pages, description="Reading pages..."), start=1
    ):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue  # skip blank / image-only pages

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(pdf_path).name,
                    "page": page_num,
                    "total_pages": total,
                },
            )
        )

    console.print(f"[green]✓ Extracted text from {len(docs)}/{total} pages[/green]")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split page-level documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in track(docs, description="Splitting into chunks..."):
        splits = splitter.split_documents([doc])
        # Keep page metadata on each chunk
        for split in splits:
            split.metadata.update(doc.metadata)
        chunks.extend(splits)

    console.print(f"[green]✓ Created {len(chunks)} chunks[/green]")
    return chunks


def ingest_pdf_file(pdf_path: str) -> int:
    """Full pipeline: load → chunk → embed → store. Returns chunks added."""
    console.rule("[bold blue]Medical Book Ingestion[/bold blue]")

    # 1. Load
    docs = load_pdf(pdf_path)
    if not docs:
        console.print("[red]❌ No text extracted. Is this a scanned PDF?[/red]")
        raise ValueError("No extractable text found in PDF.")

    # 2. Chunk
    chunks = chunk_documents(docs)

    # 3. Embed + store
    console.print(
        f"[yellow]⏳ Embedding {len(chunks)} chunks with '{settings.embedding_model}'...[/yellow]"
    )
    console.print("[dim]  (This may take a few minutes on first run)[/dim]")

    # Embed in batches of 50 to avoid memory issues
    batch_size = 50
    total_added = 0
    for i in track(
        range(0, len(chunks), batch_size), description="Embedding batches..."
    ):
        batch = chunks[i : i + batch_size]
        add_documents(batch)
        total_added += len(batch)

    console.rule()
    console.print(
        f"[bold green]✅ Done! {total_added} chunks stored in ChromaDB.[/bold green]"
    )
    console.print(
        f"[dim]Total chunks in collection: {collection_count()}[/dim]"
    )
    return total_added


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a medical PDF into ChromaDB for the RAG chatbot."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file (e.g. data/medical_book.pdf)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        console.print(f"[red]❌ File not found: {pdf_path}[/red]")
        sys.exit(1)

    ingest_pdf_file(str(pdf_path))


if __name__ == "__main__":
    main()
