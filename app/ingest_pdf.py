from pathlib import Path
from typing import List
import fitz  # PyMuPDF

from app.config import PDF_PATH, PROCESSED_TEXT_PATH


def list_pdf_paths(pdf_path: str) -> List[Path]:
    p = Path(pdf_path).expanduser()
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.pdf"))
    return []


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"\n--- PAGE {page_num} ---\n{text}")

    return "\n".join(pages)


def save_text(text: str, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")


def main() -> None:
    pdfs = list_pdf_paths(PDF_PATH)
    if not pdfs:
        raise SystemExit(f"No PDF files found under: {PDF_PATH!r} ")

    parts: List[str] = []
    for pdf_file in pdfs:
        print(f"Extracting: {pdf_file}")
        body = extract_pdf_text(pdf_file)
        if body.strip():
            parts.append(f"\n\n=== FILE: {pdf_file.name} ===\n{body}")

    combined = "\n".join(parts).strip()
    if not combined:
        raise SystemExit("All PDFs were empty after extraction.")

    save_text(combined, PROCESSED_TEXT_PATH)
    print(f"Saved merged text from {len(pdfs)} PDF(s) to: {PROCESSED_TEXT_PATH}")
    print(f"Total characters: {len(combined)}")


if __name__ == "__main__":
    main()
