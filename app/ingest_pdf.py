from pathlib import Path
import fitz  # PyMuPDF

from app.config import PDF_PATH, PROCESSED_TEXT_PATH


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
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
    print(f"Extracting text from: {PDF_PATH}")
    text = extract_pdf_text(PDF_PATH)
    save_text(text, PROCESSED_TEXT_PATH)
    print(f"Saved extracted text to: {PROCESSED_TEXT_PATH}")
    print(f"Total characters: {len(text)}")


if __name__ == "__main__":
    main()