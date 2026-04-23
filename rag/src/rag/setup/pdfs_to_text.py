from pathlib import Path

from pypdf import PdfReader

from rag.backend.constants import DATA_PATH


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(path)

    all_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    return all_text


def export_text_to_txt(text: str, export_path: Path) -> None:
    with export_path.open("w") as file:
        file.write(text)


if __name__ == "__main__":
    example_text = extract_text_from_pdf(DATA_PATH / "fisk.pdf")
    for pdf_path in DATA_PATH.glob("*.pdf"):
        pdf_text = extract_text_from_pdf(pdf_path)

        filename = f"{pdf_path.stem.casefold()}.txt"

        export_text_to_txt(pdf_text, DATA_PATH / filename)
