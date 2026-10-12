"""Extract every PDF in the repo's tmp/ folder into one table: tmp/pdf_table.csv"""
import csv
from pathlib import Path
from pypdf import PdfReader

tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
table_path = tmp_dir / "pdf_table.csv"

rows = []
for pdf_path in sorted(tmp_dir.rglob("*.pdf")):
    reader = PdfReader(pdf_path)
    content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    content = content.encode("utf-8", "ignore").decode("utf-8")  # drop broken characters
    rows.append([pdf_path.name, pdf_path.as_uri(), content])
    print("Extracted", pdf_path)

with open(table_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "file_link", "content"])
    writer.writerows(rows)

print(f"Done. {len(rows)} PDFs saved in {table_path}")
