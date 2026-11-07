from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PDF = Path(r"${USER_HOME}\Desktop\s41467-026-71411-1_reference.pdf")
OUT = ROOT / "reports" / "reference_pdf_pages"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import fitz
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        print({"status": "missing_dependency", "error": repr(exc)})
        sys.exit(1)

    doc = fitz.open(PDF)
    figure_pages: list[int] = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if "Fig." in text or "Figure" in text or "Extended Data" in text:
            figure_pages.append(i)

    indices = sorted(set(list(range(min(8, doc.page_count))) + figure_pages[:12]))
    thumbs = []
    for i in indices:
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        page_path = OUT / f"page_{i + 1:03d}.png"
        pix.save(page_path)
        im = Image.open(page_path).convert("RGB")
        im.thumbnail((360, 480))
        canvas = Image.new("RGB", (380, 520), "white")
        canvas.paste(im, ((380 - im.width) // 2, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"Page {i + 1}", fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 380, rows * 520), "white")
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 380, (idx // cols) * 520))
    sheet_path = OUT / "reference_contact_sheet.png"
    sheet.save(sheet_path)
    print({"pages": doc.page_count, "figure_pages": [i + 1 for i in figure_pages], "contact_sheet": str(sheet_path)})


if __name__ == "__main__":
    main()
