"""
parsers.py — File parsers that return text plus location metadata.

Every parser returns a :class:`ParsedDocument` carrying:
  * ``text``         — extracted plain text, lines joined with ``\n``
  * ``line_labels``  — parallel list giving each text-line a label
                       (page number / sheet name / paragraph index).
  * ``label_kind``   — one of ``"line"`` | ``"page"`` | ``"row"`` |
                       ``"paragraph"`` | ``"chapter"``.

That parallel array lets the chunker translate "lines 50-80 of the parsed
text" back into "pages 3-4" or "Sheet1 rows 12-30" for citation.

Supported formats:
  * Markdown / Plain text / source code  — line-numbered
  * PDF                                  — page-numbered (with OCR fallback)
  * DOCX                                 — paragraph-numbered
  * XLSX                                 — sheet+row labelled
  * CSV / TSV                            — row-numbered
  * HTML                                 — line-numbered (rendered)
  * EPUB                                 — chapter-numbered
  * RTF                                  — line-numbered
  * Images (PNG/JPG/JPEG/TIFF/BMP/WEBP)  — line-numbered (OCR)
  * JSON                                 — line-numbered

Heavy optional deps (pdfplumber, python-docx, openpyxl, beautifulsoup4,
ebooklib, striprtf, pytesseract, pdf2image, Pillow) are imported lazily.
A parser that finds its dep missing raises ImportError with a clean install
hint; the indexer logs and skips that file.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParsedDocument — what every parser returns
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """Structured output of a file parser.

    Attributes:
        text:        Extracted text, with newline separators.
        line_labels: One label per text line (1-indexed alignment with
                     ``text.splitlines()``). Empty list = labels are the
                     line numbers themselves (i.e. label_kind="line").
        label_kind:  Logical unit the labels refer to. Drives how the
                     indexer formats citations ("page 3-4", "Sheet1!12-30").
    """

    text: str
    line_labels: List[str] = field(default_factory=list)
    label_kind: str = "line"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FileParser(ABC):
    """Abstract base class for all file parsers."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Extract the contents of ``path`` as a :class:`ParsedDocument`."""

    def parse_text(self, path: Path) -> str:
        """Backwards-compatible helper: return only the extracted text."""
        return self.parse(path).text


# ---------------------------------------------------------------------------
# Plain-text style parsers (line-numbered)
# ---------------------------------------------------------------------------


class TextParser(FileParser):
    """Plain-text parser with graceful encoding fallback."""

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(text=path.read_text(encoding="utf-8", errors="replace"))


class MarkdownParser(FileParser):
    """Markdown — raw text; YAML front-matter is stripped upstream by the indexer."""

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(text=path.read_text(encoding="utf-8", errors="replace"))


class CodeParser(FileParser):
    """Source-code parser (raw text — language is inferred from extension)."""

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(text=path.read_text(encoding="utf-8", errors="replace"))


class JSONParser(FileParser):
    """JSON parser — pretty-prints for stable line numbers."""

    def parse(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            logger.warning("JSON decode error in %s — falling back to raw text", path)
            text = raw
        return ParsedDocument(text=text)


# ---------------------------------------------------------------------------
# PDF parser (page-numbered) — with optional OCR fallback for scans
# ---------------------------------------------------------------------------


class PDFParser(FileParser):
    """PDF parser via pdfplumber, with Tesseract OCR fallback for scanned pages."""

    def __init__(self, enable_ocr: bool = True, ocr_min_chars: int = 50, ocr_dpi: int = 200) -> None:
        self.enable_ocr = enable_ocr
        self.ocr_min_chars = ocr_min_chars
        self.ocr_dpi = ocr_dpi

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PDF parsing — install with: pip install pdfplumber"
            ) from exc

        text_lines: List[str] = []
        line_labels: List[str] = []
        ocr_pages = 0
        scanned_pdf = False

        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()

                # Empty/short page → likely scanned. Try OCR.
                if self.enable_ocr and len(page_text) < self.ocr_min_chars:
                    ocr_text = _ocr_pdf_page(path, page_idx, self.ocr_dpi)
                    if len(ocr_text.strip()) > len(page_text):
                        page_text = ocr_text.strip()
                        ocr_pages += 1
                        scanned_pdf = True

                if not page_text:
                    continue

                for line in page_text.splitlines():
                    text_lines.append(line)
                    line_labels.append(str(page_idx))
                # Blank line between pages so the chunker treats each as its own paragraph.
                text_lines.append("")
                line_labels.append(str(page_idx))

        if scanned_pdf:
            logger.info("OCR'd %d page(s) of %s", ocr_pages, path.name)

        return ParsedDocument(
            text="\n".join(text_lines),
            line_labels=line_labels,
            label_kind="page",
        )


def _ocr_pdf_page(pdf_path: Path, page_num: int, dpi: int) -> str:
    """Render one page of a PDF and OCR it. Returns "" on any failure."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        return ""
    try:
        images = convert_from_path(
            str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0]) or ""
    except Exception as exc:
        logger.debug("OCR failed for %s page %d: %s", pdf_path.name, page_num, exc)
        return ""


# ---------------------------------------------------------------------------
# Image parser (OCR)
# ---------------------------------------------------------------------------


class ImageParser(FileParser):
    """OCR an image file with Tesseract."""

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "OCR requires Pillow + pytesseract — install with: "
                "pip install Pillow pytesseract  (and `brew install tesseract` for the engine)"
            ) from exc
        try:
            img = Image.open(path)
            text = pytesseract.image_to_string(img) or ""
        except Exception as exc:
            logger.warning("OCR failed for %s: %s", path.name, exc)
            text = ""
        return ParsedDocument(text=text)


# ---------------------------------------------------------------------------
# DOCX parser (paragraph-numbered)
# ---------------------------------------------------------------------------


class DocxParser(FileParser):
    """Microsoft Word .docx parser via python-docx."""

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from docx import Document  # python-docx
        except ImportError as exc:
            raise ImportError(
                "DOCX support requires python-docx — install with: pip install python-docx"
            ) from exc

        doc = Document(str(path))
        text_lines: List[str] = []
        line_labels: List[str] = []

        for p_idx, para in enumerate(doc.paragraphs, start=1):
            txt = (para.text or "").strip()
            if not txt:
                continue
            text_lines.append(txt)
            line_labels.append(str(p_idx))
            # Blank line separator
            text_lines.append("")
            line_labels.append(str(p_idx))

        # Tables — flatten as TSV-ish rows so values stay associated with labels.
        for t_idx, table in enumerate(doc.tables, start=1):
            for r_idx, row in enumerate(table.rows, start=1):
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                line = " | ".join(cells)
                if not line.strip(" |"):
                    continue
                text_lines.append(line)
                line_labels.append(f"table{t_idx}.row{r_idx}")
            text_lines.append("")
            line_labels.append(f"table{t_idx}")

        return ParsedDocument(
            text="\n".join(text_lines),
            line_labels=line_labels,
            label_kind="paragraph",
        )


# ---------------------------------------------------------------------------
# XLSX parser (sheet+row labelled)
# ---------------------------------------------------------------------------


class XlsxParser(FileParser):
    """Excel .xlsx parser via openpyxl. Each row → one logical line."""

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ImportError(
                "XLSX support requires openpyxl — install with: pip install openpyxl"
            ) from exc

        wb = load_workbook(filename=str(path), read_only=True, data_only=True)
        text_lines: List[str] = []
        line_labels: List[str] = []

        for sheet in wb.worksheets:
            for r_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                cells = ["" if v is None else str(v) for v in row]
                line = " | ".join(cells).strip(" |")
                if not line:
                    continue
                text_lines.append(line)
                line_labels.append(f"{sheet.title}!{r_idx}")
            # Blank line between sheets so chunker breaks between them.
            text_lines.append("")
            line_labels.append(sheet.title)

        return ParsedDocument(
            text="\n".join(text_lines),
            line_labels=line_labels,
            label_kind="row",
        )


# ---------------------------------------------------------------------------
# CSV / TSV parser (row-numbered)
# ---------------------------------------------------------------------------


class CSVParser(FileParser):
    """CSV/TSV parser. Each row labelled by row number."""

    def __init__(self, delimiter: str = ",") -> None:
        self.delimiter = delimiter

    def parse(self, path: Path) -> ParsedDocument:
        text_lines: List[str] = []
        line_labels: List[str] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                for r_idx, row in enumerate(reader, start=1):
                    line = " | ".join(c.strip() for c in row if c is not None)
                    if not line.strip(" |"):
                        continue
                    text_lines.append(line)
                    line_labels.append(str(r_idx))
        except Exception as exc:
            logger.warning("CSV parse error for %s: %s", path, exc)
            text_lines = [path.read_text(encoding="utf-8", errors="replace")]
            line_labels = []

        return ParsedDocument(
            text="\n".join(text_lines),
            line_labels=line_labels,
            label_kind="row",
        )


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------


class HTMLParser(FileParser):
    """HTML parser — strips scripts/styles, keeps visible text."""

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "HTML support requires beautifulsoup4 — install with: pip install beautifulsoup4 lxml"
            ) from exc

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Collapse runs of blank lines into a single blank line.
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return ParsedDocument(text=cleaned)


# ---------------------------------------------------------------------------
# EPUB parser (chapter-numbered)
# ---------------------------------------------------------------------------


class EPUBParser(FileParser):
    """EPUB parser — one chapter per item, labelled by chapter number."""

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from ebooklib import epub, ITEM_DOCUMENT
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "EPUB support requires ebooklib + beautifulsoup4 — install with: "
                "pip install ebooklib beautifulsoup4"
            ) from exc

        book = epub.read_epub(str(path))
        text_lines: List[str] = []
        line_labels: List[str] = []
        ch_idx = 0
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            ch_idx += 1
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            chapter_text = soup.get_text(separator="\n")
            for line in chapter_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                text_lines.append(line)
                line_labels.append(str(ch_idx))
            text_lines.append("")
            line_labels.append(str(ch_idx))

        return ParsedDocument(
            text="\n".join(text_lines),
            line_labels=line_labels,
            label_kind="chapter",
        )


# ---------------------------------------------------------------------------
# RTF parser
# ---------------------------------------------------------------------------


class RTFParser(FileParser):
    """RTF parser via striprtf."""

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as exc:
            raise ImportError(
                "RTF support requires striprtf — install with: pip install striprtf"
            ) from exc
        raw = path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(text=rtf_to_text(raw) or "")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ParserRegistry:
    """Maps file extensions → parser instances. Falls back to TextParser."""

    def __init__(self, enable_ocr: bool = True, ocr_min_chars: int = 50, ocr_dpi: int = 200) -> None:
        pdf_parser = PDFParser(enable_ocr=enable_ocr, ocr_min_chars=ocr_min_chars, ocr_dpi=ocr_dpi)
        code = CodeParser()
        image = ImageParser() if enable_ocr else TextParser()

        self._parsers: Dict[str, FileParser] = {
            # Text
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".txt": TextParser(),
            ".log": TextParser(),
            # Code
            ".py": code,
            ".js": code,
            ".ts": code,
            ".jsx": code,
            ".tsx": code,
            ".mjs": code,
            ".cjs": code,
            ".java": code,
            ".go": code,
            ".rs": code,
            ".rb": code,
            ".cs": code,
            ".cpp": code,
            ".c": code,
            ".h": code,
            ".hpp": code,
            ".swift": code,
            ".kt": code,
            ".sh": code,
            ".sql": code,
            # Structured
            ".json": JSONParser(),
            ".yaml": TextParser(),
            ".yml": TextParser(),
            ".toml": TextParser(),
            ".ini": TextParser(),
            ".csv": CSVParser(","),
            ".tsv": CSVParser("\t"),
            # Office
            ".pdf": pdf_parser,
            ".docx": DocxParser(),
            ".xlsx": XlsxParser(),
            # Web / books
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".epub": EPUBParser(),
            ".rtf": RTFParser(),
            # Images (OCR)
            ".png": image,
            ".jpg": image,
            ".jpeg": image,
            ".tiff": image,
            ".tif": image,
            ".bmp": image,
            ".webp": image,
        }
        self._fallback: FileParser = TextParser()

    def get(self, extension: str) -> FileParser:
        return self._parsers.get(extension.lower(), self._fallback)

    def register(self, extension: str, parser: FileParser) -> None:
        self._parsers[extension.lower()] = parser


# Module-level singleton — settings can rebuild this at startup if OCR is disabled.
registry = ParserRegistry(
    enable_ocr=os.environ.get("LOGPOSE_ENABLE_OCR", "true").lower() not in {"0", "false", "no"},
)
