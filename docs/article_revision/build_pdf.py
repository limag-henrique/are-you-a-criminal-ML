from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph as RLParagraph, SimpleDocTemplate, Spacer, Table as RLTable, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "article_revision" / "output"


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\u2014", "-").replace("\u2013", "-").replace("\u2011", "-").replace("\u2212", "-"))


def build(source: Path, target: Path):
    doc = Document(source)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyPT", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="TitlePT", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="H1PT", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2PT", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="CaptionPT", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=10, spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="CellPT", parent=styles["BodyText"], fontName="Helvetica", fontSize=6.8, leading=8.2))
    styles.add(ParagraphStyle(name="RefPT", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.3, leading=10, alignment=TA_JUSTIFY, spaceAfter=4))
    flow = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            if text.startswith("Listas públicas de procurados não são"):
                style = styles["TitlePT"]
            elif p.style.name.startswith("Heading 1"):
                style = styles["H1PT"]
            elif p.style.name.startswith("Heading 2"):
                style = styles["H2PT"]
            elif p.style.name == "Caption":
                style = styles["CaptionPT"]
            elif p.style.name == "Referência":
                style = styles["RefPT"]
            else:
                style = styles["BodyPT"]
            if source.name == "material_suplementar.docx" and text.startswith("S3."):
                flow.append(PageBreak())
            flow.append(RLParagraph(esc(text), style))
        elif child.tag.endswith("}tbl"):
            table = Table(child, doc)
            data = [[RLParagraph(esc(cell.text.replace("\n", " ")), styles["CellPT"]) for cell in row.cells] for row in table.rows]
            if not data:
                continue
            usable = 7.1 * inch
            col_w = usable / len(data[0])
            t = RLTable(data, colWidths=[col_w] * len(data[0]), repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA8B8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.extend([t, Spacer(1, 8)])

    def page_number(canvas, document):
        canvas.saveState(); canvas.setFont("Helvetica", 8)
        canvas.drawString(0.7 * inch, 0.45 * inch, "Auditoria sociotécnica de listas públicas de procurados")
        canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Página {document.page}")
        canvas.restoreState()

    pdf = SimpleDocTemplate(str(target), pagesize=letter, leftMargin=.7*inch, rightMargin=.7*inch, topMargin=.7*inch, bottomMargin=.7*inch)
    pdf.build(flow, onFirstPage=page_number, onLaterPages=page_number)


if __name__ == "__main__":
    build(OUT / "artigo_revisado_final.docx", OUT / "artigo_revisado_final.pdf")
    build(OUT / "material_suplementar.docx", OUT / "material_suplementar.pdf")
