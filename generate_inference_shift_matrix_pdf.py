"""Generates the single-page Algorithmic Inference Shift Matrix PDF for the Investment Committee."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle
)

from mutual_fund_ml.config import get_resolved_path

# --- Corporate palette (per spec) ---
DEEP_NAVY = colors.HexColor("#1A2E40")
LIGHT_BLUE = colors.HexColor("#F4F7F9")
CHARCOAL = colors.HexColor("#333333")
WHITE = colors.white

OUTPUT_PATH = get_resolved_path("outputs/final_report") / "Algorithmic_Inference_Shift_Matrix.pdf"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MARGIN = 0.5 * inch
FOOTER_TEXT = "CONFIDENTIAL | FOR BOARD AND INVESTMENT COMMITTEE REVIEW ONLY | METRIC DECOUPLING RECONCILIATION"

styles = {
    "title": ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=13.5, leading=15.5,
        textColor=DEEP_NAVY, spaceAfter=2, alignment=TA_LEFT,
    ),
    "subtitle": ParagraphStyle(
        "subtitle", fontName="Helvetica-Oblique", fontSize=8, leading=9.5,
        textColor=CHARCOAL, spaceAfter=5,
    ),
    "col_header": ParagraphStyle(
        "col_header", fontName="Helvetica-Bold", fontSize=8.5, leading=10,
        textColor=WHITE, alignment=TA_LEFT,
    ),
    "track_name": ParagraphStyle(
        "track_name", fontName="Helvetica-Bold", fontSize=7.8, leading=9.3,
        textColor=DEEP_NAVY, alignment=TA_LEFT,
    ),
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=7.8, leading=9.3,
        textColor=CHARCOAL, alignment=TA_LEFT,
    ),
}

# Each row: (Model / Methodology, Investor Takeaway)
ROWS = [
    ("Pooled OLS Regression (Time Controls)",
     "Calendar month explains <b>97.8%</b> of return variance; fund attributes explain under <b>7.5%</b>. "
     "Favor low-cost passive or smart-beta exposure over tactical manager rotation."),
    ("Principal Component Analysis (PCA)",
     "PC1 explains <b>48.65%</b> of structural variance, driven by portfolio concentration. Use it as a "
     "clean risk gauge, independent of return noise."),
    ("K-Means Clustering",
     "Groups funds into 3 tiers: <b>Diversified</b> (n=198), <b>Core</b> (n=282), <b>Ultra-Concentrated</b> "
     "(n=36). Cap allocations to Ultra-Concentrated managers as AUM grows."),
    ("Hierarchical (Ward) Clustering",
     "Cross-checks the K-Means grouping (cophenetic correlation 0.66). Confirms the 3-tier structure is "
     "robust, not a modeling artifact."),
    ("Naive Zero-Return / Persistence Baselines",
     "The simplest possible guess beat every fitted model on the frozen holdout. Sets the true bar any "
     "forecasting claim must clear."),
    ("OLS on Principal Components (PCR)",
     "Fits well in-sample (82% R2) but was never validated on a forecasting holdout. Explanatory only, "
     "not a deployable forecast."),
    ("Ridge Regression",
     "Best-performing fitted challenger; still failed to beat the zero-return baseline on the frozen "
     "holdout."),
    ("Lasso Regression",
     "Performs in line with Ridge. Adds no incremental forecasting value."),
    ("Elastic Net",
     "Performs in line with Ridge and Lasso. Adds no incremental forecasting value."),
    ("Decision Tree",
     "Its top-ranked feature showed no reliable predictive power once tested on genuinely unseen future "
     "months."),
    ("Random Forest",
     "Failed to beat Ridge or the zero-return baseline on the frozen holdout."),
    ("Gradient Boosting",
     "Looked near-perfect in an early in-sample test (99.8% R2); confirmed as an overfitting artifact, "
     "not a real forecast."),
    ("HistGradientBoosting",
     "Formally rejected: negative R-squared on the true out-of-sample holdout."),
]


def draw_footer(canvas, doc):
    canvas.saveState()
    line_y = 0.42 * inch
    canvas.setStrokeColor(DEEP_NAVY)
    canvas.setLineWidth(1.6)
    canvas.line(MARGIN, line_y, letter[0] - MARGIN, line_y)
    canvas.setFont("Helvetica-Bold", 6.8)
    canvas.setFillColor(DEEP_NAVY)
    canvas.drawCentredString(letter[0] / 2.0, line_y - 12, FOOTER_TEXT)
    canvas.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=0.65 * inch,
        title="Algorithmic Inference Shift Matrix",
    )

    story = []
    story.append(Paragraph("Algorithmic Inference Shift Matrix", styles["title"]))
    story.append(Paragraph(
        "Every model run in this workspace, reconciled against verified pipeline outputs: "
        "Investment Committee reference table",
        styles["subtitle"],
    ))

    header_row = [
        Paragraph("Model / Methodology", styles["col_header"]),
        Paragraph("Investor Takeaway", styles["col_header"]),
    ]

    data = [header_row]
    for name, takeaway in ROWS:
        data.append([
            Paragraph(name, styles["track_name"]),
            Paragraph(takeaway, styles["body"]),
        ])

    col_widths = [1.85 * inch, 5.65 * inch]
    table = Table(data, colWidths=col_widths, repeatRows=1)

    table_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), DEEP_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D2D9")),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, DEEP_NAVY),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            table_style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE))
        else:
            table_style_cmds.append(("BACKGROUND", (0, i), (-1, i), WHITE))

    table.setStyle(TableStyle(table_style_cmds))
    story.append(table)

    doc.build(story, onFirstPage=draw_footer)
    print(f"PDF written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
