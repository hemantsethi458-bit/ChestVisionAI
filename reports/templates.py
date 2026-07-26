"""PDF report layout constants and helpers."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm

PAGE_SIZE = A4
MARGIN = 2.0 * cm

PRIMARY_COLOR = colors.HexColor("#1e3a8a")
ACCENT_COLOR = colors.HexColor("#2563eb")
TEXT_COLOR = colors.HexColor("#111827")
MUTED_COLOR = colors.HexColor("#6b7280")


def build_styles():
    """Create reusable ReportLab paragraph styles."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontSize=20,
            textColor=PRIMARY_COLOR,
            spaceAfter=12,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=MUTED_COLOR,
            spaceAfter=18,
            alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontSize=13,
            textColor=PRIMARY_COLOR,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "BodyTextCustom",
            parent=base["BodyText"],
            fontSize=10,
            textColor=TEXT_COLOR,
            leading=14,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["BodyText"],
            fontSize=8,
            textColor=MUTED_COLOR,
            leading=11,
        ),
    }
