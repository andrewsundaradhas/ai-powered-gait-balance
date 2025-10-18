"""
Clinical report generator (Step 11)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class ClinicalReportGenerator:
	def __init__(self) -> None:
		self.styles = getSampleStyleSheet()
		self.title_style = ParagraphStyle(
			"CustomTitle",
			parent=self.styles["Heading1"],
			fontSize=20,
			textColor=colors.HexColor("#1f77b4"),
			spaceAfter=12,
			alignment=1,
		)
		self.heading_style = ParagraphStyle(
			"CustomHeading",
			parent=self.styles["Heading2"],
			fontSize=13,
			textColor=colors.HexColor("#2c3e50"),
			spaceAfter=10,
			spaceBefore=10,
		)
		self.normal_style = ParagraphStyle("CustomNormal", parent=self.styles["Normal"], fontSize=11, spaceAfter=6)

	def generate_patient_report(self, analysis_data: dict, output_path: str = "reports/patient_report.pdf") -> str:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		doc = SimpleDocTemplate(output_path, pagesize=letter)
		elements: List = []
		elements.append(Paragraph("GAIT & BALANCE ANALYSIS REPORT", self.title_style))
		elements.append(Spacer(1, 0.2 * inch))
		elements.append(Paragraph("PATIENT INFORMATION", self.heading_style))
		patient_info = [
			["Patient ID:", analysis_data.get("patient_id", "N/A")],
			["Name:", analysis_data.get("name", "N/A")],
			["Age:", str(analysis_data.get("age", "N/A"))],
			["Gender:", analysis_data.get("gender", "N/A")],
			["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
			["Clinician:", analysis_data.get("clinician", "N/A")],
		]
		patient_table = Table(patient_info)
		patient_table.setStyle(
			TableStyle(
				[
					("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f8")),
					("ALIGN", (0, 0), (-1, -1), "LEFT"),
					("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
					("FONTSIZE", (0, 0), (-1, -1), 10),
					("GRID", (0, 0), (-1, -1), 1, colors.grey),
				]
			)
		)
		elements.append(patient_table)
		elements.append(Spacer(1, 0.3 * inch))
		elements.append(Paragraph("ANALYSIS RESULTS", self.heading_style))
		results_data = [
			["Metric", "Result", "Confidence"],
			["Gait Type", analysis_data.get("gait_type", "N/A"), f"{analysis_data.get('gait_confidence', 0):.1%}"],
			["Balance Status", analysis_data.get("balance_status", "N/A"), f"{analysis_data.get('balance_confidence', 0):.1%}"],
			["Suspected Disorder", analysis_data.get("disorder", "None"), f"{analysis_data.get('disorder_confidence', 0):.1%}"],
		]
		results_table = Table(results_data)
		results_table.setStyle(
			TableStyle(
				[
					("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
					("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
					("ALIGN", (0, 0), (-1, -1), "CENTER"),
					("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
					("FONTSIZE", (0, 0), (-1, 0), 11),
					("GRID", (0, 0), (-1, -1), 1, colors.black),
				]
			)
		)
		elements.append(results_table)
		elements.append(Spacer(1, 0.3 * inch))
		elements.append(Paragraph("CLINICAL RECOMMENDATIONS", self.heading_style))
		for i, rec in enumerate(analysis_data.get("recommendations", []), 1):
			elements.append(Paragraph(f"<b>{i}.</b> {rec}", self.normal_style))
		elements.append(Spacer(1, 0.3 * inch))
		elements.append(Paragraph(f"<i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", self.normal_style))
		doc.build(elements)
		return output_path

	def export_to_csv(self, analysis_results: list, output_path: str = "exports/analysis_data.csv") -> str:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		pd.DataFrame(analysis_results).to_csv(output_path, index=False)
		return output_path


