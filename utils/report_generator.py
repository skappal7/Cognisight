"""
PDF Report Generator - Create branded, professional data analysis reports
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from typing import Dict, List, Any
import io
from pathlib import Path


class ReportGenerator:
    """Generate branded PDF reports for data analysis"""
    
    def __init__(self):
        self.brand_colors = {
            'navy': colors.HexColor('#2C3E50'),
            'pink': colors.HexColor('#E91E63'),
            'light_gray': colors.HexColor('#F8F9FA'),
            'dark_gray': colors.HexColor('#666666')
        }
        
        self.logo_url = "https://raw.githubusercontent.com/skappal7/TextAnalyser/refs/heads/main/logo.png"
    
    def generate_report(
        self,
        title: str,
        query: str,
        analysis_text: str,
        data_summary: Dict[str, Any],
        visualizations: List[Dict] = None,
        tables: List[Dict] = None
    ) -> bytes:
        """
        Generate a complete PDF report
        
        Args:
            title: Report title
            query: User's original query
            analysis_text: Main analysis narrative
            data_summary: Data statistics summary
            visualizations: List of visualization info
            tables: List of data tables
            
        Returns:
            PDF as bytes
        """
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Container for PDF elements
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.brand_colors['navy'],
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.brand_colors['navy'],
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            textColor=self.brand_colors['dark_gray'],
            spaceAfter=12,
            alignment=TA_LEFT,
            leading=16
        )
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=self.brand_colors['dark_gray'],
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique'
        )
        
        # Header with logo (if available)
        try:
            # Note: In production, download and use local logo
            story.append(Spacer(1, 0.3*inch))
        except:
            pass
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        metadata_data = [
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Query:', query[:100] + '...' if len(query) > 100 else query]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1.5*inch, 5*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), self.brand_colors['navy']),
            ('TEXTCOLOR', (1, 0), (1, -1), self.brand_colors['dark_gray']),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Horizontal line
        story.append(self._create_line())
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        # Split analysis text into paragraphs
        paragraphs = analysis_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Data Summary Section
        if data_summary:
            story.append(Paragraph("Data Overview", heading_style))
            story.append(self._create_data_summary_table(data_summary))
            story.append(Spacer(1, 0.3*inch))
        
        # Visualizations info (placeholder - actual images would be embedded)
        if visualizations:
            story.append(Paragraph("Visualizations", heading_style))
            for i, viz in enumerate(visualizations, 1):
                story.append(Paragraph(
                    f"{i}. {viz.get('title', 'Visualization')}: {viz.get('chart_type', 'Chart').capitalize()} chart",
                    body_style
                ))
            story.append(Spacer(1, 0.3*inch))
        
        # Data Tables
        if tables:
            story.append(Paragraph("Data Tables", heading_style))
            for table_info in tables[:3]:  # Limit to 3 tables
                if 'dataframe' in table_info:
                    df = table_info['dataframe']
                    # Convert to table (limited rows)
                    story.append(self._create_data_table(df, table_info.get('title', 'Data')))
                    story.append(Spacer(1, 0.2*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(self._create_line())
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Developed with Streamlit with 💗 by CE Team Innovation Lab 2025",
            footer_style
        ))
        story.append(Paragraph(
            "This report was generated by AI Data Analyst Agent | Confidential",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _create_line(self):
        """Create a horizontal line"""
        from reportlab.platypus import Drawing
        from reportlab.graphics.shapes import Line
        
        d = Drawing(6.5*inch, 1)
        d.add(Line(0, 0, 6.5*inch, 0, strokeColor=self.brand_colors['pink'], strokeWidth=2))
        return d
    
    def _create_data_summary_table(self, data_summary: Dict[str, Any]):
        """Create formatted data summary table"""
        data = [
            ['Metric', 'Value'],
            ['Total Rows', f"{data_summary.get('row_count', 'N/A'):,}"],
            ['Total Columns', f"{data_summary.get('column_count', 'N/A')}"],
            ['Memory Usage', f"{data_summary.get('memory_mb', 0):.2f} MB"],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.brand_colors['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.brand_colors['light_gray']),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.brand_colors['light_gray']]),
        ]))
        
        return table
    
    def _create_data_table(self, df, title: str):
        """Create formatted data table from DataFrame (limited rows)"""
        # Convert to pandas for easier table creation
        pandas_df = df.to_pandas().head(10)  # Limit to 10 rows
        
        # Create table data
        data = [list(pandas_df.columns)]  # Header
        for _, row in pandas_df.iterrows():
            data.append([str(val)[:30] for val in row.values])  # Limit cell width
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.brand_colors['pink']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.brand_colors['light_gray']]),
        ]))
        
        return table
