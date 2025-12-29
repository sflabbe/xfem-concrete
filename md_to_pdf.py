#!/usr/bin/env python3
"""Convert markdown summary to PDF"""

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# Read markdown file
with open('FUNCTION_SUMMARY.md', 'r') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc']
)

# Create full HTML document with styling
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>XFEM Concrete Function Summary</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
            @top-center {{
                content: "XFEM Concrete Code - Function Summary";
                font-family: sans-serif;
                font-size: 10pt;
                color: #666;
            }}
            @bottom-center {{
                content: "Page " counter(page) " of " counter(pages);
                font-family: sans-serif;
                font-size: 9pt;
                color: #666;
            }}
        }}
        body {{
            font-family: 'DejaVu Sans', Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #333;
        }}
        h1 {{
            color: #1a5490;
            font-size: 20pt;
            margin-top: 20pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }}
        h2 {{
            color: #2563eb;
            font-size: 16pt;
            margin-top: 16pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 4pt;
        }}
        h3 {{
            color: #3b82f6;
            font-size: 13pt;
            margin-top: 12pt;
            margin-bottom: 6pt;
            page-break-after: avoid;
        }}
        h4 {{
            color: #60a5fa;
            font-size: 11pt;
            margin-top: 10pt;
            margin-bottom: 5pt;
            font-weight: bold;
            page-break-after: avoid;
        }}
        p {{
            margin: 6pt 0;
            text-align: justify;
        }}
        code {{
            background-color: #f3f4f6;
            padding: 2pt 4pt;
            border-radius: 2pt;
            font-family: 'DejaVu Sans Mono', monospace;
            font-size: 9pt;
        }}
        pre {{
            background-color: #f3f4f6;
            padding: 8pt;
            border-radius: 4pt;
            border-left: 3px solid #3b82f6;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        ul, ol {{
            margin: 6pt 0;
            padding-left: 20pt;
        }}
        li {{
            margin: 3pt 0;
        }}
        hr {{
            border: none;
            border-top: 1px solid #d1d5db;
            margin: 15pt 0;
        }}
        strong {{
            color: #1f2937;
            font-weight: 600;
        }}
        .toc {{
            background-color: #f9fafb;
            padding: 10pt;
            border: 1px solid #e5e7eb;
            margin: 10pt 0;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Configure fonts
font_config = FontConfiguration()

# Convert HTML to PDF
HTML(string=full_html).write_pdf(
    'FUNCTION_SUMMARY.pdf',
    font_config=font_config
)

print("PDF generated successfully: FUNCTION_SUMMARY.pdf")
