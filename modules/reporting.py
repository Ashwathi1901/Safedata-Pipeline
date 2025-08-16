import os, datetime, json
from typing import Dict, Any
import pandas as pd

HTML_TEMPLATE = '''
<html>
<head><meta charset='utf-8'><title>SafeData Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1, h2, h3 { color: #163; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
.small { color: #666; font-size: 12px; }
</style>
</head>
<body>
<h1>SafeData Pipeline Report</h1>
<p class='small'>Generated: {generated}</p>
<h2>Run Summary</h2>
<pre>{summary}</pre>
<h2>Risk Assessment</h2>
<pre>{risk_summary}</pre>
<h2>Utility Metrics</h2>
{utility_tables}
<h2>Compliance Checklist</h2>
{compliance_table}
</body></html>
'''

def df_to_html(df: pd.DataFrame):
    try:
        return df.to_html(index=False)
    except Exception:
        return '<pre>Could not render table.</pre>'

def save_html_report(path: str, summary: Dict[str, Any], risk_summary: Dict[str, Any], util_stats: Dict[str, pd.DataFrame], compliance_df: pd.DataFrame):
    util_tables_html = ''
    for k, v in util_stats.items():
        util_tables_html += f'<h3>{k}</h3>' + df_to_html(v)
    html = HTML_TEMPLATE.format(
        generated=str(datetime.datetime.utcnow()) + ' UTC',
        summary=json.dumps(summary, indent=2, default=str),
        risk_summary=json.dumps(risk_summary, indent=2, default=str),
        utility_tables=util_tables_html,
        compliance_table=df_to_html(compliance_df)
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

def try_make_pdf(html_path: str, pdf_path: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from bs4 import BeautifulSoup
    except Exception:
        return None
    try:
        from bs4 import BeautifulSoup
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        text = soup.get_text('\n')
    except Exception:
        text = 'SafeData Report (text fallback).'
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    x, y = 40, height - 40
    for line in text.split('\n'):
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(x, y, line[:110])
        y -= 14
    c.save()
    return pdf_path