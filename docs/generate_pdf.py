#!/usr/bin/env python3
"""Convert user-guide.md to a styled HTML file optimized for PDF printing."""

import markdown
import os

INPUT = os.path.join(os.path.dirname(__file__), "user-guide.md")
OUTPUT = os.path.join(os.path.dirname(__file__), "HerdAI_User_Guide.html")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --bg: #ffffff;
    --text: #1e293b;
    --text-light: #64748b;
    --border: #e2e8f0;
    --code-bg: #f1f5f9;
    --table-stripe: #f8fafc;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg);
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 50px;
}

/* Cover */
h1:first-of-type {
    font-size: 28pt;
    font-weight: 700;
    color: var(--primary);
    border-bottom: 3px solid var(--primary);
    padding-bottom: 12px;
    margin-bottom: 8px;
}

h1:first-of-type + p {
    font-size: 13pt;
    color: var(--text-light);
    margin-bottom: 30px;
}

h1 {
    font-size: 22pt;
    font-weight: 700;
    color: var(--primary-dark);
    margin-top: 40px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
    page-break-after: avoid;
}

h2 {
    font-size: 16pt;
    font-weight: 600;
    color: var(--text);
    margin-top: 32px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    font-weight: 600;
    color: var(--text);
    margin-top: 24px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

p { margin-bottom: 12px; }

a {
    color: var(--primary);
    text-decoration: none;
}

strong { font-weight: 600; }

/* Code */
code {
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    font-size: 9.5pt;
    background: var(--code-bg);
    padding: 2px 5px;
    border-radius: 4px;
    color: #be185d;
}

pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 16px 20px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 12px 0 16px 0;
    page-break-inside: avoid;
    line-height: 1.5;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: 9pt;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 20px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

th {
    background: var(--primary);
    color: white;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
}

td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
}

tr:nth-child(even) { background: var(--table-stripe); }

/* Lists */
ul, ol {
    margin: 8px 0 16px 24px;
}

li { margin-bottom: 4px; }

/* Horizontal rule */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* Blockquote */
blockquote {
    border-left: 4px solid var(--primary);
    background: #eff6ff;
    padding: 12px 20px;
    margin: 12px 0;
    border-radius: 0 8px 8px 0;
}

/* Print styles */
@media print {
    body {
        padding: 20px 30px;
        font-size: 10pt;
    }

    h1 { page-break-before: always; }
    h1:first-of-type { page-break-before: avoid; }

    pre {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }

    pre code {
        color: #1e293b !important;
    }

    th {
        background: #1e293b !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }

    a { color: var(--primary) !important; }

    .no-print { display: none; }
}

/* ASCII diagrams */
pre:has(code) {
    font-size: 9pt;
}
"""

def main():
    with open(INPUT, "r") as f:
        md_text = f.read()

    extensions = ["tables", "fenced_code", "codehilite", "toc", "attr_list"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HerdAI User Guide</title>
    <style>{CSS}</style>
</head>
<body>
    <div class="no-print" style="background:#eff6ff;padding:12px 20px;border-radius:8px;margin-bottom:24px;border:1px solid #bfdbfe;">
        <strong>To save as PDF:</strong> Press <code>Cmd+P</code> (Mac) or <code>Ctrl+P</code> (Windows/Linux) → Destination: "Save as PDF" → Save
    </div>
    {html_body}
    <footer style="margin-top:60px;padding-top:20px;border-top:2px solid #e2e8f0;color:#94a3b8;font-size:9pt;text-align:center;">
        HerdAI User Guide — Production-Grade AI Agent Framework for Go<br>
        Generated {_now()}
    </footer>
</body>
</html>"""

    with open(OUTPUT, "w") as f:
        f.write(html)

    print(f"Generated: {OUTPUT}")
    print(f"Open in browser and Print → Save as PDF")

def _now():
    from datetime import datetime
    return datetime.now().strftime('%B %d, %Y')

if __name__ == "__main__":
    main()
