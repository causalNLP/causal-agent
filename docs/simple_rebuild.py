#!/usr/bin/env python3
"""
Simple documentation rebuild script that works without full Sphinx installation.
Creates a basic HTML preview of your RST content.
"""

import os
import re
from pathlib import Path

def rst_to_basic_html(rst_content, title="Documentation"):
    """Convert basic RST to HTML for preview."""
    
    # Basic RST to HTML conversion
    html = rst_content
    
    # Convert headers
    html = re.sub(r'^(.+)\n=+\n', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^(.+)\n-+\n', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^(.+)\n\^+\n', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Convert code blocks
    html = re.sub(r'\.\. code-block:: (\w+)\n\n((?:   .+\n)*)', 
                  r'<pre><code class="\1">\2</code></pre>', html)
    
    # Convert notes
    html = re.sub(r'\.\. note::\n((?:   .+\n)*)', 
                  r'<div class="note"><strong>Note:</strong>\1</div>', html)
    
    # Convert bullet points
    html = re.sub(r'^\* (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
    
    # Convert bold text
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert italic text
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Convert inline code
    html = re.sub(r'``(.+?)``', r'<code>\1</code>', html)
    
    # Convert line breaks
    html = html.replace('\n\n', '</p><p>')
    html = '<p>' + html + '</p>'
    
    # Clean up
    html = html.replace('<p></p>', '')
    html = html.replace('<p><h', '<h')
    html = html.replace('</h1></p>', '</h1>')
    html = html.replace('</h2></p>', '</h2>')
    html = html.replace('</h3></p>', '</h3>')
    html = html.replace('<p><ul>', '<ul>')
    html = html.replace('</ul></p>', '</ul>')
    html = html.replace('<p><div', '<div')
    html = html.replace('</div></p>', '</div>')
    html = html.replace('<p><pre>', '<pre>')
    html = html.replace('</pre></p>', '</pre>')
    
    # Wrap in HTML document
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        code {{ 
            background: #f8f9fa; 
            padding: 2px 4px; 
            border-radius: 3px; 
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        pre {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}
        .note {{ 
            background: #e8f4fd; 
            padding: 15px; 
            border-radius: 5px; 
            border-left: 4px solid #3498db;
            margin: 15px 0;
        }}
        ul {{ margin: 15px 0; }}
        li {{ margin: 5px 0; }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .toc h3 {{ margin-top: 0; }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc li {{ margin: 8px 0; }}
        .toc a {{ text-decoration: none; color: #3498db; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    {html}
    
    <hr style="margin: 40px 0; border: none; border-top: 1px solid #bdc3c7;">
    <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
        <em>This is a basic preview. For full documentation features, build with Sphinx.</em>
    </p>
</body>
</html>
    """
    
    return full_html

def create_preview():
    """Create a basic HTML preview of the documentation."""
    print("🔍 Creating documentation preview...")
    
    source_dir = Path("source")
    if not source_dir.exists():
        print("❌ Source directory not found. Run from docs/ directory.")
        return
    
    # Create preview directory
    preview_dir = Path("preview")
    preview_dir.mkdir(exist_ok=True)
    
    # Process main index file
    index_file = source_dir / "index.rst"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html = rst_to_basic_html(content, "CAIS Documentation")
        
        with open(preview_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html)
        
        print("✅ Created preview/index.html")
    
    # Process other RST files
    rst_files = list(source_dir.rglob("*.rst"))
    
    for rst_file in rst_files:
        if rst_file.name == "index.rst" and rst_file.parent == source_dir:
            continue  # Already processed
        
        try:
            with open(rst_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create relative path for output
            rel_path = rst_file.relative_to(source_dir)
            output_path = preview_dir / rel_path.with_suffix('.html')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            html = rst_to_basic_html(content, f"CAIS - {rst_file.stem}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"✅ Created {output_path}")
            
        except Exception as e:
            print(f"⚠️ Error processing {rst_file}: {e}")
    
    # Create a simple index of all files
    create_file_index(preview_dir, rst_files)
    
    print(f"\n🎉 Preview created in preview/ directory")
    print(f"📖 Open preview/index.html in your browser to view")
    print(f"📁 Or run: python -m http.server 8000 (from preview/ directory)")

def create_file_index(preview_dir, rst_files):
    """Create an index of all documentation files."""
    
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>CAIS Documentation - File Index</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .file-list { list-style-type: none; padding: 0; }
        .file-list li { margin: 10px 0; }
        .file-list a { text-decoration: none; color: #3498db; }
        .file-list a:hover { text-decoration: underline; }
        .section { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>📚 CAIS Documentation Files</h1>
    <p><a href="index.html">🏠 Main Documentation</a></p>
    
    <div class="section">
        <h2>📄 All Documentation Files:</h2>
        <ul class="file-list">
    """
    
    source_dir = Path("source")
    for rst_file in sorted(rst_files):
        rel_path = rst_file.relative_to(source_dir)
        html_path = rel_path.with_suffix('.html')
        
        # Skip the main index as it's already linked above
        if str(html_path) == "index.html":
            continue
            
        index_html += f'<li>📄 <a href="{html_path}">{rel_path}</a></li>\n'
    
    index_html += """
        </ul>
    </div>
    
    <hr>
    <p><em>This is a basic preview. For full Sphinx features, install dependencies and run: make html</em></p>
</body>
</html>
    """
    
    with open(preview_dir / "files.html", 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print("✅ Created preview/files.html (file index)")

if __name__ == "__main__":
    print("🚀 Simple Documentation Preview Generator")
    print("=" * 45)
    create_preview()