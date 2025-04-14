#!/usr/bin/env python

"""
Script to convert quickstart_tutorial.py to a Jupyter notebook.
This script uses the structure of the Python file with its code and markdown comments
to create a proper Jupyter notebook.
"""

import json
import re

def parse_python_file(filename):
    """Parse a Python file with markdown comments into notebook cells."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content by comment blocks that start with #
    # Regex pattern: matches lines that start with # and continues until a non-comment line
    pattern = r'(?:^|\n)(\s*#.+?)(?=\n\S|$)'
    parts = re.split(pattern, content, flags=re.DOTALL)
    
    # First part will be empty or non-comment code
    cells = []
    
    # Process parts
    for i in range(len(parts)):
        part = parts[i].strip()
        if not part:
            continue
            
        # If part starts with #, it's a markdown cell
        if part.startswith('#'):
            # Convert Python comments to markdown
            markdown = '\n'.join(line[1:].lstrip() for line in part.split('\n'))
            cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': markdown.split('\n')
            })
        else:
            # It's a code cell
            cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': part.split('\n')
            })
    
    return cells

def create_notebook(cells, output_filename):
    """Create a Jupyter notebook from cells."""
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.10'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    input_file = "quickstart_tutorial.py"
    output_file = "quickstart_tutorial.ipynb"
    
    print(f"Converting {input_file} to {output_file}...")
    cells = parse_python_file(input_file)
    create_notebook(cells, output_file)
    print(f"Conversion complete! Created {output_file}") 