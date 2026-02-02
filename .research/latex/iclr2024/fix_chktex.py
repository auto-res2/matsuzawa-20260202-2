#!/usr/bin/env python3
import re

# Read the file
with open('main.tex', 'r') as f:
    content = f.read()

# Fix Warning 24: Delete spaces before labels
content = re.sub(r'\\label\{([^}]+)\}\s+\n', r'\\label{\1}%\n', content)

# Fix Warning 2: Add non-breaking space before \cite
content = re.sub(r'(?<!~)\s\\cite\{', r'~\\cite{', content)

# Fix Warning 36: Add space before parentheses in math mode
# This one is tricky - we need to add space before {0, 1} and similar constructs
content = re.sub(r'in \{', r'in~$\\{', content)
content = re.sub(r'\}\. A', r'\\}$. A', content)
content = re.sub(r'in \{−1, 0, 1\}', r'in~$\\{-1, 0, 1\\}$', content)

# Fix numbered lists (Warning 10 - solo parenthesis)
content = re.sub(r'^(\d+)\) ', r'\1.~', content, flags=re.MULTILINE)

# Fix Warning 11: Use \ldots instead of ...
content = re.sub(r'"([^"]*)\.\.\."', r'"\1\\ldots"', content)

# Fix Warning 12: Interword spacing (add backslash-space before PDF)
content = re.sub(r'performance\. \(filename:', r'performance.\\ (filename:', content)

# Fix Warning 13: Add \@ after CFS-eSAFE
content = re.sub(r'CFS-eSAFE:', r'CFS-eSAFE\\@:', content)

# Write the file
with open('main.tex', 'w') as f:
    f.write(content)

print("Fixed ChkTeX warnings")
