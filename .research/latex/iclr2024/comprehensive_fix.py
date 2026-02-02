#!/usr/bin/env python3
import re

# Read the file
with open('main.tex', 'r') as f:
    content = f.read()

# Fix Unicode minus sign (U+2212) to ASCII hyphen
content = content.replace('−', '-')

# Fix title line with \@
content = re.sub(r'(CFS-eSAFE):', r'\1\\@:', content)

# Fix \label commands - remove trailing spaces
content = re.sub(r'(\\label\{[^}]+\})\s+\n', r'\1%\n', content)

# Add non-breaking spaces before \cite
content = re.sub(r'(?<!~)\s+(\\cite\{)', r'~\1', content)

# Fix numbered lists - replace "1)" with "1."
content = re.sub(r'^(\d+)\)\s+', r'\1.~', content, flags=re.MULTILINE)

# Fix ellipsis
content = re.sub(r'person, \.\.\."', r'person,\\ldots"', content)

# Fix interword spacing before (filename:
content = re.sub(r'performance\. \(filename:', r'performance.\\ (filename:', content)

# Fix math mode issues - wrap variables with underscores in $...$
# These need to be in math mode
content = re.sub(r'\b([a-zA-Z])_([a-zA-Z0-9]+)\b', r'$\1_{\2}$', content)

# Fix specific function calls that should be in text mode
content = re.sub(r'\\texttt\{([^}]+)\}', lambda m: r'\texttt{' + m.group(1).replace('$', '').replace('_', r'\_') + '}', content)

# Fix curly braces in sets - need to be in math mode
content = re.sub(r'in \{0, 1\}', r'in~$\\{0, 1\\}$', content)
content = re.sub(r'in \{-1, 0, 1\}', r'in~$\\{-1, 0, 1\\}$', content)
content = re.sub(r'lies in \{-1, 0, 1\}', r'lies in~$\\{-1, 0, 1\\}$', content)
content = re.sub(r'bounded in \[-1, 1\]', r'bounded in~$[-1, 1]$', content)

# Fix specific problematic sections by putting them properly
# Fix exp(...) and min(...) to use math mode properly
content = re.sub(r'= exp\(([^)]+)\)', r'= $\\exp(\1)$', content)
content = re.sub(r'min\(SST-2, Yelp, IMDb\)', r'$\\min(\\text{SST-2, Yelp, IMDb})$', content)

# Fix identity terms set
content = re.sub(r'terms \{muslim, black, gay, disabled\}', r'terms $\\{\\text{muslim, black, gay, disabled}\\}$', content)

# Write the file
with open('main.tex', 'w') as f:
    f.write(content)

print("Applied comprehensive fixes")
