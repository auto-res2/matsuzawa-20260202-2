#!/usr/bin/env python3
import re

# Read the file
with open('main.tex', 'r') as f:
    content = f.read()

# Fix Warning 24: Remove ALL trailing spaces after %
content = re.sub(r'%\s+\n', '%\n', content)

# Fix Warning 11: Use \ldots in the specific line about "..."
content = re.sub(r'"As a Muslim person, \.\.\."', r'"As a Muslim person,\\ldots"', content)

# Fix Warning 36 on line 103: Add space before (NLL)
content = re.sub(r'log-likelihood \(NLL\)', r'log-likelihood\\ (NLL)', content)

# Fix Warning 36 on line 105: These are function notation z(Pc, x_i) and z(P, x_i)
# ChkTeX wants a space, but in math notation this is incorrect. We'll ignore these.

# Fix Warning 36 on line 174: exp(...)
content = re.sub(r'\$\\exp\$\(\\texttt\{nll\\_max\}\)', r'$\\exp(\\texttt{nll\\_max})$', content)

# Fix Warning 36 on line 176: min(...)
content = re.sub(r'min\(SST-2, Yelp, IMDb\)', r'$\\min$(SST-2, Yelp, IMDb)', content)

# Fix Warning 12 on line 246: Add interword spacing
content = re.sub(r'regression rate\. \(filename: comparison_metrics_table\.pdf\)', 
                 r'regression rate.\\ (filename: comparison_metrics_table.pdf)', content)

# Write the file
with open('main.tex', 'w') as f:
    f.write(content)

print("Fixed final ChkTeX warnings")
