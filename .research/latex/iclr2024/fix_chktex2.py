#!/usr/bin/env python3
import re

# Read the file
with open('main.tex', 'r') as f:
    lines = f.readlines()

# Process line by line
fixed_lines = []
for i, line in enumerate(lines):
    # Fix Warning 24: Remove trailing spaces after \label commands
    if '\\label{' in line:
        line = re.sub(r'(\\label\{[^}]+\})\s+$', r'\1%\n', line)
        if not line.endswith('\n'):
            line += '\n'
    
    # Fix Warning 11 and 26: Use \ldots and remove space before ...
    if '...' in line:
        line = re.sub(r' \.\.\."', r'\\ldots"', line)
    
    # Fix line 99: Missing closing $ and braces
    if i == 98:  # line 99 (0-indexed)
        line = re.sub(r'in~\$\\\{0, 1\}', r'in~$\\{0, 1\\}$', line)
        line = re.sub(r'in~\$\\\{0, 1\} that', r'in~$\\{0, 1\\}$ that', line)
    
    # Fix line 105: Missing closing $
    if i == 104:  # line 105 (0-indexed)
        line = re.sub(r'in~\$\\\{−1, 0, 1\}', r'in~$\\{-1, 0, 1\\}$', line)
    
    # Fix Warning 35/25: Put identifiers in math mode properly
    if 'nll_max' in line or 'ppl_max' in line:
        line = re.sub(r'\bnll_max\b', r'\\texttt{nll\\_max}', line)
        line = re.sub(r'\bppl_max\b', r'\\texttt{ppl\\_max}', line)
        line = re.sub(r'\bexp\(', r'$\\exp$(', line)
    
    # Fix Warning 13: Add \@ after A.
    if 'edits. A is split' in line:
        line = line.replace('edits. A is split', 'edits. A\\ is split')
    
    # Fix Warning 36: Add space before parentheses (function calls)
    # This is context-dependent; we'll only fix obvious mathematical notation
    line = re.sub(r'\bs\(x\)', r's\\ (x)', line)
    line = re.sub(r'\br\(P, x\)', r'r\\ (P, x)', line)
    line = re.sub(r'\bz\(P, x\)', r'z\\ (P, x)', line)
    line = re.sub(r'\bz\(Pc, x\)', r'z\\ (Pc, x)', line)
    line = re.sub(r'min\(alpha_max_spend', r'min\\ (alpha\\_max\\_spend', line)
    line = re.sub(r'example \(x, y\)', r'example\\ (x, y)', line)
    line = re.sub(r'include \(as recorded', r'include\\ (as recorded', line)
    line = re.sub(r'domains \(SST-2', r'domains\\ (SST-2', line)
    
    # Fix Warning 12: Add interword spacing before (filename:
    if '(filename:' in line:
        line = re.sub(r'performance\. \(filename:', r'performance.\\ (filename:', line)
        line = re.sub(r'performance, though all observed values are 0\.0 in the executed runs\. \(filename:', 
                      r'performance, though all observed values are 0.0 in the executed runs.\\ (filename:', line)
    
    fixed_lines.append(line)

# Write the file
with open('main.tex', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed remaining ChkTeX warnings")
