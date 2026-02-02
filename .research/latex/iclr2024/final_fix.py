#!/usr/bin/env python3
import re

# Read the file
with open('main.tex', 'r') as f:
    lines = f.readlines()

# Process line by line for better control
fixed_lines = []
for i, line in enumerate(lines):
    # Fix Unicode minus sign (U+2212) to ASCII hyphen
    line = line.replace('−', '-')
    
    # Fix title line with \@
    if 'CFS-eSAFE:' in line and '\\title' in line:
        line = line.replace('CFS-eSAFE:', 'CFS-eSAFE\\@:')
    
    # Fix \label commands - remove trailing spaces
    if '\\label{' in line:
        line = re.sub(r'(\\label\{[^}]+\})\s+$', r'\1%\n', line)
        if not line.endswith('\n'):
            line += '\n'
    
    # Add non-breaking spaces before \cite (but not if already has ~)
    if '\\cite{' in line and not line[line.index('\\cite{')-1] == '~':
        line = re.sub(r'([a-zA-Z])\s+\\cite\{', r'\1~\\cite{', line)
    
    # Fix numbered lists - replace "1)" with "1."
    line = re.sub(r'^(\d+)\)\s+', r'\1.~', line)
    
    # Fix ellipsis
    if 'person, ..."' in line:
        line = line.replace('person, ..."', 'person,\\ldots"')
    
    # Fix interword spacing before (filename:
    if '(filename:' in line and '.\\ (filename:' not in line:
        line = re.sub(r'([a-z])\. \(filename:', r'\1.\\ (filename:', line)
    
    # Fix specific variables that need math mode - only in specific contexts
    # x_i, d_i, E_lambda, etc. in natural text (not already in math mode)
    if not '\\texttt{' in line and not '$' in line or line.count('$') % 2 == 0:
        # Only fix if not already in some form of math/code context
        line = re.sub(r'\bx_i\b', r'$x_i$', line)
        line = re.sub(r'\bd_i\b', r'$d_i$', line)
        line = re.sub(r'\bE_lambda\b', r'$E_{\\lambda}$', line)
        line = re.sub(r'\bs_j\b', r'$s_j$', line)
        line = re.sub(r'\bH0_opt\b', r'$H0_{\\text{opt}}$', line)
        line = re.sub(r'\bH0_all\b', r'$H0_{\\text{all}}$', line)
        line = re.sub(r'\bH0_sj\b', r'$H0_{sj}$', line)
        line = re.sub(r'\bD_opt\b', r'$D_{\\text{opt}}$', line)
        line = re.sub(r'\bA_disc\b', r'$A_{\\text{disc}}$', line)
        line = re.sub(r'\bA_conf\b', r'$A_{\\text{conf}}$', line)
    
    # Fix curly braces in sets - need to be in math mode
    line = re.sub(r'in \{0, 1\}', r'in $\\{0, 1\\}$', line)
    line = re.sub(r'in \{-1, 0, 1\}', r'in $\\{-1, 0, 1\\}$', line)
    line = re.sub(r'lies in \{-1, 0, 1\}', r'lies in $\\{-1, 0, 1\\}$', line)
    line = re.sub(r'bounded in \[-1, 1\]', r'bounded in $[-1, 1]$', line)
    
    # Fix exp(...) and min(...) to use math mode properly - but only simple cases
    if '= exp(nll_max)' in line:
        line = line.replace('= exp(nll_max)', '= $\\exp(\\textit{nll\\_max})$')
    if 'min(SST-2, Yelp, IMDb)' in line:
        line = line.replace('min(SST-2, Yelp, IMDb)', '$\\min$(SST-2, Yelp, IMDb)')
    
    # Fix identity terms set
    if 'terms {muslim, black, gay, disabled}' in line:
        line = line.replace('terms {muslim, black, gay, disabled}', 
                           'terms $\\{\\text{muslim, black, gay, disabled}\\}$')
    
    # Fix function notation like r(P, x), z(P, x) etc - these are OK in text, no fix needed
    
    fixed_lines.append(line)

# Write the file
with open('main.tex', 'w') as f:
    f.writelines(fixed_lines)

print("Applied final fixes")
