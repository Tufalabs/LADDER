"""
base_questions.py

This file contains a list of base integral problems that are considered challenging.
Each problem is represented as a string which we expect to follow the pattern:
    integrate(<integrand>, x)
For example: "integrate(1/(x**2 - x + 1), x)"
"""

#2020 - 2024 MIT intergration bee questions
#https://math.mit.edu/~yyao1/integrationbee.html
#just the qualifier tests

BASE_QUESTIONS = [
    "integrate(1/(x**2 - 2*x + 3), x)",
    "integrate(2025/(2023*2024), x)",
    "integrate((x - 1)*log(x+1)/(x + 1)/log(x-1), x)",
    "integrate(x*log(x) + 2*x, x)",
    "integrate(1/(x*log(x) + 2*x), x)",
    "integrate(arccos(sin(x)), x)",
    "integrate((cos(x) + cot(x) + csc(x) + 1)/(sin(x) + tan(x) + sec(x) + 1), x)",
  
]
