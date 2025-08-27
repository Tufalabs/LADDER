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
    # Semifinal #1
    "integrate(x*(e**(-x) + 1)/(e**x - 1), (x, 0, oo))",
    "integrate(x**5 * e**(-x) * sin(x), (x, 0, oo))",
    "integrate(2*log(log((x+1)/x)) / log((x**2 - x + 17/4)), (x, 1, 2))",
    "integrate(((x**3 - 3*x)**3 - 3*(x**3 - 3*x)*sqrt(x**2 - 4)), (x, 2, 5))",
    # Semifinal #2 (MIT Integration Bee: Semifinal Problems)
    "integrate((sqrt(x+1) - sqrt(x))**pi, (x, 0, pi))",
    "integrate( (((x**2 - 2)**2 - 2)**2 - 2)**2 - 2, (x, -2, 2) )",
    "integrate(tanh(x)/(x*cosh(2*x)), (x, 0, oo))",
    "integrate(sin(4*atan(x)), x)",
    # Semifinal MIT Integration Bee: Semifinal #1 (second block)
    "integrate(e**(cos(x))*cos(2*x + sin(x)), x)",
    "integrate(9*x**(9 - x**90) + 9*x**(99 - x**900) + 9*x**(909 - x**990) + 9*x**(999 - x**9000) + ... , (x, 0, 1))",
    "integrate((2*cos(x) - cos(2021*x) - 2*cos(2022*x) - cos(2023*x) + 2)/(1 - cos(2*x)), (x, 0, pi))",
    "integrate((3*log(x) - 1 + 2*x)/(x*log(x) + x**2 + 2*x**4), x)",
    # Semifinal MIT Integration Bee: Semifinal Tiebreakers
    "integrate(sec(x)**5, x)",
    "integrate(sech(2*x + 1 - 1/(x - 1) - 2/(x + 1)), (x, -oo, oo))",
    "integrate(sin(x)*sin(2*x)*sin(3*x)/x**3, (x, 0, oo))",
    "integrate((1 + log(x))*(1 + log(log(x))), x)",
    "integrate(e**(-x/2)*((x**2 + 1)/2)**2, (x, 0, oo))",
    "integrate(tan(x)*sec(x)**2*cos(2*x)*e**(2*cos(x)), x)",
    # Semifinal MIT Integration Bee: Semifinal #2 (second block; duplicates omitted)
    # Next block (if any new unique items appear, they are appended below)
    # (For brevity, further ambiguous or duplicate items have been omitted.)
    
    # --- Additional Semifinal Problems (from your later block) ---
    # Semifinal #1 Problem (second set)
    "integrate(e**(cos(x))*cos(2*x + sin(x)), x)",
    "integrate(x*e**(-2*x)/(e**(-x)+1), (x, 0, oo))",
    "integrate(sin(cot(x)**2)*sec(x)**2, (x, 0, pi/2))",
    "integrate(cosh(x)**2*tanh(2*x), x)",
    # Semifinal Tiebreakers (if not duplicate)
    "integrate(sec(x)**5, x)",  # already included above
    "integrate(sech(2*x+1 - 1/(x-1) - 2/(x+1)), (x, -oo, oo))",  # duplicate
    "integrate(sin(x)*sin(2*x)*sin(3*x)/x**3, (x, 0, oo))",         # duplicate
    "integrate((1+log(x))*(1+log(log(x))), x)",                      # duplicate
    "integrate(e**(-x/2)*((x**2+1)/2)**2, (x, 0, oo))",               # duplicate
    "integrate(tan(x)*sec(x)**2*cos(2*x)*e**(2*cos(x)), x)",          # duplicate
]
