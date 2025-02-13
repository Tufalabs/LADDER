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
    # Finals Problem 1
    "integrate(sqrt((sin(20*x) + 3*sin(21*x) + sin(22*x))**2 + (cos(20*x) + 3*cos(21*x) + cos(22*x))**2), x)",
    # Finals Problem 2
    "integrate(e**(-2*x)*sin(3*x)/x, (x, 0, oo))",
    # Finals Problem 3
    "integrate(cos(2022*x)*sin(10050*x)*sin(50*x)*sin(10251*x)*sin(51*x), (x, 0, 2*pi))",
    # Finals Problem 4
    "integrate(x**(1/3)*(1-x)**(2/3), (x, 0, 1))",
    # Finals Problem 5
    # Interpreting the printed notation as the logarithm (base 10) of an integral:
    "log10(integrate(10**(-x**3), (x, 2022, oo)))",
    
    # (Second block of Finals problems)
    # Finals Problem 1 (alternate set)
    "integrate(sqrt(3)*tan(x)*(sin(x)+cos(x))**2, (x, 0, pi/2))",
    # Finals Problem 2 (alternate set)
    "integrate(((sin(2*x)*sin(3*x)*sin(5*x)*sin(30*x))/(sin(x)*sin(6*x)*sin(10*x)*sin(15*x)))**2, (x, 0, pi))",
    # Finals Problem 3 (alternate set)
    "integrate(sqrt(x**2+1) + cbrt(x**4+x**2+1), (x, -1/2, 1/2))",
    # Finals Problem 4 (alternate set)
    "1020*integrate(x**29 - 48*x**10 + 575, (x, 2, oo))",
    # Finals Problem 5 (alternate set)
    "integrate(sum((floor(2**n*x))**(3*n)/factorial(2), (n, 1, oo)), (x, 0, 1))",
    
    # (Next block – additional Finals problems)
    # Finals Problem 1 (third set)
    "integrate(e**(x/2)*cos(x)/(sqrt(3)*(3*cos(x)+4*sin(x))), x)",
    # Finals Problem 2 (third set)
    "integrate(log(2*e**x - 1)/(e**x - 1), (x, 0, oo))",
    # Finals Problem 3 (third set)
    "integrate(1/(x**4 + x**3 + x**2 + x + 1), (x, -oo, oo))",
    # Finals Problem 4 (third set)
    "integrate(3*sqrt(1+x**3) + sqrt(1-x**3), (x, -1/3, 1))",
    # Finals Problem 5 (third set)
    "integrate(max_{n>=0}((floor(2**n*x) - 2**n*x - 1/4)/2**n), (x, 0, 1))",
    
    # Finals Tiebreakers
    # Tiebreakers Problem 1
    "integrate(1/sqrt(4*x**4+1), x)",
    # Tiebreakers Problem 2
    "integrate(((sin(2*x)-5*sin(x))*sin(x))/(cos(2*x)-10*cos(x)+13), (x, 0, 2*pi))",
    # Tiebreakers Problem 3
    "integrate(sqrt(x**4 - 4*x + 3), x)",
    # Tiebreakers Problem 4
    "integrate(sin(2*x)**2*cos(3*x)**2/(4*cos(2*x)**2*(4*cos(3*x)**2-3)), (x, -oo, oo))",
    # Tiebreakers Problem 5
    "integrate((floor(x)*x**2)/(x**6-1), (x, 2, oo))",
    
    # Lightning Round
    # Lightning Round Problem 1
    "integrate((1 - x**3/2)**(3/2) - (1 - x**2/3)**(2/3), (x, 0, 1))",
    # Lightning Round Problem 2
    "integrate((x/(x-1))**4, x)",
    # Lightning Round Problem 3
    "integrate((tan(1012*x)+tan(1013*x))*cos(1012*x)*cos(1013*x)/cos(2025*x), x)",
]
