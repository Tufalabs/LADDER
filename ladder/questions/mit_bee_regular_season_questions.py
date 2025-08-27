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
    # Regular Season Problem 1
    "integrate(max(sin(x), sin(2*x)), (x, 0, 2*pi))",
    # Regular Season Problem 2
    "integrate((x+1)*(x+2)*(x+3)*(x+4), (x, 0, 1))",
    # Regular Season Problem 3
    "integrate(min(2*x, 5 - x**2) - max(-x**2, 2*x - 5), (x, 0, 3))",
    # Regular Season Problem 4 (ambiguous – reproduced as given)
    "integrate(1 - 1/(1-1) ... 1/(1-1)x | {z} 2023 (1−)'s, x)",
    # Regular Season Problem 5
    "integrate(x*cot(x), (x, 0, pi/2))",
    # Regular Season Problem 6
    "integrate(x**(6 + x**4 - x**2 - 1)/x**4 * exp(x+1/x), x)",
    # Regular Season Problem 7
    "integrate(q*(x+1)**3*(x-1), x)",
    # Regular Season Problem 8
    "integrate(x*sin(x)**4, (x, 0, pi))",
    # Regular Season Problem 9
    "integrate(x**5 - 1, x)",
    # Regular Season Problem 10
    "integrate(sin(2*x)*cos(3*x), x)",
    # Regular Season Problem 11
    "integrate(p**2*log(x) + 1/(p**2*log(x)), x)",
    # Regular Season Problem 12
    "integrate(log(cos(x))/cos(x)**2, x)",
    # Regular Season Problem 13
    "integrate(sin(x - sin(x - sin(x - ...))), (x, 0, pi/2 + 1))",
    # Regular Season Problem 14
    "integrate(floor(x)*x*ceiling(x), (x, 0, 100))",
    # Regular Season Problem 15
    "integrate(1/((x-1)**2 + 3*(x-3)**4 + 5*(x-5)**6) * (1/(x-1) + 1/(x-3)**3 + 1/(x-5)**5)**2, (x, -oo, oo))",
    # Regular Season Problem 16
    "integrate(sin(3*x+cos(5*x)**4)**2, (x, 0, pi))",
    # Regular Season Problem 17
    "integrate((-1)**(floor(x) + floor(x/sqrt(2)) + floor(x/sqrt(3))), (x, 0, 5))",
    # Regular Season Problem 18
    "integrate(Product((x+n)/(x+n+1), (n, 1, oo)), (x, 0, oo))",
    # Regular Season Problem 19
    "integrate(sin(23*x)/sin(x), (x, 0, pi/2))",
    # Regular Season Problem 20
    "integrate((floor(x/2)/floor(x) + ceiling(x/2)/ceiling(x)), (x, 1, 100))",
    #
    # Next block (duplicates from earlier problems are omitted)
    #
    # Regular Season Problem 1 (second occurrence)
    "integrate(floor(log(43, x)), (x, 1, 2024))",
    # Regular Season Problem 2 (second occurrence)
    "integrate(1/(x**(2024 - x)*4047), x)",
    # Regular Season Problem 3 (second occurrence)
    "integrate(x**2*(1-x)**2024, (x, 0, 1))",
    # Regular Season Problem 4 (second occurrence)
    "integrate((2023*x + 1)/(x**2 + 2024), x)",
    # Regular Season Problem 5 (second occurrence)
    "integrate(sec(x)**2*exp(-sec(x)**2), (x, 0, pi/2))",
    # Regular Season Problem 6 (second occurrence)
    "integrate(cot(x)*cot(2*x), x)",
    # Regular Season Problem 7 (second occurrence)
    "integrate(sinh(x)**2*tanh(2*x), x)",
    # Regular Season Problem 8 (second occurrence)
    "integrate(arctan(sqrt(x)), x)",
    # Regular Season Problem 9 (second occurrence)
    "integrate(x*log(x)/(x**4 + 1), (x, 0, oo))",
    # Regular Season Problem 10 (second occurrence)
    "integrate(floor(floor(x)), (x, 0, 10))",
    # Regular Season Problem 11 (second occurrence)
    "integrate(e**(-x)*sqrt(1 + cot(arccos(e**(-x)))**2), (x, 0, 1))",
    # Regular Season Problem 12 (second occurrence)
    "integrate(1, (x, 1, 3))",  # interpreted from a sum of ones
    # Regular Season Problem 13 (second occurrence)
    "integrate(2*x*(1-x)**2/(1+x**2), (x, 0, 1))",
    # Regular Season Problem 14 (second occurrence)
    "integrate(e**(e*x) + 3*x, x)",
    # Regular Season Problem 15 (second occurrence)
    "integrate((1 - Abs(x)/sqrt(3))**2, (x, -sqrt(3)/2, sqrt(3)/2))",
    # Regular Season Problem 16 (second occurrence; duplicate)
    # (already included above)
    # Regular Season Problem 17 (second occurrence; duplicate)
    # Regular Season Problem 18 (second occurrence; duplicate)
    # Regular Season Problem 19 (second occurrence; duplicate)
    # Regular Season Problem 20 (second occurrence; duplicate)
    #
    # Next new block:
    # Regular Season Problem 1 (new block)
    "integrate(sqrt(x), (x, 0, 100))",
    # Regular Season Problem 2 (new block)
    "integrate(log(1+x)/x**2, x)",
    # Regular Season Problem 3 (new block; ambiguous limits)
    "integrate(cos(arcsin(arccos(sin(x)))), (x, pi/(2+1), pi/(2-1)))",
    # Regular Season Problem 4 (new block)
    "integrate(Abs((x-2)*(x-1)*x*(x+1)*(x+2)), (x, -2, 2))",
    # Regular Season Problem 5 (new block)
    "integrate(2020*sin(2019*x)*cos(2019*x) - 8084*sin(2021*x)*cos(2021*x), x)",
    # Regular Season Problem 6 (new block)
    "integrate((3*x**3+2*x**2+1)/sqrt(x**3+1), x)",
    # Regular Season Problem 7 (new block; ambiguous – best guess)
    "integrate(1/(sin(x)**4*cos(x)**4), x)",
    # Regular Season Problem 8 (new block)
    "integrate((x+sin(x))/(1+cos(x)), x)",
    # Regular Season Problem 9 (new block)
    "integrate(sinh(x)**3*cosh(x)**2, x)",
    # Regular Season Problem 10 (new block)
    "integrate(4*x**3/(2*x), x)",
    # Regular Season Problem 11 (new block)
    "integrate((cos(x)-sin(x))/(x+cos(x)), x)",
    # Regular Season Problem 12 (new block)
    "integrate(pi*sin(pi*sqrt(x))/sqrt(x), (x, 0, 441))",
    # Regular Season Problem 13 (new block)
    "integrate(tan(x)**2, x)",
    # Regular Season Problem 14 (new block)
    "integrate((x - bxc)**2, (x, 0, 256))",  # bxc is left as-is
    # Regular Season Problem 15 (new block)
    "integrate(exp(sqrt(4)*x), x)",
    # Regular Season Problem 16 (new block)
    "integrate(cos(x)*cot(x), x)",
    # Regular Season Problem 17 (new block)
    "integrate(2*log(x) + log(x)**2, x)",
    # Regular Season Problem 18 (new block)
    "integrate(x**3/(1+x**2), x)",
    # Regular Season Problem 19 (new block)
    "integrate(1/(2-2*x+x**2), x)",
    # Regular Season Problem 20 (new block)
    "integrate(sin(x)*log(sin(x)), x)",
    #
    # Next block:
    # Regular Season Problem 21
    "integrate(x/(1-x)**4, x)",
    # Regular Season Problem 22
    "integrate(p**(12-3*x**2), x)",
    # Regular Season Problem 23
    "integrate(sec(x)**5*tan(x)**3, x)",
    # Regular Season Problem 24
    "integrate((1-sin(x))/x, (x, -pi/4, pi/4))",
    # Regular Season Problem 25
    "integrate(1/(x*sqrt(x**2-2)), x)",
    #
    # Next block:
    # Regular Season Problem 1 (new block; duplicate check)
    "integrate(1/sqrt((x-1)**2), x)",
    # Regular Season Problem 2 (new block; duplicate of x^(1/4)*log(x))
    "integrate(x**(1/4)*log(x), x)",
    # Regular Season Problem 3 (new block)
    "integrate(1/((1+sqrt(x))*sqrt(x-x**2)), x)",
    # Regular Season Problem 4 (new block)
    "integrate(1/sqrt(x*(sqrt(4*x)+1)**10), x)",
    # Regular Season Problem 5 (new block)
    "integrate(sin(acos(x)), x)",
    # Regular Season Problem 6 (new block)
    "integrate(1/sqrt(1-4*x-x**2), x)",
    # Regular Season Problem 7 (new block)
    "integrate(log(1/x), (x, 1/4, 1/2))",
    # Regular Season Problem 8 (new block)
    "integrate(1/(1+sin(x)), (x, 0, pi/2))",
    # Regular Season Problem 9 (new block)
    "integrate(sqrt(x)/(sqrt(2012-x)+sqrt(x)), (x, 1, 2011))",
    # Regular Season Problem 10 (new block)
    "integrate((x-1)/((x+1)*sqrt(x**3+x**2+x)), x)",
    # Regular Season Problem 11 (new block)
    "integrate((x**4+4*x**3+6*x**2+4*x+1)/(x**3-3*x**2+3*x-1), (x, -1, 0))",
    # Regular Season Problem 12 (new block)
    "integrate((cos(x)*log(x)+sin(x))/x, x)",
    # Regular Season Problem 13 (new block)
    "integrate(1/(x**3-x), x)",
    # Regular Season Problem 14 (new block)
    "integrate(x*asin(x)/sqrt(1-x**2), (x, 0, 1/2))",
    # Regular Season Problem 15 (new block)
    "integrate(x*(1-x)**99, (x, 0, 1))",
    # Regular Season Problem 16 (new block)
    "integrate(sin(4*x)/sin(x), (x, 0, pi/2))",
    # Regular Season Problem 17 (new block)
    "integrate((x-1)**2/(1+x**(1/3)), x)",
    # Regular Season Problem 18 (new block)
    "integrate(sqrt(2*x**2-1), x)",
    # Regular Season Problem 19 (new block)
    "integrate(sqrt(exp(x)-1), x)",
    # Regular Season Problem 20 (new block)
    "integrate(x/(x**4+4), x)",
    # Regular Season Problem 21 (new block)
    "integrate((cos(x)-sin(x))**2, (x, 0, 2))",
    # Regular Season Problem 22 (new block)
    "integrate(x*cosh(x)/sinh(x)**2, x)",
    # Regular Season Problem 23 (new block)
    "integrate(x**5/sqrt(1+x**3), (x, 0, 2))",
    # Regular Season Problem 24 (new block)
    "integrate((x**7-1)*log(x), (x, 0, 1))",
    # Regular Season Problem 25 (new block)
    "integrate(x-1, x)"
]
