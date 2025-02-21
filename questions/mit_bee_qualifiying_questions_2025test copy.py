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
    "integrate(x + (x**0.5)/(1 + (x**0.5)), x)",
    
    "integrate(exp(x + 1)/(exp(x) + 1), x)",
    
    "integrate((3*sin(x) - sin(3*x))**(1/3), x)",
    
    "integrate(log(x**(log(x**x)))/x**2, x",
    
    "integrate(cos(20*x)*sin(25*x), x, -pi/2, pi/2)",
    
    "integrate(sin(x)*cos(x)*tan(x)*cot(x)*sec(x)*csc(x), x, 0, 2*pi)",
    
    "integrate((x*log(x)*cos(x) - sin(x))/(x*log(x)**2), x)",
    
    "integrate((2*x - 1 + log(2*x)), x, 1, 2)",
    
    "integrate(x**2024*(1 - x**2025)**2025, x, 0, 1)",
    
    "integrate((x - 1/2)*(x - 1)*x, x, 0, 10)",
    
    "integrate(floor(x)/2, x, 0, 20)",
    
    "integrate((exp(2*x)*(x**2 + x))/(x*exp(x)*4 + 1), x)",
    
    "integrate(sec(x)**4 - tan(x)**4, x)",
    
    "integrate(sqrt(x*(1 - x)), x, 0, 1)",
    
    "integrate(sin(4*x)*cos(x)/(cos(2*x)*sin(x)), x)",
    
    "integrate(sin(x)*sinh(x), x)",
    
    "integrate(sin(x)*cos(pi/3 - x), x, 0, pi/3)",
    
    "integrate((cos(x) + cos(x + 2*pi/3) + cos(x - 2*pi/3))**2, x)",
    
    "integrate(sum((-1)**k*x**(2*k)), x, 0, 1)",
    # Add more integrals as needed
]
