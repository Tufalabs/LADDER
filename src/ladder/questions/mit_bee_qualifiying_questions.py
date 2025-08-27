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
    "integrate(x/(x**2024 - 1)/(x**506 - 1), x)",
    "integrate((5*x**3 - 3*x)**2, x)",
    "integrate((sin(x) + cos(x))**11, x)",
    "integrate((sinh(x) + cosh(x))**11, x)",
    "integrate(csc(x)**2 * tan(x)**2024, x)",
    "integrate(cos(x)*(log(cos(x)) - x*tan(x)), x)",
    "integrate(exp(-(x-2024)**2/4), x)",
    "integrate((1 - 1/x**2)*exp(x+1/x), x)",
    "integrate((x + 1 - exp(-x))*exp(x*exp(x)), x)",
    "integrate((arctan(x)/(1-x**2) + arctanh(x)/(1+x**2)), x)",
    # "integrate(sum(sin(k*pi/2)*x/factorial(k), (k, 0, oo)), x)",
    "integrate(sum(x**(2*n-1012), (n, 0, 2024)), x)",
    "integrate(x**4/(3 - 6*x + 6*x**2 - 4*x**3 + 2*x**4), x)",
    "integrate(x + ContinuedFraction([1, x, 1, x]), x)",
    "integrate(x*log(x), x)",
    "integrate(sech(x), x)",
    "integrate(exp(x)*(1 + exp(x))*log(1 + exp(x)), x)",
    "integrate((1 + x + x**2 + x**3 + x**4)*(1 - x + x**2 - x**3 + x**4), x)",
    "integrate(x**5, x)",
    "integrate(x + sin(x) + x*cos(x) + sin(x)*cos(x), x)",
    "integrate(sin(x)**2 + cos(x)**2 + tan(x)**2 + cot(x)**2 + sec(x)**2 + csc(x)**2, x)",
    "integrate(floor(2023*sin(x)), x)",
    "integrate((2*log(x) + 1)*exp(log(x)**2), x)",
    "integrate((1 - x)**3 + (x - x**2)**3 + (x**2 - 1)**3 - 3*(1 - x)*(x - x**2)*(x**2 - 1), x)",
    "integrate(Abs(Abs(Abs(Abs(x) - 1) - 1) - 1), x)",
    "integrate(sin(x)**6 + cos(x)**6 + 3*sin(x)**2*cos(x)**2, x)",
    "integrate((x + E + 1)**x*exp(x), x)",
    "integrate(sqrt(2*x/(x + 1))/(2 - x**2), x)",
    "integrate((1 + 2*x**2022)/(x + x**2023), x)",
    "integrate(3*sin(20*x)*cos(23*x) + 20*sin(43*x), x)",
    # "integrate(Product(1/(1 + x**(2**k)), (k, 0, oo)), x)",
    "integrate(sin(x)/(2*exp(x) + cos(x) + sin(x)), x)",
    "integrate(log(x/pi)/(log(x)**log(pi*E)), x)",
    "integrate(x**5 + 5*x**4 + 10*x**3 + 8*x**2 + x, x)",
    "integrate((1 + cos(x))/(x + sin(x)), x)",
    "integrate((arctan(x) + arccot(x))/x, x)",
    "integrate(x**2 - floor(x)*ceiling(x), x)",
    "integrate(sinh(x)/(cosh(x) - sinh(x)), x)",
    "integrate(x/(sqrt(x - 1) + sqrt(x + 1)), x)",
    "integrate(cos(x + cos(x)), x)",
    "integrate(x**3*sin(x**2), x)",
    "integrate(x/(1 - x**4), x)",
    "integrate(1/cosh(x)**2, x)",
    "integrate(exp(exp(x)) - exp(exp(x) - x), x)",
    "integrate(Limit(sin(pi/3)*sin(pi/3)*sin(pi/3)*x, n, oo), x)",
    "integrate(sqrt(1 - sqrt(x)), x)",
    "integrate(x**3/(1 + x + x**2/2 + x**3/6), x)",
    "integrate(sin(x + sin(x)) - sin(x - sin(x)), x)",
    "integrate(tan(x)**4*sec(x)**3 + tan(x)**2*sec(x)**5, x)",
    "integrate((1 + log(x))*log(log(x)), x)",
    "integrate(1/(1 + sin(x)) + 1/(1 + cos(x)) + 1/(1 + tan(x)) + 1/(1 + cot(x)) + 1/(1 + sec(x)) + 1/(1 + csc(x)), x)",
    "integrate(1/sqrt(x - x**2), x)",
    "integrate(sum(binomial(n + 3, n)*x**n, (n, 0, oo)), x)",
    "integrate(1/(1 + cos(x)**2), x)",
    "integrate(log(2*x)/(x*log(x)), x)",
    "integrate(1/(exp(x) + 1), x)",
    "integrate(log(x)*log(log(x))/x, x)",
    "integrate(log((1 + x)/(1 - x)), x)",
    "integrate(1/(x**2 + (x - 1)**2), x)",
    "integrate(sqrt(x*sqrt(x*sqrt(x))), x)",
    "integrate(sin(x)**4*cos(x)**4*(cos(x) + sin(x))*(cos(x) - sin(x)), x)",
    "integrate(log(x**2 + 1), x)",
    "integrate(cos(x)**(2020), x)",
    "integrate((2*x + 1)/(2*x**2 + 2*x + 1), x)",
    "integrate(arcsin(x)/x**3, x)",
    "integrate(sin(2*x)*cos(cos(x)), x)",
    "integrate(sin(sin(x) - x), x)",
    "integrate(1/(x - 1) + sum((k + 1)*x**k, (k, 0, 2018))/sum(x**k, (k, 0, 2019)), x)",
    "integrate(1/(tan(x)**(sqrt(2020)) + 1), x)",
    "integrate(x*(1 - x)**2020, x)",
    "integrate(sec(x)**4*tan(x)/(sec(x)**4 + 4), x)",
    "integrate(x**2*x/(2*log(x) + 2), x)",
    "integrate(sqrt(1 - x**2), x)",
    "integrate(x**5*exp(-x**4), x)",
    "integrate(tan(cos(x)), x)",
    "integrate((x + 1)/(x*(x + log(x))), x)",
    "integrate(exp(x+exp(x)) + exp(x-exp(x)), x)",
    "integrate(1/(1 - x**2), x)",
    "integrate(2/log(x), x)",
    "integrate((cos(3*x) + sin(2*x))*(-sin(2019*x) + cos(3*x)), x)",
    "integrate(cos(x)*cos(sin(x))*cos(sin(sin(x))), x)",
    "integrate(exp(-2019/(4*t**2))*t**2, x)",
    "integrate(sin(sqrt(x)), x)",
    "integrate(sqrt(x)/(1 + x), x)",
    "integrate(cos(x)*cos(2*x)*cos(3*x), x)",
    "integrate(Limit(exp(-x**(2*n)), n, oo), x)",
    "integrate(x/(1*log(x)), x)",
    "integrate((sin(20*x) + sin(19*x))/(cos(20*x) + cos(19*x)), x)",
    "integrate(exp(x)*(cos(x)**2 + cos(x)*sin(x) - sin(x)**2), x)",
    "integrate(sin(x)/sin(x + pi/4), x)",
    "integrate(1/(x + sqrt(3)*x), x)",
    "integrate(x/(x**2 + 1)*(2*log(x) + 1), x)",
    "integrate((2*x**3 - 1)/(x*(x**3 + 1)), x)",
    "integrate(cos(arctan(x)), x)",
  
    # 1.
    "integrate(tan(cos(x)), (x, 0, 2*pi))",
    # 2. (no limits detected; the lone extra line is dropped)
    "integrate(x*(x + log(x)), x)",
    # 3.
    "integrate(exp(x+e) + exp(x-e), x)",
    # 4.
    "integrate(1 - x**2, (x, -1/2, 1/2))",
    # 5.
    "integrate(2*log(x), (x, 0, 2))",
    # 6.
    "integrate((cos(3*x) + sin(2*x)) * (- sin(2019*x) + cos(3*x)), (x, -2*pi, 2*pi))",
    # 7.
    "integrate(cos(x)*cos(sin(x))*cos(sin(sin(x))), x)",
    # 8.
    "integrate(exp(-2019/(4*t**2))*t**2, (t, 0, oo))",
    # 9.
    "integrate(sin(sqrt(x)), x)",
    # 10.
    "integrate(sqrt(x)/(1+x), (x, 0, 1))",
    # 11.
    "integrate(cos(x)*cos(2*x)*cos(3*x), (x, 0, 2*pi))",
    # 12.
    "Limit(integrate(exp(-x**(2*n)), (x, -oo, oo)), n, oo)",
    # 13.
    "integrate(x*log(x), (x, 0, E))",
    # 14.
    "integrate((sin(20*x)+sin(19*x))/(cos(20*x)+cos(19*x)), (x, 0, pi/100))",
    # 15.
    "integrate(exp(x)*cos(x)**2 + exp(x)*cos(x)*sin(x) - exp(x)*sin(x)**2, x)",
    # 16.
    "integrate(sin(x)/sin(x+pi/4), (x, 0, pi/2))",
    # 17.
    "integrate((1+sqrt(3))*x, x)",
    # 18.
    "integrate(x/(x**2+1)*(2*log(x)+1), (x, 0, 2))",
    # 19.
    "integrate((2*x**3 - 1)/(x*(x**3+1)), x)",
    # 20.
    "integrate(cos(arctan(x)), x)",
    
    # 21.
    "integrate(exp(x) + exp(x+2), x)",
    # 22.
    "integrate(s*x**3 * r*x**4 * q*x*sqrt(5)*x, x)",  # Ambiguous; placeholder interpretation.
    # 23.
    "integrate(Abs(sin(2018*x)), (x, 0, 2018*pi))",
    # 24.
    "integrate(tan(x)+cot(x), x)",
    # 25.
    "integrate(x**5/(2+x**12), x)",
    # 26.
    "integrate(cos(x)*cosh(x) + sin(x)*sinh(x), x)",
    # 27.
    "integrate(exp(x) + cos(x)*exp(x) + sin(x)*exp(x), x)",
    # 28.
    "integrate(sin(cos(sin(x)))*sin(sin(x))*cos(x), x)",
    # 29.
    "integrate(1+sin(x), x)",
    # 30.
    "integrate(cos(x)/(1-cos(2*x)), x)",
    # 31.
    "integrate(exp(x)*(1/x+log(x)), x)",
    # 32.
    "integrate(tanh(x)**2, x)",
    # 33.
    "integrate((2017*x**2016 + 2018*x**2017)/(4034 + 2*x + 4035 + x + 4036), x)",  # Very ambiguous.
    # 34.
    "integrate((sin(2*x)-sin(2*x))/(cos(2*x)-cos(2*x), x)",  # This evaluates to 0; unclear original.
    # 35.
    "integrate(x**25/(25*x**16 + 25 + x**9/25), x)",  # Ambiguous.
    # 36.
    "integrate(cos(x)/(2-cos(x)**2), (x, 0, pi/2))",
    # 37.
    "integrate((1+x**2)**(3/2), x)",
    # 38.
    "integrate(p*x*sqrt(x) - x**2, x)",  # 'p' is ambiguous.
    # 39.
    "integrate((x-1)/(x+x**2*log(x)), x)",  # Ambiguous.
    # 40.
    "integrate(csc(x)*sec(x), x)",
    
    # 41.
    "integrate(x**2/sqrt(x**3+2), x)",
    # 42.
    "integrate(log(x)/x**2, (x, 1, oo))",
    # 43.
    "integrate(sech(x), x)",
    # 44.
    "integrate(x**3/(e*x**2), x)",  # Simplifies to integrate(x/e, x).
    # 45.
    "integrate(1/(x*sqrt(x**2-1)), (x, 1, 2))",
    # 46.
    "integrate(1/(x*(x**2+1)), (x, 1, oo))",
    # 47.
    "integrate(acosh(x), x)",
    # 48.
    "integrate(exp(-2*x**2-5*x-3), (x, -oo, oo))",
    # 49.
    "integrate(sin(sqrt(x)), x)",
    # 50.
    "integrate(1/(x+1/x)**2, (x, 0, oo))",
    
    # 51.
    "integrate((2+x)*exp(-x)/x**3, x)",
    # 52.
    "integrate(p*x*(1-x), x)",  # 'p' is ambiguous.
    # 53.
    "integrate(tanh(x)*exp(x), x)",
    # 54.
    "integrate(p*sin(x)+1, (x, 0, pi/2))",  # 'p' again is ambiguous.
    # 55.
    "Limit(In, n, oo)",  # where In is defined by a series of integrals.
    # 56.
    "integrate(sin(x+pi/4)**2*exp(x**2), (x, -oo, oo))",
    # 57.
    "integrate(3*x**2/(x**3+1)**2 * exp(-x**(6-2*x**3)), (x, -oo, oo))",  # Ambiguous exponent.
    # 58.
    "integrate(1/(1+tan(x)**2017), (x, 0, pi/2))",
    # 59.
    "integrate(exp(2*x)*cos(3*x), x)",
    # 60.
    "integrate(cos(x)*(cos(x)+1)*tan(x)*(1+log(cos(x))), x)",  # Interpretation of the messy term.
    
    # 61.
    "integrate(tanh(x), x)",
    # 62.
    "integrate(Abs(x**3 - x), (x, -4, 4))",
    # 63.
    "integrate(log(sqrt(x)), x)",
    # 64.
    "integrate(exp(x+e) - exp(x-e), x)",
    # 65.
    "integrate(log(log(x))/(x*log(x)), x)",
    # 66.
    "integrate(1/(1+tan(x)**2), (x, 0, pi/3))",
    # 67.
    "integrate(arcsin(3*x**(1/3)), x)",
    # 68.
    "integrate(blog2(x)*xc, x)",  # Ambiguous: 'blog2 xc'
    # 69.
    "integrate(exp(x)*cos(x) - exp(x)*sin(x), x)",
    # 70.
    "integrate(x**3*exp(-x**2), (x, 0, oo))",
    # 71.
    "integrate((2*exp(x**2)/(x+1))*cos(x) - (exp(x**2)+x)*sin(x), x)",
    # 72.
    "integrate((1+sqrt(x)+x**(1/3))*(1+x**(-1/2)+x**(-1/3)), x)",
    # 73.
    "integrate(sin(sin(x))*cos(sin(x))*cos(x), x)",
    # 74.
    "integrate((cos(x)+sin(x))/x**2 + (sin(x)-cos(x))/x, x)",
    # 75.
    "integrate(x**3/sqrt(x**2+1), x)",
    # 76.
    "integrate(x/(x**4+x**2+1), x)",
    # 77.
    "integrate(exp(2016*x+6048*x), x)",
    # 78.
    "integrate((1-cos(x))/sin(x), (x, pi/3, pi/2))",
    # 79.
    "integrate(1/(1-x+x**2-x**3), x)",
    # 80.
    "integrate(1/(2+cosh(x)), (x, 0, oo))",
    # Block 1:
    "integrate(cos(x)**4 - sin(x)**4, x)",
    "integrate(x/sqrt(2+4*x), x)",
    "integrate(cos(sqrt(x))/sqrt(x), (x, 0, 8))",
    "integrate(sec(x), x)",
    "integrate(exp(sin(x))*tan(x)*csc(x), (x, 0, pi/2))",
    "integrate(x*(log(x))**2, (x, 1, e))",
    "integrate(1/(5 + 4*sqrt(x) + x), x)",
    "integrate(x**2015, x)",
    "integrate(x/((x-3)*(x+5)**2), (x, 0, 2))",
    "integrate(log(1+log(x))/x, x)",
    "integrate(sqrt(csc(x) - sin(x)), x)",
    "integrate(1/sqrt(x**2+25), x)",
    "integrate((log(x,2) - 1/(x*log(x,2))), (x, 2, e))",
    "integrate(arctan(exp(x))/exp(3*x), x)",
    "integrate(Abs(x-1)/(Abs(x-2)+Abs(x-3)), (x, 0, 4))",
    "integrate(1/(sin(x)**4+cos(x)**4), (x, 0, 2*pi))",
    "integrate((1+exp(x))/(1-exp(x)), x)",
    "integrate(tan(x)**4, x)",
    "integrate(sin(x)*tan(x)**2, x)",
    "integrate((x+1)/(x**2+2*x+3), x)",

    # Block 2:
    "integrate(log(x**2), (x, 1, e))",
    "integrate(sin(sqrt(3)*x), (x, -9, 9))",
    "integrate(diff(h(exp(1+x-x**2)), x), (x, 0, oo))",
    "integrate(r*x + q*x + sqrt(x), (x, 0, 2))",
    "integrate(sqrt(x)*exp(sqrt(x)), x)",
    "integrate(sin(2*x)*cos(3*x), x)",
    "integrate(Abs(1+2*sin(x)), (x, 0, 2*pi))",
    "integrate(x*(1-x)**2014, x)",
    "integrate(arcsinh(x), x)",
    "integrate(x**2/(x-1), (x, -1, 0))",
    "integrate(x*arctan(x), x)",
    "integrate(1/(x**2 - 15*x - 2014), x)",
    "integrate(exp(x)*log(1+x**2) - 2*(1+x)*arctan(x), x)",
    "integrate((arcsin(x))**2, x)",
    "integrate(sqrt(x**2-1)/x, x)",
    "integrate(x*sec(4*x)**2, x)",
    "integrate(6 - 11*x + 6*x**2 - x**3, x)",
    "integrate(1/(b1 - log(1-x,2)), (x, 0, 1))",
    "integrate(q*x + p*x**2 + 1, (x, 0, 1/sqrt(3)))",
    "integrate(2+cos(x), (x, 0, 5*pi/2))",

    # Block 3:
    "integrate(log(x**2) - 2*log(2*x), x)",
    "integrate(exp(Abs(x)), (x, -1, 3))",
    "integrate((log(x)*cos(x) - sin(x)/x)/(log(x)**2), x)",
    "integrate(x**3 - 3*x**2 + 3*x - 1, (x, 1, 11))",
    "integrate(12 - 3*x**2, (x, 0, 2))",
    "integrate(x + (x-3)/(7+sin(x-3)), (x, 0, 6))",
    "integrate(sin(x)/(1+tan(x)**2), x)",
    "integrate((x**5 - x**3 + x**2 - 1)/(x**4 - x**3 + x - 1), x)",
    "integrate(log(x), (x, 0, 1))",
    "integrate(1 - exp(-x), x)",
    "integrate(sin(x)**2*cos(x)**2, (x, 0, pi))",
    "integrate(pi*sin(pi*sqrt(x))/sqrt(x), (x, 0, 441))",
    "integrate(tan(x)**2, x)",
    "integrate((x - b)**2, (x, 0, 256))",
    "integrate(exp(2*sqrt(x)), x)",
    "integrate(cos(x)*cot(x), x)",
    "integrate(2*log(x) + log(x)**2, x)",
    "integrate(x/(1+x**2), x)",
    "integrate(1/(2-2*x+x**2), x)",
    "integrate(sin(x)*log(sin(x)), x)",
    "integrate(x/(1-x**4), x)",
    "integrate(sec(x)**5*tan(x)**3, x)",
    "integrate(1/(1-sin(x)), (x, -pi/4, pi/4))",
    "integrate(x/sqrt(x**2-2), x)",

    # Block 4:
    "integrate(1/(x-1), x)",
    "integrate(x**(1/4)*log(x), x)",
    "integrate((1+sqrt(x))/(sqrt(x)-x**2), x)",
    "integrate(1/sqrt(x*(sqrt(4*x)+1)**10), x)",
    "integrate(sin(acos(x)), (x, 0, 1))",
    "integrate(1/sqrt(1-4*x-x**2), x)",
    "integrate(log(1/x), (x, 1/4, 1/2))",
    "integrate(1/(1+sin(x)), (x, 0, pi/2))",
    "integrate(sqrt(x)/(sqrt(2012-x)+sqrt(x)), (x, 1, 2011))",
    "integrate((x-1)/((x+1)*sqrt(x**3+x**2+x)), x)",
    "integrate((x**4+4*x**3+6*x**2+4*x+1)/(x**3-3*x**2+3*x-1), (x, -1, 0))",
    "integrate((cos(x)*log(x)+sin(x))/x, x)",
    "integrate(1/(x**3-x), x)",
    "integrate(x*asin(x)/sqrt(1-x**2), (x, 0, 1/2))",
    "integrate(x*(1-x)**99, (x, 0, 1))",
    "integrate(sin(4*x)/sin(x), (x, 0, pi/2))",
    "integrate((x-1)**2/(1+x**(1/3)), x)",
    "integrate(sqrt(2*x**2-1), x)",
    "integrate(sqrt(exp(x)-1), x)",
    "integrate(x/(x**4+4), x)",
    "integrate((cos(x)-sin(x))**2, (x, 0, 2))",
    "integrate(x*cosh(x)/sinh(x)**2, x)",
    "integrate(x**5/sqrt(1+x**3), (x, 0, 2))",
    "integrate((x**7-1)*log(x), (x, 0, 1))",
    "integrate(x-1, x)",

    # Block 5:
    "integrate(log(x**2) - 2*log(2*x), x)",
    "integrate(exp(Abs(x)), (x, -1, 3))",
    "integrate((log(x)*cos(x)-sin(x)/x)/(log(x)**2), x)",
    "integrate(x**3-3*x**2+3*x-1, (x, 1, 11))",
    "integrate(12-3*x**2, (x, 0, 2))",
    "integrate(x+(x-3)/(7+sin(x-3)), (x, 0, 6))",
    "integrate(sin(x)/(1+tan(x)**2), x)",
    "integrate((x**5-x**3+x**2-1)/(x**4-x**3+x-1), x)",
    "integrate(log(x), (x, 0, 1))",
    "integrate(1-exp(-x), x)",
    "integrate(sin(x)**2*cos(x)**2, (x, 0, pi))",
    "integrate(pi*sin(pi*sqrt(x))/sqrt(x), (x, 0, 441))",
    "integrate(tan(x)**2, x)",
    "integrate((x-b)**2, (x, 0, 256))",
    "integrate(exp(2*sqrt(x)), x)",
    "integrate(cos(x)*cot(x), x)",
    "integrate(2*log(x)+log(x)**2, x)",
    "integrate(x/(1+x**2), x)",
    "integrate(1/(2-2*x+x**2), x)",
    "integrate(sin(x)*log(sin(x)), x)",
    "integrate(x/(1-x**4), x)",
    "integrate(sec(x)**5*tan(x)**3, x)",
    "integrate(1/(1-sin(x)), (x, -pi/4, pi/4))",
    "integrate(x/sqrt(x**2-2), x)",

    # Block 6:
    "integrate(1/(x-1), x)",
    "integrate(x**(1/4)*log(x), x)",
    "integrate((1+sqrt(x))/(sqrt(x)-x**2), x)",
    "integrate(1/sqrt(x*(sqrt(4*x)+1)**10), x)",
    "integrate(sin(acos(x)), (x, 0, 1))",
    "integrate(1/sqrt(1-4*x-x**2), x)",
    "integrate(log(1/x), (x, 1/4, 1/2))",
    "integrate(1/(1+sin(x)), (x, 0, pi/2))",
    "integrate(sqrt(x)/(sqrt(2012-x)+sqrt(x)), (x, 1, 2011))",
    "integrate((x-1)/((x+1)*sqrt(x**3+x**2+x)), x)",
    "integrate((x**4+4*x**3+6*x**2+4*x+1)/(x**3-3*x**2+3*x-1), (x, -1, 0))",
    "integrate((cos(x)*log(x)+sin(x))/x, x)",
    "integrate(1/(x**3-x), x)",
    "integrate(x*asin(x)/sqrt(1-x**2), (x, 0, 1/2))",
    "integrate(x*(1-x)**99, (x, 0, 1))",
    "integrate(sin(4*x)/sin(x), (x, 0, pi/2))",
    "integrate((x-1)**2/(1+x**(1/3)), x)",
    "integrate(sqrt(2*x**2-1), x)",
    "integrate(sqrt(exp(x)-1), x)",
    "integrate(x/(x**4+4), x)",
    "integrate((cos(x)-sin(x))**2, (x, 0, 2))",
    "integrate(x*cosh(x)/sinh(x)**2, x)",
    "integrate(x**5/sqrt(1+x**3), (x, 0, 2))",
    "integrate((x**7-1)*log(x), (x, 0, 1))",
    "integrate(x-1, x)",

    # Block 7:
    "integrate(log(x**2)-2*log(2*x), x)",
    "integrate(exp(Abs(x)), (x, -1, 3))",
    "integrate((log(x)*cos(x)-sin(x)/x)/(log(x)**2), x)",
    "integrate(x**3-3*x**2+3*x-1, (x, 1, 11))",
    "integrate(12-3*x**2, (x, 0, 2))",
    "integrate(x+(x-3)/(7+sin(x-3)), (x, 0, 6))",
    "integrate(sin(x)/(1+tan(x)**2), x)",
    "integrate((x**5-x**3+x**2-1)/(x**4-x**3+x-1), x)",
    "integrate(log(x), (x, 0, 1))",
    "integrate(1-exp(-x), x)",
    "integrate(sin(x)**2*cos(x)**2, (x, 0, pi))",
    "integrate(pi*sin(pi*sqrt(x))/sqrt(x), (x, 0, 441))",
    "integrate(tan(x)**2, x)",
    "integrate((x-b)**2, (x, 0, 256))",
    "integrate(exp(2*sqrt(x)), x)",
    "integrate(cos(x)*cot(x), x)",
    "integrate(2*log(x)+log(x)**2, x)",
    "integrate(x/(1+x**2), x)",
    "integrate(1/(2-2*x+x**2), x)",
    "integrate(sin(x)*log(sin(x)), x)",
    "integrate(x/(1-x**4), x)",
    "integrate(sec(x)**5*tan(x)**3, x)",
    "integrate(1/(1-sin(x)), (x, -pi/4, pi/4))",
    "integrate(x/sqrt(x**2-2), x)",
]
