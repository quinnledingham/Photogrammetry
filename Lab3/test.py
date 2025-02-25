from sympy import *

x = symbols('x')
expr = cos(x)
expr = diff(expr, x)
expr = expr.subs(x, 1)
print(expr.evalf())