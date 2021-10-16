import random

def calc_fx(a = 0, b = 0, c = 0, x = 0.0):
    return a*(x**2) + b*x + c

derivative_X = ( calc_fx(a, b, c, X + h) - calc_fx(a, b, c, X - h) ) / (2 * h)