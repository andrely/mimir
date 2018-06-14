from math import isnan

import sys

GOLD = 1.618034
GLIMIT = 100
TINY = 1e-20
R = .61803399
TOL = 3.0e-8
C = 1. - R

def nan_guard(x):
    if isnan(x):
        return sys.float_info.max
    else:
        return x


def sign(a, b):
    if b == 0.:
        return a

    return a * b / abs(b)


def bracket(a, b, func):
    ax = a
    bx = b
    fa = func(a)
    fb = func(b)

    if fb > fa:
        ax, bx = bx, ax
        fb, fa = fa, fb

    cx = bx + GOLD*(bx - ax)
    fc = func(cx)

    while fb > fc:
        r = (bx - ax)*(fb - fc)
        q = (bx - cx)*(fb - fa)
        u = bx - ((bx - cx)*q - (bx - ax)*r) / 2.0*(sign(max(abs(q - r), TINY), q - r))
        ulim = bx + GLIMIT*(cx - bx)

        if (bx - u)*(u - cx) > 0.:
            fu = func(u)

            if fu < fc:
                ax = bx
                bx = u
                fa = fb
                fb = fu

                return ax, bx, cx, fa, fb, fc
            elif fu > fb:
                cx = u
                fc = fu

                return ax, bx, cx, fa, fb, fc

            u = cx + GOLD*(cx - bx)
            fu = func(u)
        elif ((cx - u )*(u - ulim)) > 0.:
            fu = func(u)

            if fu < fc:
                bx, cx, u = cx, u, u + GOLD*(u - cx)
                fb, fc, fu = fc, fu, func(u)
        elif ((u - ulim)*(ulim - cx)) > 0.:
            u = ulim
            fu = func(u)
        else:
            u = cx + GOLD*(cx - bx)
            fu = func(u)

        ax, bx, cx = bx, cx, u
        fa, fb, fc = fb, fc, fu

    if ax > cx:
        return cx, bx, ax, fc, fb, fa
    else:
        return ax, bx, cx, fa, fb, fc


def golden_section(bracketing, func):
    ax, bx, cx, fa, fb, fc = bracketing
    iter = 1

    x0 = ax
    x3 = cx

    if abs(cx - bx) > abs(bx - ax):
        x1 = bx
        x2 = bx - C*(cx - bx)
    else:
        x2 = bx
        x1 = bx - C*(bx - ax)

    f1 = func(x1)
    f2 = func(x2)

    while abs(x3 - x0) > TOL*abs(x2 - x1) and iter < 50:
        iter += 1

        if f2 < f1:
            x0, x1, x2 = x1, x2, R*x2 + C*x3
            f1, f2 = f2, func(x2)
        else:
            x3, x2, x1 = x2, x1, R*x1 + C*x0
            f2, f1 = f1, func(x1)

    if f1 < f2:
        xmin = x1
        fmin = f1
    else:
        xmin = x2
        fmin = f2

    return xmin, fmin


def line_search(x, y, func):
    inner_func = lambda x: nan_guard(func(x))

    return golden_section(bracket(x, y, inner_func), inner_func)
