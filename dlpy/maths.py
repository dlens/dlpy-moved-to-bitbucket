'''
Some common mathematical functions
'''

from enum import Enum
import numpy as np

def linear_interp(x, x1, y1, x2, y2):
    '''
    Linearly interpolates x between (x1,y1) and (x2,y2).  Note this function
    does not care if x1 <= x <= x2.
    :param x:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: The interpolated value, unless x1==x2, in which case we assume the
    line is y = (y1+y2)/2, i.e. the constant function of the average of the y
    values.
    '''
    if x1 == x2:
        return (y1+y2)/2
    else:
        m = (y2-y1)/(x2-x1)
        return y1 + m*(x-x1)


class Decayer(Enum):
    EXP=1,
    LINEAR=2

def decay_exponential(x, x0, y0, m, B):
    '''
    Creates a
    :param x:
    :param x0:
    :param y0:
    :param m:
    :param B:
    :return:
    '''
    if m==0:
        raise Exception("Cannot have zero slope for decay")
    elif y0 == B:
        # There is no decay to do, just send back B
        return B
    else:
        c = m/(y0-B)
        A = (y0-B)/np.exp(c*x0)
        return A*np.exp(c*x)+B

def decay_linear(x, x0, y0, m, B, k=1):
    '''
    Creates a funciton of the form f(x)=A/(x-c)^k+B where
    f(x0)=y0
    and
    f'(x0)=m
    and then evaluates f(x) for the given value x.
    :param x: the value to evaluate our function at.
    :param x0: the x coord of the point whose value we know
    :param y0: the y coord of the point whose value we know
    :param m: the slope at the point x0
    :param B: the asymptotic value
    :param k: the power to raise the denominator to
    :return:
    '''
    if m == 0:
        c = x0-1
    else:
        c = x0 + (y0-B)*k/m
    A = (y0-B)*(x0-c)**k
    return A/(x-c) + B


def slope(pt1, pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])


class DecayType(Enum):
    LINEAR=1,
    EXPONENTIAL=2



def pw_linear(x, pts, lhs_asymptote, rhs_asymptote, decay_type=DecayType.LINEAR, decay_rate=1):
    '''
    Does piecewise linear function definition between the pts=((x1,y1), (x2, y2), ..., (x_n, y_n)) where
    this function assumes x1 < x2 < .....  If you input x < x1 then it decays from y1 to lhs_asymptote.
    If you input x > x_n it decays from y_n to rhs_asymptote.
    :param x:
    :param pts:  A list of the form (x1, y1), ..., (x_n, y_n) where x1<x2<...<x_n
    :param lhs_asymptote:
    :param rhs_asymptote:
    :param decay_type: Linear or Exponential decay?
    :param decay_rate: If linear, what power to use on the denominator (defaults to linear denom)
    :return: The value
    '''
    if len(pts) <= 0:
        raise Exception("No points to linearly interpolate between")
    if x < pts[0][0]:
        if len(pts)==1:
            m=1
        else:
            m=slope(pts[1], pts[0])
        if decay_type == DecayType.LINEAR:
            return decay_linear(x, pts[0][0], pts[0][1], m, lhs_asymptote, decay_rate)
        elif decay_type == DecayType.EXPONENTIAL:
            return decay_exponential(x, pts[0][0], pts[0][1], m, lhs_asymptote)
        else:
            raise Exception("Unknown decay type")
    elif x > pts[-1][0]:
        if len(pts) == 1:
            m=1
        else:
            m=slope(pts[-1], pts[-2])
        if decay_type == DecayType.LINEAR:
            return decay_linear(x, pts[-1][0], pts[-1][1], m, rhs_asymptote, decay_rate)
        elif decay_type == DecayType.EXPONENTIAL:
            return decay_exponential(x, pts[-1][0], pts[-1][1], m, rhs_asymptote)
    #Okay we have made it to the regular linear interpolation phase
    for i in range(1, len(pts)):
        if pts[i][0] >= x:
            return linear_interp(x, pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
    # We should never make it here
    raise Exception("Should not make it here")

