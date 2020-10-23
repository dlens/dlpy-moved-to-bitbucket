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

def decay_exponential(x, x0, y0, m, C, return_params=False):
    '''
    Creates a
    :param x:
    :param x0:
    :param y0:
    :param m:
    :param C:
    :param return_params: If true we don't calculate, we only return a tuple of (A,B,C)
    the parameters
    :return:
    '''
    if m==0:
        raise Exception("Cannot have zero slope for decay")
    elif y0 == C:
        # There is no decay to do, just send back B
        return C
    else:
        B = m/(y0 - C)
        A = (y0 - C) / np.exp(B * x0)
        if return_params:
            return (A,B,C)
        else:
            return A * np.exp(B*x) + C

def decay_linear(x, x0, y0, m, C, k=1, return_params=False):
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
    :param C: the asymptotic value
    :param k: the power to raise the denominator to
    :param return_params: If true we don't calculate, we only return a tuple of (A,B,C)
    the parameters
    :return:
    '''
    if m == 0:
        B = x0-1
    else:
        B = x0 + (y0 - C) * k / m
    A = (y0 - C) * (x0 - B) ** k
    if return_params:
        return (A,B,C)
    else:
        return A / (x-B) + C


def slope(pt1, pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])


class DecayType(Enum):
    POWER= 1,
    EXPONENTIAL=2



def pw_linear(x, pts, lhs_asymptote, rhs_asymptote, decay_type=DecayType.POWER, decay_rate=1, return_params=False):
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
    :param return_params: If True, we do not actually calculate, instead we calculate the decay rate parameters
    and return them in a tuple of 2 elements.  The first element is the LHS decay params, the second is the
    RHS decay params.
    :return: The value
    '''
    if len(pts) <= 0:
        raise Exception("No points to linearly interpolate between")
    if return_params:
        # We only want the decay params associated with this
        LHS=None
        RHS=None
        if len(pts)==1:
            m0=1
            m1=1
        else:
            m0=slope(pts[1], pts[0])
            m1 = slope(pts[-1], pts[-2])

        if decay_type == DecayType.POWER:
            LHS = decay_linear(x, pts[0][0], pts[0][1], m0, lhs_asymptote, decay_rate, return_params=True)
            RHS = decay_linear(x, pts[-1][0], pts[-1][1], m0, rhs_asymptote, decay_rate, return_params=True)
        elif decay_type == DecayType.EXPONENTIAL:
            LHS = decay_exponential(x, pts[0][0], pts[0][1], m1, lhs_asymptote, return_params=True)
            RHS = decay_exponential(x, pts[-1][0], pts[-1][1], m1, rhs_asymptote, return_params=True)
        return (LHS,RHS)
    if x < pts[0][0]:
        if len(pts)==1:
            m=1
        else:
            m=slope(pts[1], pts[0])
        if decay_type == DecayType.POWER:
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
        if decay_type == DecayType.POWER:
            return decay_linear(x, pts[-1][0], pts[-1][1], m, rhs_asymptote, decay_rate)
        elif decay_type == DecayType.EXPONENTIAL:
            return decay_exponential(x, pts[-1][0], pts[-1][1], m, rhs_asymptote)
    #Okay we have made it to the regular linear interpolation phase
    for i in range(1, len(pts)):
        if pts[i][0] >= x:
            return linear_interp(x, pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
    # We should never make it here
    raise Exception("Should not make it here")
