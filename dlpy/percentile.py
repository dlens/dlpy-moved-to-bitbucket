'''
Calculations around percentiling.

'''
from dlpy.maths import  pw_linear, DecayType
from collections import OrderedDict
import numpy as np

def std_perc(X,s):
    '''
    Given a sequence of numbers X, calculate the percentile rank/score of the item s
    using the standard definition
    :param X:
    :param s:
    :return:
    '''
    e=0
    l=0
    for x in X:
        if x < s:
            l+=1
        elif x==s:
            e+=1
    return (l + 0.5*e)/len(X)


def gcp_sorted_deduped(X, s, epsilon=0.01, decay_type=DecayType.POWER, decay_rate=1,
                       do_inverse=False,
                       return_params=False):
    '''
    Calculates the grading compatible percentile score s in X.
    :param X: The array of numbers, already sorted, without duplicates
    :param s: The value to score
    :param epsilon: The epsilon, if it is too big we throw an error.
    :param decay_type: Power or Exponential decay?
    :param decay_rate: If Power decay, what power to use?
    :param return_params: If True we do not evaluate, instead we return a tuple like
    (pts, LHS, RHS) where pts are the points for the piecewise linear function, LHS
    is the parameters for the LHS decay, and similarly for RHS.
    :return:
    '''
    #First we setup our data points
    pts = []
    nMinus1 = len(X)-1
    pts.append((X[0], epsilon))
    for i in range(1,nMinus1):
        pts.append((X[i], i/nMinus1))
    pts.append((X[-1], 1-epsilon))
    if return_params:
        (LHS,RHS) = pw_linear(s,pts, 0, 1, decay_type=decay_type, decay_rate=decay_rate, return_params=True)
        return pts, LHS, RHS
    else:
        return pw_linear(s, pts, 0, 1, decay_type=decay_type, decay_rate=decay_rate)

def sort_dedupe(X):
    '''
    Sorts the elements of X and removes duplicates
    :param X:
    :return: New list with elements sorted and deduplicated
    '''
    deduped = list(OrderedDict.fromkeys(X))
    deduped.sort()
    return deduped

def gcp(X, s, epsilon=0.01, decay_type=DecayType.POWER, decay_rate=1, return_params=False):
    '''
    Very much like gcp_sorted_deduped, except it does not expect X to be deduped and sorted,
    we handle that.
    :param X:
    :param s:
    :param epsilon:
    :param decay_type: Power or Exponential decay?
    :param decay_rate: If Power decay, what power to use?
    :param return_params: If True we do not evaluate, instead we return a tuple like
    (pts, LHS, RHS) where pts are the points for the piecewise linear function, LHS
    is the parameters for the LHS decay, and similarly for RHS.
    :return:
    '''
    Y = sort_dedupe(X)
    return gcp_sorted_deduped(Y, s, epsilon, decay_type=decay_type,
                              decay_rate=decay_rate,
                              return_params=return_params)

def gcp_sorted_deduped_inverse(X, v, epsilon=0.01, decay_type=DecayType.POWER, decay_rate=1):
    '''
    Calculates the inverse of the grading compatible percentile score s in X.
    :param X: The array of numbers, already sorted, without duplicates
    :param v: The percentile to get the value of
    :param epsilon: The epsilon, if it is too big we throw an error.
    :param decay_type: Power or Exponential decay?
    :param decay_rate: If Power decay, what power to use?
    :return:
    '''
    #First we setup our data points
    inv_pts = []
    pts = []
    nMinus1 = len(X)-1
    inv_pts.append((epsilon, X[0]))
    pts.append((X[0], epsilon))
    for i in range(1,nMinus1):
        inv_pts.append((i/nMinus1, X[i]))
        pts.append((X[i], i / nMinus1))
    inv_pts.append((1-epsilon, X[-1]))
    pts.append((X[-1], 1-epsilon))
    LHS, RHS = pw_linear(v, pts, 0, 1, decay_type=decay_type, decay_rate=decay_rate, return_params=True)
    if v < epsilon:
        # We are in the LHS decay, let's see if we are less than 0
        if v <= 0:
            # We can never get to this percentile, fail
            raise Exception("Cannot have 0 or negative percentile")
        else:
            # We need to invert the decay function
            A,B,C=LHS
            if decay_type == DecayType.POWER:
                return (A/(v-C))**(1/decay_rate) + B
            elif decay_type == DecayType.EXPONENTIAL:
                return  np.log((v-C)/A)/B
            else:
                raise Exception("Unknown decay type")
    elif v > 1 - epsilon:
        # We are in RHS decay, see if it makes sense
        if v >= 1:
            raise Exception("Cannot have percentil 1 or greater")
        else:
            # We can invert the decay function
            A,B,C = RHS
            if decay_type == DecayType.POWER:
                return (A/(v-C))**(1/decay_rate) + B
            elif decay_type == DecayType.EXPONENTIAL:
                return  np.log((v-C)/A)/B
            else:
                raise Exception("Unknown decay type")
    else:
        # We are in between, we can just pw_linear it
        return pw_linear(v, inv_pts, 0, 1, decay_type=decay_type, decay_rate=decay_rate)


def gcp_inverse(X, v, epsilon=0.01, decay_type=DecayType.POWER, decay_rate=1, return_params=False):
    '''
    Very much like gcp_sorted_deduped, except it does not expect X to be deduped and sorted,
    we handle that.
    :param X:
    :param v: The percentile to get the inverse value of
    :param epsilon:
    :param decay_type: Power or Exponential decay?
    :param decay_rate: If Power decay, what power to use?
    :param return_params: If True we do not evaluate, instead we return a tuple like
    (pts, LHS, RHS) where pts are the points for the piecewise linear function, LHS
    is the parameters for the LHS decay, and similarly for RHS.
    :return:
    '''
    Y = sort_dedupe(X)
    return gcp_sorted_deduped_inverse(Y, v, epsilon, decay_type=decay_type,
                                      decay_rate=decay_rate)

def gcp_approx_pts(X, epsilon, percentiles=None, decay_type=DecayType.POWER, decay_rate=1):
    '''
    Gets the piecewise linear points to use to approximate the GCP function with the sequence X.
    :param X:
    :param epsilon:
    :param percentiles: If None we use [epsilon, 20, 40, 50, 60, 80, 1-epsilon]
    :return: The list of 2-tuples whose first value is the value that gives rise to percentiles[i]
    and whose 2nd value is perentile[i].  This could then be passed to the pw_linear function to evaluate
    this piecewise linear approximation.
    '''
    Y = sort_dedupe(X)
    if percentiles is None:
        percentiles = [epsilon, 0.2, 0.4, 0.5, 0.6, 0.8, 1-epsilon]
    pts = []
    for per in percentiles:
        pts.append((gcp_sorted_deduped_inverse(Y, per, epsilon, decay_type=decay_type, decay_rate=decay_rate), per))
    return pts

