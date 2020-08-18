import numpy as np


def wtd_mean(values, weights):
    '''
    Calculates the weighted mean of a set of values
    :param values: The values to take the weighted mean of
    :param weights: The weights
    :return: 
    '''
    values = np.array(values)
    weights = np.array(weights)
    return np.dot(values, weights) / weights.sum()


def lin_interp(x, x1, y1, x2, y2):
    '''
    Performs linear interpolation between (x1,y1) and (x2,y2) to the point x.
    If x is not between x1 and x2, no worries, we just follow the linear map onwards.
    The map is f(x) = m (x-x1) + y1, where
    m=(y2-y1)/(x2-x1)
    However, if x1==x2, our function is the constant function
    f(x) = (y1+y2)/2
    :param x: The input value to interpolate to
    :param x1: The x coordinate of our first point
    :param y1: The y coordinate of our first point
    :param x2: The x coordinate of our second point
    :param y2: The y coordinate of our second point
    :return: The linearly interpolated value as described above
    '''
    if x1 == x2:
        return (y2+y1)/2
    m = (y2-y1)/(x2-x1)
    #print("x="+str(x)+"interp ("+str(x1)+","+str(y1)+") to ("+str(x2)+","+str(y2)+")")
    return m*(x-x1)+y1


def approx_equal(a, b, per_diff=1e-15) -> bool:
    '''
    Checks if 2 values are approximately equal, but checking
    | a - b | / (avg(|a|, |b|) < per_diff
    note: if a==b==0 then we return True, and do not divide by zero
    :param a: The first value to check
    :param b: The second value to check
    :param per_diff: The maximum percent difference to allow.
    :return: True/False
    '''
    if (a == 0) and (b == 0):
        return True
    return np.abs(b-a)/((np.abs(a)+np.abs(b))/2) <= per_diff


def wtd_median(values, weights):
    '''
    Calculates the weighted median of a set of values.
    :param values: The values to calculated the weighted median of
    :param weights: The weights of the values
    :return: The weighted median
    '''
    pairs = list(zip(values, weights))
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    #print("Sorted Pairs="+str(sorted_pairs))
    total_weight = np.sum(weights)
    half_way = total_weight/2
    if len(values) == 0:
        # Nothing to do
        return 0
    elif len(values) == 1:
        # We have only 1 value
        return values[0]
    #Now we have at least 2 values, we start with first value
    prev_area = 0
    xs_with_cumulative_area_equal = []
    for i in range(len(sorted_pairs)):
        (x, weight) = sorted_pairs[i]
        area = prev_area+weight
        #print("area="+str(area)+" prev_area="+str(prev_area)+" str")
        if approx_equal(area, half_way) and len(xs_with_cumulative_area_equal) == 0:
            # Okay we are equal
            next_x = sorted_pairs[i+1][0]
            xs_with_cumulative_area_equal.append(x+0.5)
        elif area > half_way:
            # We have gone from being below to being above, the median is here
            # Unless we have other places with area equal
            if len(xs_with_cumulative_area_equal) > 0:
                # Okay we need to average the previous equal areas
                xs_with_cumulative_area_equal.append(x-0.5)
                return np.mean(xs_with_cumulative_area_equal)
            # Alright nothing was previously equal, do this
            (prev_x, prev_weight) = sorted_pairs[i-1]
            if i!=(len(sorted_pairs)-1):
                next_x = sorted_pairs[i+1][0]
            else:
                next_x = (x - prev_x) + x
            return lin_interp(half_way, prev_area, x-0.5, area, x+0.5)
        prev_area = area
    return None