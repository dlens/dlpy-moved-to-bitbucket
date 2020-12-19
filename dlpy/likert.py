'''
For bundling numeric values into standard scales
'''
from enum import Enum
from dlpy.maths import pw_linear
from dlpy.percentile import std_perc
from scipy.stats import zscore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class StandardLikert(Enum):
    L = 1
    l = 2
    m = 3
    h = 4
    H = 5

    def __str__(self):
        if self == StandardLikert.L:
            return "Very Low"
        elif self == StandardLikert.l:
            return "Low"
        elif self == StandardLikert.m:
            return "Medium"
        elif self == StandardLikert.h:
            return "High"
        elif self == StandardLikert.H:
            return "Very High"
        else:
            return "Unknown"

    def __repr__(self):
        return str(self)

    def ivalue(self):
        if self == StandardLikert.L:
            return 1
        elif self == StandardLikert.l:
            return 2
        elif self == StandardLikert.m:
            return 3
        elif self == StandardLikert.h:
            return 4
        elif self == StandardLikert.H:
            return 5
        else:
            return "Unknown"

    def value(self):
        if self == StandardLikert.L:
            return 0.0
        elif self == StandardLikert.l:
            return 0.25
        elif self == StandardLikert.m:
            return 0.50
        elif self == StandardLikert.h:
            return 0.75
        elif self == StandardLikert.H:
            return 1.0
        else:
            return "Unknown"

    def inverse(self):
        '''Returns the 6-self version of itself'''
        if self == StandardLikert.L:
            return StandardLikert.H
        elif self == StandardLikert.l:
            return StandardLikert.h
        elif self == StandardLikert.m:
            return self
        elif self == StandardLikert.h:
            return StandardLikert.l
        elif self == StandardLikert.H:
            return StandardLikert.L
        else:
            return "Unknown"


# These are the scales with 1, 2, 3, 4, etc unique items in it.
L=StandardLikert.L
l=StandardLikert.l
m=StandardLikert.m
h=StandardLikert.h
H=StandardLikert.H

SmallLikertScales = [
    [m],
    [l, h],
    [l, m, h],
    [l, m, m, h],
    [L, l, m, h, H],
    [L, l, m, m, h, H],
    [L, l, l, m, h, h, H],
    [L, l, l, m, m, h, h, H],
    [L, L, l, l, m, h, h, H, H],
    [L, L, l, l, m, m, h, h, H, H],
    [L, L, l, l, m, m, m, h, h, H, H],
    [L, L, l, l, l, m, m, h, h, h, H, H],
    [L, L, l, l, l, m, m, m, h, h, h, H, H],
    [L, L, L, l, l, l, m, m, h, h, h, H, H, H],
    [L, L, L, l, l, l, m, m, m, h, h, h, H, H, H]
]


def likert_from_01_grade(score):
    '''
    Returns the likert scale value of the 0 to 1 grade
    :param score:
    :return:
    '''
    if score < 0.2:
        return StandardLikert.L
    elif score < 0.4:
        return StandardLikert.l
    elif score < 0.6:
        return StandardLikert.m
    elif score < 0.8:
        return StandardLikert.h
    else:
        return StandardLikert.H

ZSCORE_CUTOFFS_VALUES = [(-0.84162123, 0.2), (-0.2533471, 0.4) ,  (0.2533471, 0.6) ,  (0.84162123, 0.8)]
ZSCORE_INTERPRETTER = lambda x: pw_linear(x, ZSCORE_CUTOFFS_VALUES, 0, 1);

def likert_using_zscores(values, do_cluster=True, cluster_epsilon=0.01, cluster_delta=0.09):
    '''
    Returns the standard likert evaluation of the values using
    :param values: Any list-like object of numbers
    :return: The likert evaluation of those numbers.  Returns a simple list
    '''
    if do_cluster:
        clustered_values = cluster_values_simple(values, cluster_epsilon, cluster_delta)
    else:
        clustered_values = values
    unique_values = list(set(clustered_values))
    unique_values.sort()
    ix_in_unique = [unique_values.index(value) for value in clustered_values]
    zscores = zscore(unique_values)
    evaled_zscores = [ZSCORE_INTERPRETTER(x) for x in zscores]
    rval = [likert_from_01_grade(evaled_zscores[index]) for index in ix_in_unique]
    return rval

def likert_using_percentile(values, do_cluster=True, cluster_epsilon=0.01, cluster_delta=0.09):
    '''
    Returns the standard likert evaluation of the values using
    :param values: Any list-like object of numbers
    :return: The likert evaluation of those numbers.  Returns a simple list
    '''
    if do_cluster:
        clustered_values = cluster_values_simple(values, cluster_epsilon, cluster_delta)
    else:
        clustered_values = values
    unique_values = list(set(clustered_values))
    unique_values.sort()
    ix_in_unique = [unique_values.index(value) for value in clustered_values]
    percs = [std_perc(unique_values, x) for x in unique_values]
    rval = [likert_from_01_grade(percs[index]) for index in ix_in_unique]
    return rval


def likert_using_small_count(values, cluster_epsilon=0.05, cluster_delta=0.2):
    clustered_values = cluster_values_simple(values, cluster_epsilon, cluster_delta)
    unique_values = list(set(clustered_values))
    if len(unique_values) > len(SmallLikertScales):
        raise Exception("Too many values to use the small_count likert algorithm")
    unique_values.sort()
    ix_in_unique = [unique_values.index(value) for value in clustered_values]
    smallScale = SmallLikertScales[len(unique_values)-1]
    rval = [smallScale[ix_in_unique[i]] for i in range(len(values))]
    return rval

def likert_using_default(values, cluster_epsilon=0.05, cluster_delta=0.2):
    '''
    If the number of values after clustering is small enough that we can use
    the small_count version, we do so, otherwise we use the percentile version
    :param values:
    :param cluster_epsilon:
    :param cluster_delta:
    :return:
    '''
    try:
        return likert_using_small_count(values, cluster_epsilon, cluster_delta)
    except:
        pass
    return likert_using_percentile(values, cluster_epsilon, cluster_delta)

def average_likerts(list_of_likerts)->StandardLikert:
    '''
    Takes the average of a list of likert scores. This is done by
    1. Averaging their 1-5 scores
    2. Converting to an integer 1-5 score by the rule
    If < 1.99: 1 (Very Low)
    elseif < 2.75: 2 (Low)
    elseif < 3.25: 3 (Medium)
    elseif < 4.01: 4 (High)
    else: 5 (Very High)
    :param list_of_likerts:
    :return:
    '''
    scores = [v.ivalue() for v in list_of_likerts]
    mean = np.mean(scores)
    #print(scores)
    #print(mean)
    if mean < 1.99:
        return StandardLikert.L
    elif mean < 2.75:
        return StandardLikert.l
    elif mean < 3.25:
        return StandardLikert.m
    elif mean < 4.01:
        return StandardLikert.h
    else:
        return StandardLikert.H

class ClusterOfNumber:
    def __init__(self, init_value=None, init_index=None):
        if init_value is None:
            self.values=[]
            self.indices=[]
        else:
            self.values=[init_value]
            self.indices=[init_index]

    def center(self):
        return np.mean(self.values)

    def append(self, value, index=None):
        self.values.append(value)
        self.indices.append(index)

    def breakup(self):
        '''
        Returns a list of ClusterOfNumber with each value as a single entry
        :return:
        '''
        return [ClusterOfNumber(self.values[i], self.indices[i]) for i in range(len(self.values))]


    @staticmethod
    def clustered_values_from_clusters(clusters):
        max_index = max([max(cluster.indices) for cluster in clusters])
        rval = [None] * (max_index+1)
        for cluster in clusters:
            for index in cluster.indices:
                rval[index] = cluster.center()
        return rval


def cluster_values_simple(values, epsilon=0.05, delta=0.2):
    clusters = cluster_simple(values, epsilon, delta)
    return ClusterOfNumber.clustered_values_from_clusters(clusters)


def cluster_simple(values, epsilon=0.05, delta=0.2):
    if epsilon < 0:
        raise Exception("Epsilon must be greater than zero")
    if delta < 0:
        raise Exception("delta must be greater than zero")
    if delta <= epsilon:
        raise Exception("delta must be bigger than epsilon")
    ix_sorted = np.argsort(values)
    scale = np.max(values) - np.min(values)
    if scale == 0:
        scale = 1.0
    last_value = None
    currentCluster = ClusterOfNumber()
    rval=[currentCluster]
    for index in ix_sorted:
        val = values[index]
        if last_value is None:
            last_value = val
        if val - last_value < epsilon*scale:
            #We can add this to the cluster, however the cluster might be too big now
            currentCluster.append(val, index);
            if currentCluster.values[-1] - currentCluster.values[0] > delta*scale:
                #The last one in rval is the current, we must remove it
                rval.pop()
                #We have gotten too big, break up into individual clusters
                for ptCluster in currentCluster.breakup():
                    rval.append(ptCluster)
                last_value = None
                currentCluster = ClusterOfNumber()
                rval.append(currentCluster)
            else:
                last_value = val
        else:
            # We have a cluster, add it and get ready to restart
            currentCluster = ClusterOfNumber(val, index)
            rval.append(currentCluster);
            last_value = val
    return rval

def table_likerts(values):
    values.sort()
    likerts_z = likert_using_zscores(values)
    likerts_z_nc = likert_using_zscores(values, do_cluster=False)
    likerts_per = likert_using_percentile(values)
    likerts_per_nc = likert_using_percentile(values, do_cluster=False)
    df = pd.DataFrame(
        {'Values': values, 'ZScore': likerts_z, 'ZScoreNoCluster': likerts_z_nc, 'Percentile': likerts_per,
         'PercentileNoCluster': likerts_per_nc})
    try:
        likerts_small = likert_using_small_count(values)
        df['SmallCount'] = likerts_small
    except:
        pass
    return df


def plot_likert_pt(x, level, val, size=10):
    plot_string = ''
    if val is StandardLikert.L:
        plot_string = 'ro'
    elif val is StandardLikert.l:
        plot_string = 'mX'
    elif val is StandardLikert.m:
        plot_string = 'ys'
    elif val is StandardLikert.h:
        plot_string = 'g^'
    elif val is StandardLikert.H:
        plot_string = 'bX'
    plt.plot(x, level, plot_string, markersize=size)


def plot_likerts(values, figsize=(12, 7), marker_size=10):
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    #plt.figure(figsize=figsize)
    df = table_likerts(values)
    names = df.columns[1:]
    ncols = len(df.columns)
    xs = df.iloc[:, 0]
    for x in xs:
        plt.axvline(x=x, ls='--', color='black')

    for colint in range(1, ncols):
        col = df.iloc[:, colint]
        for x, val in zip(xs, col):
            plot_likert_pt(x, colint, val, marker_size)
    plt.yticks(range(1, len(names) + 1), names)
    legend_elements = [
        Line2D([0], [0], color='w', markerfacecolor='r', marker='o', label='Very Low', markersize=10),
        Line2D([0], [0], color='w', markerfacecolor='m', marker='X', label='Low', markersize=10),
        Line2D([0], [0], color='w', markerfacecolor='y', marker='s', label='Medium', markersize=10),
        Line2D([0], [0], color='w', markerfacecolor='g', marker='^', label='High', markersize=10),
        Line2D([0], [0], color='w', markerfacecolor='b', marker='X', label='Very High', markersize=10)
    ]

    ax.legend(handles=legend_elements, loc='center')

