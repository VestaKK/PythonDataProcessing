import pandas as pd
import numpy as np
import csv

from matplotlib import pyplot as plt

"""
    A function which computes the Pearson Correlation between two features
    and also graphs it cause that makes the data look really cool
"""
def pearson_r(feature_a, feature_b, graph=False):
    
    df = pd.DataFrame({feature_a.name: feature_a, feature_b.name: feature_b}) 
    df.dropna(inplace=True)

    feature_a = df[feature_a.name]
    feature_b = df[feature_b.name]

    # compute the mean
    mean_a = feature_a.mean()
    mean_b = feature_b.mean()
    
    # compute the numerator of pearson r
    numerator = sum((feature_a - mean_a) * (feature_b - mean_b))
    
    # compute the denominator of pearson r
    denominator = np.sqrt(sum((feature_a - mean_a) ** 2) * sum((feature_b - mean_b) ** 2))

    # For graphing
    if graph:
        plt.scatter(x=feature_a, y=feature_b, alpha=0.01)
        plt.title(f'{feature_a.name.upper()} vs {feature_b.name.upper()}')
        plt.savefig(f'Scatter_plots/{feature_a.name}-{feature_b.name}.png')
        plt.close()

    return numerator/denominator
    

"""
    Finds all possible combinations of scatter plots, plots them, and \
    outputs to a png file
"""    
def scatter_plots():
    continuous_variables = ['level',
                        'damage_total', 
                        'vision_score', 
                        'gold_earned', 
                        'kills',
                        'assists', 
                        'kda', 
                        'deaths', 
                        'damage_building', 
                        'time_cc', 
                        'damage_taken', 
                        'damage_turrets', 
                        'damage_objectives', 
                        'turret_kills']

    all_regions = pd.read_csv('CSVFILES/ALL_REGIONS.csv')
    all_regions.loc[:, continuous_variables]
    
    # Finding all combinations of data set features and
    # graphs each one
    for i in range(len(continuous_variables) - 1):
        for j in range(len(continuous_variables) - 1 - i):
            column1 = all_regions[continuous_variables[i]]
            column2 = all_regions[continuous_variables[j + 1 + i]]
            pearson_r(column1, column2, True)
    return

