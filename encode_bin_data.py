import pandas as pd
import numpy as np
import math
import json

from matplotlib import pyplot as plt


"""
    Calculates bins for a column and changes the column values to their
    binned equivalents
"""
def bin_quantity(column): 
    
    COLUMN_MAX = column.max()
    COLUMN_MIN = column.min()

    LOWER = ((COLUMN_MAX - COLUMN_MIN) * (1/3)) + COLUMN_MIN
    UPPER = ((COLUMN_MAX - COLUMN_MIN) * (2/3)) + COLUMN_MIN

    for index, row in column.iteritems():
        if not math.isnan(row):
            if row < LOWER:
                bound = 0
            elif row >= LOWER and row < UPPER:
                bound = 1
            else:
                bound = 2
            column.at[index] = bound
        
    return column # -> outputs a series

"""
    Encodes and bins the altered data set, which will later be used by the
    supervised ML model to determine an appropriate learning model
"""
def encode_bin_data():
    clean_regions_df = pd.read_csv('CSVFILES/CLEAN_REGION_DATA.csv')

    # Encoding summoner spells
    print('Creating encode dictionaries for nominal and discrete data...')
    accepted_spells = list(clean_regions_df['summoner_spells'].value_counts().index)
    # Out of the 8 possible spells, we want 'Other to be last
    accepted_spells.remove('Other') 
    accepted_spells.append('Other')

    summoner_spell_encode = {}
    i = 0
    for spell_combo in accepted_spells:
        try:
            summoner_spell_encode[spell_combo]
        except KeyError:
            summoner_spell_encode[spell_combo] = i
        i += 1

    # Encoding Champions
    list_champion = clean_regions_df['champion'].unique().tolist()
    list_champion = sorted([i for i in list_champion if not (type(i) == float)])
    
    i = 0 
    champion_encode = {}
    for champion in list_champion:
        try:
            champion_encode[champion]
        except KeyError:
            champion_encode[champion] = i
        i += 1
    
    # Encoding minions_killed, role and region
    minion_encode = {'Few' : 0,
                    'Many' : 1}

    role_encode = {'TopLane_Jungle' : 0,
                   'Other' : 1}
                
    region_encode = {'region.KR' : 0,
                     'region.NA' : 1,
                     'region.EU': 2}

    # Outputting BINNED_DATA_ENCODES.json
    print('Outputting encoding dictionary...')
    encodes = {'summoner_spell_encode' : summoner_spell_encode,
               'champion_encode' : champion_encode,
               'minion_encode' : minion_encode,
               'role_encode' : role_encode,
               'region_encode' : region_encode}

    with open("CSVFILES/BINNED_DATA_ENCODES.json", "w") as fp:
        json.dump(encodes, fp, indent=4)

    
    # Begin Encoding dataset
    print('Encoding nominal and discrete variables...')
    binned_data = clean_regions_df.copy()
    
    for index, row in binned_data.iterrows():
        column_names  = ['champion', 'role', 'minions_killed', 'summoner_spells', 'region']
        encode = [champion_encode, role_encode, minion_encode, summoner_spell_encode, region_encode]

        for column_index in range(len(column_names)):
            if not (type(row[column_names[column_index]]) == float):
                binned_data.at[index, column_names[column_index]] = encode[column_index][row[column_names[column_index]]]   

    # Bins all continous variables using decided method of 3 uniform bins
    bin_features = ['binned_damage_objectives', 'binned_damage_building',
        'binned_damage_taken', 'binned_damage_total', 'binned_gold_earned',
        'binned_kda', 'binned_level', 'binned_kills', 'binned_deaths', 
        'binned_assists', 'binned_time_cc', 'binned_vision_score']

    print('Binning continuous data...')
    for feature in bin_features:
        binned_data[feature] = bin_quantity(binned_data[feature[7:]].copy())

    # Output binned_data to a csv file
    print('Outputting BINNED_DATA.csv...')
    binned_data.to_csv('CSVFILES/BINNED_DATA.csv', index=False)

    """
    # Unquote this to see what the distributions
    # of the bins look like 

    for i in bin_features:
        print(binned_data[i].value_counts())
    """

    return

