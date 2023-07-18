import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


"""
    Goes through the data, imputing on rows that can be calculated with certainty.
    It converts spells to a more consistent and machine readable format and removes rows
    that cannot be imputed on due to significance of missing data. Finally, it alters
    the column structure of the data set and outputs this new result as a png file
"""
def initial_impute_clean_data():

    # Data wrangling
    clean_regions_df = pd.read_csv('CSVFILES/ALL_REGIONS.csv')

    # Numerical variables are all floats, might cause precision issues
    for column in clean_regions_df:
        if column in ['champion', 'side', 'role', 'minions_killed', 'region', 'kda']:
            continue
        for row in clean_regions_df[column].dropna():
            if round(row) != row:
                print(row)
                print(column)
    # no columns printed so it looks like its fine

    # variables that can be exactly calculated using other variables in the dataset
    kda_check = ['kda', 'assists', 'deaths', 'kills']
    objectives_check = ['damage_objectives', 'damage_turrets']

    # Check for duplicates, highly unlikely so most likely wouldn't exist naturally
    print(f'\nCurrent number of duplicates: {clean_regions_df.duplicated().sum()}\n')
    # Outputs the number of missing values for kda_check and objecitve_check columns

    print(f"Before imputation (Number of Empty Cells): \n{clean_regions_df[kda_check + objectives_check].isna().sum()}\n")
    print('Imputing...\n')

    # Iterates through every row in the dataframe
    for index, row in clean_regions_df.iterrows():
        
        # imputes damage_objectives from damage_turrets and vice versa if possible
        if row[objectives_check].isna().tolist().count(True) == 1:
            if row.isna()['damage_objectives']:
                clean_regions_df.at[index, 'damage_objectives'] = clean_regions_df.at[index, 'damage_turrets']
            elif row.isna()['damage_turrets']:
                clean_regions_df.at[index, 'damage_turrets'] = clean_regions_df.at[index, 'damage_objectives']

        # calculates kda_check variables using other values if possible
        if row[kda_check].isna().tolist().count(True) == 1:
            if row.isna()['kda']:
                deaths = row['deaths'] if row['deaths'] else 1 
                clean_regions_df.at[index, 'kda'] = (row['assists'] + row['kills']) / deaths
            elif row.isna()['assists']:
                clean_regions_df.at[index, 'assists'] = row['kda'] * row['deaths'] - row['kills']
            elif row.isna()['kills']:
                clean_regions_df.at[index, 'kills'] = row['kda'] * row['deaths'] - row['assists']
            elif row.isna()['deaths'] and row['kda'] != 0:
                clean_regions_df.at[index, 'deaths'] = (row['assists'] + row['kills']) / row['kda']

    # Outputs the number of missing values for kda_check and objecitve_check columns
    print(f"After imputation (Number of Empty Cells): \n{clean_regions_df[kda_check + objectives_check].isna().sum()}\n")

    # You can see this has caused precision issues
    list_kills = clean_regions_df['kills'].dropna().unique().tolist()
    list_int_kills = clean_regions_df['kills'].dropna().round(0).unique().tolist()

    """ Unquote and reformate to look at precision errors in 'kills'
    ----------------------------------------------------------------------------
    print(f'before rounding: {len(list_kills)} unique values in \'kills\'')
    print(f'after rounding: {len(list_int_kills)} unique values in \'kills\'\n')
    ----------------------------------------------------------------------------
    """

    print('Rounding Columns (preventing precision errors)...\n')
    # Just round everything to be safe
    clean_regions_df = clean_regions_df.round({'d_spell': 0, 'f_spell': 0, 'assists': 0, 
        'damage_objectives': 0, 'damage_building': 0, 'damage_turrets': 0, 'deaths': 0,
        'kills': 0, 'level': 0, 'time_cc': 0, 'damage_taken': 0, 'turret_kills': 0, 'vision_score': 0})

    # Outputs the counts of empty cells left in the data set
    print(f"Empty Cells remaining: \n{clean_regions_df.isna().sum()}\n")

    # Make spells columns machine readable by grouping d_spell and f_spell into one format
    print("Converting spells to machine readable code...")
    spells = ['d_spell','f_spell']
    for index, row in clean_regions_df.iterrows():
        summoner_spells = row[spells]

        if (summoner_spells.isna().sum() > 0):
            list_spells = np.nan # indicating at least 1 summoner spell is missing
            continue

        # sorts the two spells by the number closest to 4 on the right
        list_spells = sorted(list(summoner_spells.array), key=lambda x: abs(x - 4.0), reverse=True)
        clean_regions_df.at[index, 'summoner_spells'] = str(list_spells)

    # Distribution of summoner_spells before condensing
    print('Calculating distribution of summoner_spells before consdensing...')
    fig = plt.figure(figsize=(15,20))
    ax = plt.gca()
    ax.set_ylabel('Count', fontsize=20)
    ax.set_xlabel('Summoner Spells', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax = clean_regions_df['summoner_spells'].value_counts().plot.bar()
    fig.suptitle('Distribution of summoner_spells pre-condensing', fontsize=20)
    print('Saving distribution plot as png...')
    fig.savefig('Graphs/pre_summoner_spells.png')
    plt.close(fig)
    
    print('Condensing...')
    # We decided that some spell combos were so uncommon they would be too hard to predict
    # produces a data series of the proportion of each spell combo in the dataset
    spell_ratio = clean_regions_df['summoner_spells'].value_counts()/len(clean_regions_df)
    THRESHOLD = 0.025
    accepted_spells = []

    # if the proportion is > THRESHOLD add to accepted_spells
    for row in spell_ratio.iteritems():
        RATIO = 1
        SPELL_COMBO = 0
        if row[RATIO] >= THRESHOLD:
            accepted_spells.append(row[SPELL_COMBO])

    # replace all non-accepted_spells with 'Other'
    for index, row in clean_regions_df.iterrows():
        summoner_spell = row['summoner_spells']
        if summoner_spell not in accepted_spells and not(type(summoner_spell) == float):
            clean_regions_df.at[index, 'summoner_spells'] = 'Other'

    # Distribution of summoner_spells after condensing...
    print('Calculating distribution of summoner_spells after condensing...')
    fig = plt.figure(figsize=(15,20))
    ax = plt.gca()
    ax = clean_regions_df['summoner_spells'].value_counts().plot.bar()
    ax.set_ylabel('Count', fontsize=20)
    ax.set_xlabel('Summoner Spells', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.suptitle('Distribution of summoner_spells post-condensing', fontsize=20)
    print('Saving distribution plot as png...\n')
    fig.savefig('Graphs/post_summoner_spells.png')
    plt.close(fig)

    # Remove unnecessary columms from data set
    for column in ['side', 'd_spell', 'f_spell', 'damage_turrets']:
        print(f'Removing column ({column})...')
        clean_regions_df.drop(columns=[column], inplace=True)
    
    # Drop rows with missing champion values as it's unrealistic to impute
    print('\nDropping rows with nan values in column (champion)...')
    clean_regions_df.dropna(subset = ['champion'], inplace=True)
    
    # Drop rows with missing summoner_spells values as it's our target variable
    print('Dropping rows with nan values in column (summoner_spells)...\n')
    clean_regions_df.dropna(subset = ['summoner_spells'], inplace=True)

    # Drops duplicates
    print(f'Number of duplicates in Data Set: {clean_regions_df.duplicated().sum()}')
    print('Removing duplicates...\n')
    clean_regions_df.drop_duplicates(inplace=True)

    # Final result of imputing and cleaning data
    print(f"Empty Cells remaining: \n{clean_regions_df.isna().sum()}\n")
    clean_regions_df.to_csv('CSVFILES/CLEAN_REGION_DATA.csv', index=False)

    return
